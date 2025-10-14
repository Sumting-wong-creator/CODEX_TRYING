const contentPort = chrome.runtime.connect({ name: 'content' });
let allowPageInstructions = false;
let agentOverlay = null;
let agentActive = false;
let agentSessionId = null;

contentPort.onDisconnect.addListener(() => {
  console.debug('[HAWA][content] Port disconnected');
});

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (!message) return;
  switch (message.type) {
    case 'read-page':
      allowPageInstructions = Boolean(message.args?.allowInstructions);
      readPage().then(payload => {
        contentPort.postMessage({ type: 'read-page-ready', payload });
        sendResponse && sendResponse({ ok: true, payload });
      }).catch(error => {
        contentPort.postMessage({ type: 'read-page-error', error: error.message });
        sendResponse && sendResponse({ ok: false, error: error.message });
      });
      return true;
    case 'execute-tool':
      executeTool(message.tool, message.args || {}, message.toolCallId)
        .then(result => {
          contentPort.postMessage({ type: 'tool-result', toolCallId: message.toolCallId, result });
        })
        .catch(error => {
          contentPort.postMessage({ type: 'tool-error', toolCallId: message.toolCallId, error: error.message });
        });
      break;
    case 'agent-overlay-show':
      showAgentOverlay(message.sessionId);
      break;
    case 'agent-overlay-hide':
      hideAgentOverlay();
      break;
    case 'agent-overlay-status':
      if (agentOverlay) {
        agentOverlay.querySelector('[data-status]').textContent = message.status || '';
      }
      break;
    case 'inject-topbar':
      injectTopbar(message.summary || '');
      break;
    default:
      break;
  }
});

async function readPage() {
  const selection = window.getSelection?.()?.toString() || '';
  const headings = Array.from(document.querySelectorAll('h1, h2, h3')).slice(0, 12).map(el => collapseWhitespace(el.textContent));
  const priceCandidates = findPrices();
  const instructions = allowPageInstructions ? collectInstructionBlocks() : [];
  return {
    url: location.href,
    title: document.title,
    selection: collapseWhitespace(selection),
    headings,
    priceCandidates,
    instructions,
    timestamp: Date.now()
  };
}

function findPrices() {
  const textContent = document.body?.innerText || '';
  const regex = /(₪\s?\d+(?:[,.]\d+)?|\$\s?\d+(?:[,.]\d+)?|חינם)/g;
  const matches = new Set();
  let match;
  while ((match = regex.exec(textContent)) !== null) {
    matches.add(match[0]);
    if (matches.size >= 8) break;
  }
  return Array.from(matches);
}

function collectInstructionBlocks() {
  const elements = Array.from(document.querySelectorAll('[data-instruction], script[type="application/json"], meta[name*="instruction" i]'));
  return elements.slice(0, 10).map(el => {
    if (el.dataset?.instruction) return collapseWhitespace(el.dataset.instruction);
    if (el.tagName === 'SCRIPT') return collapseWhitespace(el.textContent);
    if (el.tagName === 'META') return collapseWhitespace(el.getAttribute('content') || '');
    return collapseWhitespace(el.textContent || '');
  }).filter(Boolean);
}

async function executeTool(tool, args, toolCallId) {
  switch (tool) {
    case 'click':
      return handleClick(args, toolCallId);
    case 'type':
      return handleType(args);
    case 'scrollTo':
      return handleScroll(args);
    default:
      throw new Error(`Unsupported tool: ${tool}`);
  }
}

async function handleClick(args, toolCallId) {
  const element = locateElement(args);
  if (!element) {
    throw new Error('No matching element to click.');
  }
  if (requiresConfirmation(args)) {
    const confirmed = await confirmStep(args.intent || 'Are you sure?');
    if (!confirmed) {
      throw new Error('User cancelled action.');
    }
  }
  element.focus({ preventScroll: true });
  element.click();
  return { clicked: true };
}

async function handleType(args) {
  const element = locateElement(args);
  if (!element) {
    throw new Error('No matching element to type into.');
  }
  const value = String(args.value ?? '');
  element.focus({ preventScroll: false });
  if (args.replace || element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
    element.value = value;
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
  } else {
    element.textContent = value;
  }
  return { typed: true };
}

async function handleScroll(args) {
  if (args.selector) {
    const element = document.querySelector(args.selector);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      return { scrolled: true };
    }
  }
  if (typeof args.position === 'string') {
    if (args.position === 'top') window.scrollTo({ top: 0, behavior: 'smooth' });
    if (args.position === 'bottom') window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    return { scrolled: true };
  }
  window.scrollBy({ top: 200, behavior: 'smooth' });
  return { scrolled: true };
}

function locateElement(args) {
  if (args.selector) {
    const found = document.querySelector(args.selector);
    if (found) return found;
  }
  if (args.role) {
    const candidate = document.querySelector(`[role="${CSS.escape(args.role)}"]`);
    if (candidate) return candidate;
  }
  if (args.text) {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT, null);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (node.childElementCount > 0) continue;
      if (collapseWhitespace(node.textContent || '').toLowerCase().includes(args.text.toLowerCase())) {
        return node.parentElement || node;
      }
    }
  }
  return null;
}

function requiresConfirmation(intent) {
  if (!intent) return false;
  const keywords = ['checkout', 'purchase', 'submit', 'order', 'pay'];
  return keywords.some(keyword => intent.toLowerCase().includes(keyword));
}

function confirmStep(intentText) {
  return new Promise(resolve => {
    const modal = document.createElement('div');
    modal.className = 'hawa-confirm-modal';
    modal.innerHTML = `
      <div class="hawa-confirm-dialog" role="dialog" aria-modal="true">
        <h2>Confirm action</h2>
        <p>${escapeHtml(intentText)}</p>
        <div class="hawa-confirm-actions">
          <button type="button" data-action="cancel">Cancel</button>
          <button type="button" data-action="confirm">Proceed</button>
        </div>
      </div>`;
    document.body.appendChild(modal);
    const handler = (event) => {
      const action = event.target?.dataset?.action;
      if (!action) return;
      event.preventDefault();
      modal.remove();
      resolve(action === 'confirm');
      modal.removeEventListener('click', handler);
    };
    modal.addEventListener('click', handler);
  });
}

function showAgentOverlay(sessionId) {
  agentActive = true;
  agentSessionId = sessionId;
  if (agentOverlay) return;
  agentOverlay = document.createElement('div');
  agentOverlay.className = 'hawa-agent-overlay';
  agentOverlay.innerHTML = `
    <div class="hawa-agent-frame" role="status" aria-live="polite">
      <div class="hawa-agent-header">
        <span class="hawa-agent-name">HAWA Agent</span>
        <button type="button" class="hawa-agent-stop" title="Stop agent" aria-label="Stop agent">E STOP</button>
      </div>
      <div class="hawa-agent-status" data-status>Preparing instructions…</div>
    </div>`;
  document.documentElement.appendChild(agentOverlay);
  agentOverlay.querySelector('.hawa-agent-stop').addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'agent-stop-request', sessionId: agentSessionId });
  });
}

function hideAgentOverlay() {
  agentActive = false;
  agentSessionId = null;
  if (agentOverlay) {
    agentOverlay.remove();
    agentOverlay = null;
  }
}

function injectTopbar(summary) {
  if (document.getElementById('hawa-topbar')) return;
  fetch(chrome.runtime.getURL('topbar.html')).then(res => res.text()).then(html => {
    const container = document.createElement('div');
    container.innerHTML = html;
    const topbar = container.firstElementChild;
    topbar.id = 'hawa-topbar';
    document.body.appendChild(topbar);
    const shadow = topbar.attachShadow({ mode: 'open' });
    Promise.all([
      fetch(chrome.runtime.getURL('topbar.css')).then(r => r.text()),
      fetch(chrome.runtime.getURL('topbar.js')).then(r => r.text())
    ]).then(([cssText, jsText]) => {
      const style = document.createElement('style');
      style.textContent = cssText;
      shadow.appendChild(style);
      const content = document.createElement('div');
      content.innerHTML = summary;
      shadow.appendChild(content);
      const script = document.createElement('script');
      script.textContent = jsText;
      shadow.appendChild(script);
    });
  }).catch(error => console.error('HAWA topbar injection failed', error));
}

function collapseWhitespace(value) {
  return (value || '').replace(/\s+/g, ' ').trim();
}

function escapeHtml(value) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const confirmStyles = document.createElement('style');
confirmStyles.textContent = `
.hawa-confirm-modal { position: fixed; inset: 0; background: rgba(12, 12, 14, 0.55); backdrop-filter: blur(4px); display: grid; place-items: center; z-index: 2147483647; font-family: 'Segoe UI', system-ui, sans-serif; }
.hawa-confirm-dialog { width: min(320px, calc(100% - 2rem)); background: color-mix(in srgb, var(--ha-surface, #101015) 85%, transparent); padding: 1.25rem; border-radius: 16px; box-shadow: 0 24px 48px rgba(0,0,0,0.35); color: var(--ha-foreground, #f5f6fb); border: 1px solid rgba(255,255,255,0.08); }
.hawa-confirm-dialog h2 { margin: 0 0 0.5rem; font-size: 1.1rem; }
.hawa-confirm-dialog p { margin: 0 0 1rem; font-size: 0.95rem; opacity: 0.85; }
.hawa-confirm-actions { display: flex; gap: 0.75rem; justify-content: flex-end; }
.hawa-confirm-actions button { padding: 0.45rem 0.9rem; border-radius: 999px; border: none; font-weight: 600; cursor: pointer; }
.hawa-confirm-actions [data-action="cancel"] { background: rgba(255,255,255,0.08); color: inherit; }
.hawa-confirm-actions [data-action="confirm"] { background: linear-gradient(135deg, #6750ff, #b388ff); color: white; }
.hawa-agent-overlay { position: fixed; inset: 0; pointer-events: none; background: radial-gradient(circle at top left, rgba(143, 129, 255, 0.2), transparent 55%), radial-gradient(circle at bottom right, rgba(90, 248, 255, 0.18), transparent 60%); backdrop-filter: blur(2px); z-index: 2147483600; animation: hawaPulse 6s ease-in-out infinite; }
.hawa-agent-frame { position: absolute; top: 24px; left: 24px; padding: 1rem; border-radius: 18px; background: rgba(16, 15, 35, 0.65); border: 1px solid rgba(149, 128, 255, 0.45); color: #f4f3ff; min-width: 220px; pointer-events: auto; box-shadow: 0 18px 36px rgba(48, 32, 128, 0.4); }
.hawa-agent-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem; gap: 0.75rem; }
.hawa-agent-name { font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; font-size: 0.8rem; opacity: 0.9; }
.hawa-agent-stop { border: none; border-radius: 999px; padding: 0.35rem 0.85rem; background: #ff2f4c; color: white; font-weight: 700; cursor: pointer; letter-spacing: 0.08em; box-shadow: 0 6px 16px rgba(255, 47, 76, 0.45); }
.hawa-agent-status { font-size: 0.85rem; opacity: 0.8; }
@keyframes hawaPulse { 0%, 100% { opacity: 0.65; } 50% { opacity: 1; } }
@media (prefers-color-scheme: light) {
  .hawa-confirm-modal { background: rgba(248, 248, 252, 0.55); }
  .hawa-confirm-dialog { background: rgba(255, 255, 255, 0.95); color: #1b1b21; border-color: rgba(103, 80, 255, 0.2); }
  .hawa-confirm-actions [data-action="cancel"] { background: rgba(103, 80, 255, 0.08); color: #1b1b21; }
  .hawa-agent-overlay { background: radial-gradient(circle at top left, rgba(103, 80, 255, 0.14), transparent 50%), radial-gradient(circle at bottom right, rgba(0, 180, 216, 0.12), transparent 55%); }
  .hawa-agent-frame { background: rgba(248, 246, 255, 0.92); color: #1d1735; border-color: rgba(103, 80, 255, 0.35); }
  .hawa-agent-stop { box-shadow: 0 6px 16px rgba(255, 47, 76, 0.35); }
}
`;
document.documentElement.appendChild(confirmStyles);

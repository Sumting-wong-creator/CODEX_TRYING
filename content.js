const port = chrome.runtime.connect({ name: 'content' });

const TOOL_TIMEOUT = 15000;
let topbarFrame = null;
let agentOverlayEl = null;
let agentOverlayStyleEl = null;
let agentOverlayActive = false;

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message) return;
  if (message.type === 'read-page') {
    handleReadPage(message.args || {})
      .then(payload => {
        port.postMessage({ type: 'read-page-ready', payload });
      })
      .catch(error => {
        port.postMessage({ type: 'read-page-error', error: error.message });
      });
    sendResponse({ ok: true });
    return true;
  }
  if (message.type === 'execute-tool') {
    executeTool(message.tool, message.args || {}, message.toolCallId).catch(error => {
      port.postMessage({ type: 'tool-error', toolCallId: message.toolCallId, error: error.message });
    });
    sendResponse({ ok: true });
    return true;
  }
  if (message.type === 'inject-topbar') {
    renderTopbar(message.summary || {});
    sendResponse({ ok: true });
  }
  if (message.type === 'sidebar-open-request') {
    chrome.runtime.sendMessage({ type: 'open-sidebar' });
    sendResponse({ ok: true });
  }
  if (message.type === 'agent-overlay') {
    if (message.active) {
      showAgentOverlay(message);
    } else {
      hideAgentOverlay();
    }
    sendResponse({ ok: true });
  }
});

async function handleReadPage(args) {
  const selection = getSelectionText();
  const headings = Array.from(document.querySelectorAll('h1, h2, h3')).map(el => ({
    tag: el.tagName,
    text: el.textContent.trim().slice(0, 200)
  }));
  const language = document.documentElement.getAttribute('lang') || navigator.language || '';
  const instructions = collectPageInstructions(args.allowInstructions);
  const priceCandidates = detectPrices();
  const meta = {
    description: getMetaContent('description'),
    ogTitle: getMetaContent('og:title'),
    ogDescription: getMetaContent('og:description')
  };
  return {
    url: location.href,
    title: document.title,
    selection,
    headings,
    instructions,
    priceCandidates,
    language,
    meta
  };
}

function getSelectionText() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return '';
  return selection.toString().trim().slice(0, 4000);
}

function collectPageInstructions(allowInstructions) {
  const nodes = [];
  const metaTags = document.querySelectorAll('meta[name*="instruction" i], meta[name*="prompt" i], meta[name*="directive" i]');
  metaTags.forEach(tag => {
    const content = tag.getAttribute('content');
    if (content) nodes.push(content.trim());
  });
  const dataAttr = document.querySelectorAll('[data-agent-instructions], [data-ai-instructions]');
  dataAttr.forEach(el => {
    const txt = el.getAttribute('data-agent-instructions') || el.getAttribute('data-ai-instructions');
    if (txt) nodes.push(txt.trim());
  });
  const sanitized = nodes.filter(Boolean).map(s => s.slice(0, 2000));
  if (allowInstructions) {
    return sanitized;
  }
  return sanitized.filter(item => !isPromptInjection(item));
}

function isPromptInjection(text) {
  const lowered = text.toLowerCase();
  return /ignore (all|any|previous)/.test(lowered) ||
    /(disable|turn off) (safety|guard)/.test(lowered) ||
    /(forget|wipe) (instructions|memory)/.test(lowered) ||
    /click\s+allow/.test(lowered);
}

function detectPrices() {
  const priceRegex = /(₪\s?\d+[\d,.]*|\$\s?\d+[\d,.]*|חינם)/g;
  const matches = [];
  const walker = document.createTreeWalker(document.body || document.documentElement, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      if (!node.nodeValue) return NodeFilter.FILTER_REJECT;
      if (node.nodeValue.trim().length < 2) return NodeFilter.FILTER_REJECT;
      priceRegex.lastIndex = 0;
      if (!priceRegex.test(node.nodeValue)) return NodeFilter.FILTER_SKIP;
      return NodeFilter.FILTER_ACCEPT;
    }
  });
  let current;
  while ((current = walker.nextNode())) {
    const text = current.nodeValue.trim();
    const context = current.parentElement ? current.parentElement.innerText.trim().slice(0, 200) : text;
    const found = text.match(priceRegex) || [];
    found.forEach(price => {
      matches.push({ price, context });
    });
  }
  return matches.slice(0, 20);
}

function getMetaContent(name) {
  const el = document.querySelector(`meta[name="${name}"]`) || document.querySelector(`meta[property="${name}"]`);
  return el ? (el.getAttribute('content') || '').trim() : '';
}

async function executeTool(tool, args, toolCallId) {
  try {
    let result;
    switch (tool) {
      case 'click':
        result = await performClick(args);
        break;
      case 'type':
        result = await performType(args);
        break;
      case 'scrollTo':
        result = await performScroll(args);
        break;
      default:
        throw new Error(`Unsupported tool ${tool}`);
    }
    port.postMessage({ type: 'tool-result', toolCallId, result });
  } catch (error) {
    port.postMessage({ type: 'tool-error', toolCallId, error: error.message });
  }
}

function showAgentOverlay({ conversationId }) {
  if (agentOverlayActive) {
    updateAgentOverlay(conversationId);
    return;
  }
  ensureAgentOverlayStyles();
  agentOverlayEl = document.createElement('div');
  agentOverlayEl.className = 'hawa-agent-overlay';
  agentOverlayEl.innerHTML = `
    <div class="hawa-agent-top">
      <button type="button" class="hawa-agent-stop" title="Stop agent task" aria-label="Emergency stop">E STOP</button>
      <div class="hawa-agent-status" role="status">Agent mode active &mdash; HAWA is working</div>
    </div>
  `;
  const stopBtn = agentOverlayEl.querySelector('.hawa-agent-stop');
  stopBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ type: 'agent-estop' });
    hideAgentOverlay();
  });
  document.documentElement.appendChild(agentOverlayEl);
  updateAgentOverlay(conversationId);
  agentOverlayActive = true;
}

function updateAgentOverlay(conversationId) {
  if (!agentOverlayEl) return;
  const status = agentOverlayEl.querySelector('.hawa-agent-status');
  if (status) {
    status.textContent = conversationId
      ? `Agent mode active — session ${conversationId.slice(0, 8)}`
      : 'Agent mode active — HAWA is working';
  }
}

function hideAgentOverlay() {
  agentOverlayActive = false;
  if (agentOverlayEl) {
    agentOverlayEl.remove();
    agentOverlayEl = null;
  }
}

function ensureAgentOverlayStyles() {
  if (agentOverlayStyleEl) return;
  const styles = document.createElement('style');
  styles.id = 'hawa-agent-overlay-styles';
  styles.textContent = `
    .hawa-agent-overlay {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 2147483646;
      animation: hawa-overlay-fade 320ms ease;
    }
    .hawa-agent-overlay::before,
    .hawa-agent-overlay::after {
      content: '';
      position: absolute;
      top: 0;
      bottom: 0;
      width: 140px;
      background: linear-gradient(180deg, rgba(109, 91, 255, 0.32), rgba(109, 91, 255, 0.08));
      filter: blur(30px);
      opacity: 0.8;
      animation: hawa-overlay-glow 3.6s ease-in-out infinite alternate;
    }
    .hawa-agent-overlay::before { left: 0; }
    .hawa-agent-overlay::after { right: 0; }
    .hawa-agent-top {
      position: absolute;
      top: 20px;
      left: 20px;
      display: flex;
      align-items: center;
      gap: 18px;
      pointer-events: auto;
      font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    }
    .hawa-agent-stop {
      background: radial-gradient(circle at 30% 30%, #ff7b7b, #d72638);
      color: #ffffff;
      border: none;
      border-radius: 999px;
      padding: 10px 22px;
      font-weight: 700;
      letter-spacing: 0.25em;
      text-transform: uppercase;
      box-shadow: 0 16px 32px rgba(215, 38, 56, 0.45);
      cursor: pointer;
      transition: transform 160ms ease, box-shadow 160ms ease;
    }
    .hawa-agent-stop:hover {
      transform: translateY(-2px) scale(1.03);
      box-shadow: 0 24px 40px rgba(215, 38, 56, 0.55);
    }
    .hawa-agent-status {
      pointer-events: auto;
      padding: 10px 18px;
      border-radius: 999px;
      font-size: 0.75rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      background: rgba(109, 91, 255, 0.3);
      color: #fff;
      border: 1px solid rgba(255, 255, 255, 0.4);
      backdrop-filter: blur(14px);
    }
    @media (prefers-color-scheme: light) {
      .hawa-agent-status {
        color: #1b1f2f;
        border-color: rgba(109, 91, 255, 0.45);
        background: rgba(109, 91, 255, 0.18);
      }
    }
    @keyframes hawa-overlay-glow {
      from {
        opacity: 0.6;
        transform: scaleX(0.98);
      }
      to {
        opacity: 0.95;
        transform: scaleX(1.06);
      }
    }
    @keyframes hawa-overlay-fade {
      from {
        opacity: 0;
      }
      to {
        opacity: 1;
      }
    }
  `;
  (document.head || document.documentElement).appendChild(styles);
  agentOverlayStyleEl = styles;
}

function resolveElement({ selector, text, role }) {
  if (selector) {
    const el = document.querySelector(selector);
    if (el) return el;
  }
  if (role) {
    const byRole = document.querySelectorAll(`[role="${role}"]`);
    for (const candidate of byRole) {
      if (!text || candidate.innerText.trim().toLowerCase().includes(text.toLowerCase())) {
        return candidate;
      }
    }
  }
  if (text) {
    const candidates = Array.from(document.querySelectorAll('button, a, [role], input[type="submit"], input[type="button"], summary'));
    for (const candidate of candidates) {
      const compare = (candidate.innerText || candidate.value || '').trim().toLowerCase();
      if (compare && compare.includes(text.toLowerCase())) {
        return candidate;
      }
    }
    const xpath = document.evaluate(`//*[contains(translate(normalize-space(text()), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), ${JSON.stringify(text.toLowerCase())})]`, document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
    if (xpath.snapshotLength > 0) {
      return xpath.snapshotItem(0);
    }
  }
  return null;
}

async function performClick(args) {
  const el = resolveElement(args);
  if (!el) {
    throw new Error('Clickable element not found.');
  }
  if (!isElementVisible(el)) {
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    await waitFor(300);
  }
  const requiresConfirm = shouldConfirm(el, args.intent);
  if (requiresConfirm) {
    const proceed = await showConfirmModal(el, args.intent || 'important action');
    if (!proceed) {
      throw new Error('Action cancelled by user.');
    }
  }
  el.click();
  return { status: 'clicked', selector: getElementSelector(el) };
}

async function performType(args) {
  const el = resolveElement(args);
  if (!el) {
    throw new Error('Input element not found.');
  }
  if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement || el.isContentEditable)) {
    throw new Error('Target is not an editable field.');
  }
  if (!isElementVisible(el)) {
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    await waitFor(200);
  }
  if (args.replace || !el.value) {
    if (el.isContentEditable) {
      el.textContent = args.value;
    } else {
      el.value = args.value;
    }
  } else {
    if (el.isContentEditable) {
      el.textContent += args.value;
    } else {
      el.value += args.value;
    }
  }
  el.dispatchEvent(new Event('input', { bubbles: true }));
  el.dispatchEvent(new Event('change', { bubbles: true }));
  return { status: 'typed', valueLength: args.value.length };
}

async function performScroll(args) {
  if (args.position === 'top') {
    window.scrollTo({ top: 0, behavior: 'smooth' });
    return { status: 'scrolled', position: 'top' };
  }
  if (args.position === 'bottom') {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
    return { status: 'scrolled', position: 'bottom' };
  }
  const el = resolveElement(args);
  if (el) {
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    return { status: 'scrolled', selector: getElementSelector(el) };
  }
  return { status: 'noop' };
}

function isElementVisible(el) {
  const rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0 && rect.bottom >= 0 && rect.top <= (window.innerHeight || document.documentElement.clientHeight);
}

function shouldConfirm(el, intent) {
  if (intent && /(purchase|checkout|submit|confirm)/i.test(intent)) {
    return true;
  }
  const text = (el.innerText || el.value || '').toLowerCase();
  return /(buy|checkout|submit|order|cart)/.test(text);
}

function showConfirmModal(element, intent) {
  return new Promise(resolve => {
    const overlay = document.createElement('div');
    overlay.className = 'awa-confirm-overlay';
    overlay.innerHTML = `
      <div class="awa-confirm-dialog" role="dialog" aria-modal="true">
        <h2 dir="auto">Confirm Step</h2>
        <p dir="auto">The assistant wants to ${intent}. Do you approve?</p>
        <div class="awa-confirm-actions">
          <button class="awa-confirm-approve">Proceed</button>
          <button class="awa-confirm-cancel">Cancel</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);
    const cleanup = () => {
      overlay.remove();
    };
    overlay.querySelector('.awa-confirm-approve').addEventListener('click', () => {
      cleanup();
      resolve(true);
    });
    overlay.querySelector('.awa-confirm-cancel').addEventListener('click', () => {
      cleanup();
      resolve(false);
    });
  });
}

function waitFor(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function getElementSelector(el) {
  if (!(el instanceof Element)) return '';
  if (el.id) return `#${el.id}`;
  const path = [];
  while (el && el.nodeType === Node.ELEMENT_NODE && path.length < 5) {
    let selector = el.nodeName.toLowerCase();
    if (el.className) {
      const classes = Array.from(el.classList).slice(0, 3);
      if (classes.length) selector += '.' + classes.join('.');
    }
    path.unshift(selector);
    el = el.parentElement;
  }
  return path.join(' > ');
}

function renderTopbar(summary) {
  if (!topbarFrame) {
    topbarFrame = document.createElement('iframe');
    topbarFrame.src = chrome.runtime.getURL('topbar.html');
    topbarFrame.style.position = 'fixed';
    topbarFrame.style.top = '0';
    topbarFrame.style.left = '0';
    topbarFrame.style.right = '0';
    topbarFrame.style.height = '64px';
    topbarFrame.style.zIndex = '2147483646';
    topbarFrame.style.border = 'none';
    topbarFrame.style.boxShadow = '0 2px 6px rgba(0,0,0,0.2)';
    document.documentElement.appendChild(topbarFrame);
    document.documentElement.style.setProperty('--awa-topbar-offset', '64px');
    document.documentElement.style.scrollMarginTop = '64px';
  }
  topbarFrame.contentWindow?.postMessage({ type: 'awa-topbar-update', summary }, '*');
}

window.addEventListener('message', event => {
  if (event.data?.type === 'awa-topbar-close') {
    if (topbarFrame) {
      topbarFrame.remove();
      topbarFrame = null;
      document.documentElement.style.removeProperty('--awa-topbar-offset');
      document.documentElement.style.scrollMarginTop = '';
    }
  }
});

const style = document.createElement('style');
style.textContent = `
  .awa-confirm-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2147483645;
  }
  .awa-confirm-dialog {
    background: #fff;
    padding: 20px;
    max-width: 360px;
    border-radius: 8px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.2);
    font-family: system-ui, sans-serif;
    text-align: start;
  }
  .awa-confirm-dialog h2 {
    margin-top: 0;
  }
  .awa-confirm-actions {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
    margin-top: 20px;
  }
  .awa-confirm-actions button {
    border: none;
    border-radius: 4px;
    padding: 8px 14px;
    cursor: pointer;
    font-weight: 600;
  }
  .awa-confirm-approve {
    background: #1a73e8;
    color: white;
  }
  .awa-confirm-cancel {
    background: #f1f3f4;
    color: #202124;
  }
`;
document.documentElement.appendChild(style);

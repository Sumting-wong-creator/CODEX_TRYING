import { renderMarkdown } from './utils/markdown.js';

const port = chrome.runtime.connect({ name: 'sidebar' });

const chatLog = document.getElementById('chatLog');
const composerForm = document.getElementById('composerForm');
const composerInput = document.getElementById('composerInput');
const resetButton = document.querySelector('.hawa-reset');
const modeButtons = Array.from(document.querySelectorAll('.mode-btn'));
const quickButtons = Array.from(document.querySelectorAll('.hawa-quick-btn'));
const allowToggle = document.getElementById('allowInstructions');

let state = createInitialState();
let pendingBubble = null;
let pendingText = '';

init();

function init() {
  loadPreferences();
  composerForm.addEventListener('submit', handleSubmit);
  composerInput.addEventListener('input', autoResize);
  resetButton.addEventListener('click', () => resetConversation(true));
  modeButtons.forEach((button) => {
    button.addEventListener('click', () => selectMode(button.dataset.mode));
  });
  quickButtons.forEach((button) => {
    button.addEventListener('click', () => runQuickAction(button.dataset.action));
  });
  allowToggle.addEventListener('change', () => {
    state.allowInstructions = allowToggle.checked;
    savePreferences();
  });
  port.onMessage.addListener(handlePortMessage);
}

function createInitialState() {
  return {
    conversationId: crypto.randomUUID(),
    history: [],
    mode: 'ask',
    allowInstructions: false,
    streaming: false
  };
}

function loadPreferences() {
  chrome.storage.local.get('sidebarPrefs').then((data) => {
    if (data.sidebarPrefs?.allowInstructions) {
      state.allowInstructions = true;
      allowToggle.checked = true;
    }
  });
}

function savePreferences() {
  chrome.storage.local.set({ sidebarPrefs: { allowInstructions: state.allowInstructions } });
}

function handleSubmit(event) {
  event.preventDefault();
  const prompt = composerInput.value.trim();
  if (!prompt) return;
  sendMessage(prompt, { fromQuickAction: false });
}

function sendMessage(prompt, { fromQuickAction, quickAction } = {}) {
  if (state.streaming) {
    port.postMessage({ type: 'stop-session', payload: { conversationId: state.conversationId } });
  }
  appendUserBubble(prompt);
  composerInput.value = '';
  autoResize();
  pendingBubble = appendAssistantBubble('');
  pendingText = '';
  state.streaming = true;

  const payload = {
    conversationId: state.conversationId,
    history: [...state.history],
    prompt,
    mode: state.mode,
    allowInstructions: state.allowInstructions,
    quickAction: quickAction || (fromQuickAction ? 'summarize' : undefined)
  };

  port.postMessage({ type: 'start-session', payload });
}

function handlePortMessage(message) {
  if (!message) return;
  switch (message.type) {
    case 'token':
      if (message.conversationId !== state.conversationId) return;
      pendingText += message.token;
      updateAssistantBubble(pendingText, true);
      break;
    case 'complete':
      if (message.conversationId !== state.conversationId) return;
      finalizeAssistant(message.finalText);
      if (message.finalText) {
        state.history.push({ role: 'assistant', text: message.finalText });
      }
      state.streaming = false;
      break;
    case 'status':
      if (message.conversationId && message.conversationId !== state.conversationId) return;
      appendStatus(message.message || 'Something went wrong.');
      state.streaming = false;
      break;
    case 'quick-action':
      handleQuickActionMessage(message);
      break;
    default:
      break;
  }
}

function appendUserBubble(text) {
  const bubble = createBubble('user');
  const body = bubble.querySelector('.bubble-body');
  body.textContent = text;
  state.history.push({ role: 'user', text });
  scrollToBottom();
}

function appendAssistantBubble(text) {
  const bubble = createBubble('assistant');
  if (text) updateAssistantBubble(text, false, bubble);
  return bubble;
}

function updateAssistantBubble(text, isStreaming, bubbleRef) {
  const bubble = bubbleRef || pendingBubble || appendAssistantBubble('');
  const body = bubble.querySelector('.bubble-body');
  if (text) {
    renderMarkdown(body, text);
  } else {
    body.textContent = '';
  }
  bubble.classList.toggle('token-stream', isStreaming);
  scrollToBottom();
}

function finalizeAssistant(text) {
  if (!pendingBubble) {
    pendingBubble = appendAssistantBubble('');
  }
  updateAssistantBubble(text || 'HAWA could not respond.', false, pendingBubble);
  pendingBubble.classList.remove('token-stream');
  pendingBubble = null;
  pendingText = '';
}

function appendStatus(text) {
  const status = document.createElement('div');
  status.className = 'status-line';
  status.textContent = text;
  chatLog.appendChild(status);
  scrollToBottom();
}

function createBubble(role) {
  const article = document.createElement('article');
  article.className = `chat-bubble ${role}`;
  article.setAttribute('dir', 'auto');
  const author = document.createElement('div');
  author.className = 'bubble-author';
  author.textContent = role === 'user' ? 'You' : 'HAWA';
  const body = document.createElement('div');
  body.className = 'bubble-body';
  article.append(author, body);
  chatLog.appendChild(article);
  scrollToBottom();
  return article;
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatLog.parentElement.scrollTop = chatLog.parentElement.scrollHeight;
  });
}

function autoResize() {
  composerInput.style.height = 'auto';
  composerInput.style.height = `${composerInput.scrollHeight}px`;
}

function resetConversation(animate) {
  port.postMessage({ type: 'stop-session', payload: { conversationId: state.conversationId } });
  chatLog.innerHTML = '';
  state = createInitialState();
  state.mode = modeButtons.find((btn) => btn.classList.contains('active'))?.dataset?.mode || 'ask';
  state.allowInstructions = allowToggle.checked;
  pendingBubble = null;
  pendingText = '';
  if (animate) {
    const shell = document.querySelector('.hawa-shell');
    shell.classList.add('resetting');
    setTimeout(() => shell.classList.remove('resetting'), 400);
  }
}

function selectMode(mode) {
  state.mode = mode;
  modeButtons.forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-pressed', active ? 'true' : 'false');
  });
}

function runQuickAction(action) {
  if (action === 'summarize') {
    sendMessage('Please summarize this page.', { fromQuickAction: true, quickAction: 'summarize' });
  }
}

function handleQuickActionMessage(message) {
  const basePrompt = message.action === 'summarize'
    ? 'Please summarize this page.'
    : message.action === 'claim-epic'
    ? 'Draft a short "Claim Epic" summary for this content.'
    : '';
  if (!basePrompt) return;
  const prompt = message.selection ? `Focus on this selection first:\n${message.selection}\n\n${basePrompt}` : basePrompt;
  sendMessage(prompt, { fromQuickAction: true, quickAction: message.action });
}

window.addEventListener('beforeunload', () => {
  port.disconnect();
});

import { renderMarkdown } from './utils/markdown.js';

const messagesEl = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const formEl = document.getElementById('composer');
const stopBtn = document.getElementById('stop-stream');
const newChatBtn = document.getElementById('new-chat');
const allowInstructionsToggle = document.getElementById('allow-instructions');
const modeTabs = Array.from(document.querySelectorAll('.mode-tab'));
const quickActionButtons = document.querySelectorAll('.quick-action-btn');
const sendBtn = document.getElementById('send');

const port = chrome.runtime.connect({ name: 'sidebar' });
let conversationId = crypto.randomUUID();
let history = [];
let streaming = false;
let assistantBuffer = '';
let currentAssistantNode = null;
let currentMode = 'ask';
let settings = {
  allowInstructions: false
};

init();

function init() {
  setMode(currentMode, { silent: true });
  chrome.storage.local.get(['settings']).then(({ settings: storedSettings }) => {
    if (storedSettings) {
      settings = { ...settings, ...storedSettings };
      allowInstructionsToggle.checked = !!settings.allowInstructions;
    }
  });
  chrome.storage.local.get(['recentChats']).then(({ recentChats }) => {
    if (recentChats?.length) {
      renderSystemMessage('Welcome back. Continue where you left off or start something new.');
    } else {
      renderSystemMessage('HAWA is ready. Ask a question or let her explore in Agent mode.');
    }
  });
  sendBtn.disabled = true;
}

modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    const mode = tab.dataset.mode;
    if (mode && mode !== currentMode) {
      setMode(mode);
    }
  });
});

promptEl.addEventListener('input', () => {
  autoSizePrompt();
  sendBtn.disabled = !promptEl.value.trim();
});

autoSizePrompt();

allowInstructionsToggle.addEventListener('change', persistSettings);

formEl.addEventListener('submit', event => {
  event.preventDefault();
  if (streaming) return;
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  sendPrompt(prompt);
});

stopBtn.addEventListener('click', () => {
  port.postMessage({ type: 'stop-request' });
  streaming = false;
  stopBtn.disabled = true;
  sendBtn.disabled = !promptEl.value.trim();
});

newChatBtn.addEventListener('click', () => startNewChat());

quickActionButtons.forEach(button => {
  button.addEventListener('click', () => {
    const action = button.dataset.action;
    if (action) {
      runQuickAction(action);
    }
  });
});

port.onMessage.addListener(message => {
  if (!message) return;
  switch (message.type) {
    case 'token':
      if (message.conversationId !== conversationId) return;
      ensureAssistantMessage();
      assistantBuffer += message.token;
      renderMarkdown(currentAssistantNode.querySelector('.content'), assistantBuffer);
      break;
    case 'complete':
      if (message.conversationId !== conversationId) return;
      finalizeAssistantMessage();
      streaming = false;
      stopBtn.disabled = true;
      addToHistory('assistant', assistantBuffer);
      saveRecentChat();
      sendBtn.disabled = !promptEl.value.trim();
      break;
    case 'status':
      renderSystemMessage(message.message, message.status);
      streaming = false;
      stopBtn.disabled = true;
      sendBtn.disabled = !promptEl.value.trim();
      break;
    case 'tool-response':
      if (message.tool === 'readPage') {
        renderSystemMessage('Shared fresh page context with HAWA.');
      }
      break;
    case 'quick-action':
      handleQuickAction(message);
      break;
    default:
      break;
  }
});

function setMode(mode, { silent = false } = {}) {
  currentMode = mode === 'agent' ? 'agent' : 'ask';
  document.documentElement.dataset.mode = currentMode;
  modeTabs.forEach(tab => {
    const active = tab.dataset.mode === currentMode;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  promptEl.placeholder = currentMode === 'agent'
    ? 'Describe the task for HAWA to perform in a fresh workspace…'
    : 'Ask HAWA…';
  if (!silent) {
    renderSystemMessage(currentMode === 'agent'
      ? 'Agent mode engages a dedicated workspace tab. HAWA will narrate each action and wait for your go-ahead.'
      : 'Ask mode keeps HAWA focused on this page. Provide context or ask for help.');
  }
}

function sendPrompt(prompt) {
  appendMessage('user', prompt);
  assistantBuffer = '';
  currentAssistantNode = appendMessage('assistant', '');
  streaming = true;
  stopBtn.disabled = false;
  sendBtn.disabled = true;
  const context = history.map(item => ({ ...item }));
  const request = {
    conversationId,
    prompt,
    history: context,
    allowInstructions: allowInstructionsToggle.checked,
    mode: currentMode
  };
  port.postMessage({ type: 'start-request', request });
  addToHistory('user', prompt);
  promptEl.value = '';
  autoSizePrompt();
}

function ensureAssistantMessage() {
  if (!currentAssistantNode) {
    currentAssistantNode = appendMessage('assistant', '');
  }
}

function finalizeAssistantMessage() {
  if (!currentAssistantNode) return;
  renderMarkdown(currentAssistantNode.querySelector('.content'), assistantBuffer);
  currentAssistantNode = null;
  assistantBuffer = '';
}

function appendMessage(role, text) {
  const li = document.createElement('li');
  li.className = `message ${role}`;
  const roleEl = document.createElement('div');
  roleEl.className = 'role';
  roleEl.textContent = role === 'assistant' ? 'HAWA' : role === 'system' ? 'System' : 'You';
  const content = document.createElement('div');
  content.className = 'content';
  content.dir = 'auto';
  if (role === 'assistant') {
    renderMarkdown(content, text);
  } else {
    content.textContent = text;
  }
  li.append(roleEl, content);
  messagesEl.appendChild(li);
  messagesEl.parentElement.scrollTop = messagesEl.parentElement.scrollHeight;
  return li;
}

function addToHistory(role, text) {
  history.push({ role, text });
  if (history.length > 16) {
    history = history.slice(history.length - 16);
  }
}

function startNewChat({ announce = true } = {}) {
  history = [];
  conversationId = crypto.randomUUID();
  messagesEl.innerHTML = '';
  currentAssistantNode = null;
  assistantBuffer = '';
  streaming = false;
  stopBtn.disabled = true;
  promptEl.value = '';
  autoSizePrompt();
  document.body.classList.add('chat-resetting');
  setTimeout(() => document.body.classList.remove('chat-resetting'), 480);
  if (announce) {
    renderSystemMessage('New chat started. HAWA is listening.');
  }
}

function renderSystemMessage(text, status = 'info') {
  const li = appendMessage('system', text);
  li.classList.add(`status-${status}`);
}

function persistSettings() {
  settings.allowInstructions = allowInstructionsToggle.checked;
  chrome.storage.local.set({ settings });
  port.postMessage({ type: 'persist-settings', settings });
}

function handleQuickAction(message) {
  const { action, selection } = message;
  runQuickAction(action, selection);
}

function runQuickAction(action, selection = '') {
  const prompt = buildQuickActionPrompt(action, selection);
  if (!prompt) return;
  if (currentMode !== 'ask') {
    setMode('ask', { silent: true });
  }
  startNewChat({ announce: false });
  renderSystemMessage('Summarizing the latest view of this page…');
  sendPrompt(prompt);
}

function buildQuickActionPrompt(action, selection = '') {
  if (action === 'summarize') {
    if (selection) {
      return `Summarize the selected content with clear bullets and a headline. Selection:\n\n${selection}`;
    }
    return 'Read the current page using available tools and produce a crisp summary with highlights, key takeaways, and next actions.';
  }
  return '';
}

function saveRecentChat() {
  const title = history.findLast?.(item => item.role === 'user')?.text?.slice(0, 80) || 'Conversation';
  chrome.storage.local.get(['recentChats']).then(({ recentChats }) => {
    const updated = [{ id: conversationId, title, timestamp: Date.now() }, ...(recentChats || []).filter(item => item.id !== conversationId)];
    chrome.storage.local.set({ recentChats: updated.slice(0, 10) });
  });
}

function autoSizePrompt() {
  promptEl.style.height = 'auto';
  const next = Math.min(promptEl.scrollHeight, 180);
  promptEl.style.height = `${Math.max(next, 48)}px`;
}

if (!Array.prototype.findLast) {
  Array.prototype.findLast = function(predicate) {
    for (let i = this.length - 1; i >= 0; i -= 1) {
      if (predicate(this[i], i, this)) return this[i];
    }
    return undefined;
  };
}

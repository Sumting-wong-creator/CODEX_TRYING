import { renderMarkdown } from './utils/markdown.js';

const messagesEl = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const formEl = document.getElementById('composer');
const stopBtn = document.getElementById('stop-stream');
const newChatBtn = document.getElementById('new-chat');
const tempSlider = document.getElementById('temperature');
const tempValue = document.getElementById('temperature-value');
const allowInstructionsToggle = document.getElementById('allow-instructions');
const modelSelect = document.getElementById('model');
const quickActionButtons = document.querySelectorAll('.quick-action-btn');

const port = chrome.runtime.connect({ name: 'sidebar' });
let conversationId = crypto.randomUUID();
let history = [];
let streaming = false;
let assistantBuffer = '';
let currentAssistantNode = null;
let settings = {
  allowInstructions: false,
  temperature: 0.4,
  model: 'models/gemini-2.5-flash'
};

init();

function init() {
  chrome.storage.local.get(['settings']).then(({ settings: storedSettings }) => {
    if (storedSettings) {
      settings = { ...settings, ...storedSettings };
      allowInstructionsToggle.checked = !!settings.allowInstructions;
      tempSlider.value = settings.temperature?.toString() ?? '0.4';
      tempValue.textContent = parseFloat(tempSlider.value).toFixed(1);
    }
    modelSelect.value = settings.model || 'models/gemini-2.5-flash';
  });
  chrome.storage.local.get(['recentChats']).then(({ recentChats }) => {
    if (recentChats?.length) {
      renderSystemMessage('Loaded recent chat context. Start typing to continue.');
    }
  });
}

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
      break;
    case 'status':
      renderSystemMessage(message.message, message.status);
      streaming = false;
      stopBtn.disabled = true;
      break;
    case 'tool-response':
      if (message.tool === 'readPage') {
        renderSystemMessage('Page context shared with the assistant.');
      }
      break;
    case 'quick-action':
      handleQuickAction(message);
      break;
    default:
      break;
  }
});

formEl.addEventListener('submit', event => {
  event.preventDefault();
  if (streaming) return;
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  sendPrompt(prompt);
});

tempSlider.addEventListener('input', () => {
  tempValue.textContent = parseFloat(tempSlider.value).toFixed(1);
});

tempSlider.addEventListener('change', persistSettings);
allowInstructionsToggle.addEventListener('change', () => {
  persistSettings();
});
modelSelect.addEventListener('change', persistSettings);

stopBtn.addEventListener('click', () => {
  port.postMessage({ type: 'stop-request' });
  streaming = false;
  stopBtn.disabled = true;
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

function sendPrompt(prompt) {
  appendMessage('user', prompt);
  assistantBuffer = '';
  currentAssistantNode = appendMessage('assistant', '');
  streaming = true;
  stopBtn.disabled = false;
  const context = history.map(item => ({ ...item }));
  const request = {
    conversationId,
    prompt,
    history: context,
    temperature: parseFloat(tempSlider.value),
    allowInstructions: allowInstructionsToggle.checked,
    model: modelSelect.value
  };
  port.postMessage({ type: 'start-request', request });
  addToHistory('user', prompt);
  promptEl.value = '';
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
  roleEl.textContent = role === 'assistant' ? 'Assistant' : role === 'system' ? 'System' : 'You';
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
  if (announce) {
    renderSystemMessage('New chat started.');
  }
}

function renderSystemMessage(text, status = 'info') {
  const li = appendMessage('system', text);
  li.classList.add(`status-${status}`);
}

function persistSettings() {
  settings.allowInstructions = allowInstructionsToggle.checked;
  settings.temperature = parseFloat(tempSlider.value);
  settings.model = modelSelect.value;
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
  startNewChat({ announce: false });
  if (action === 'summarize') {
    renderSystemMessage('Summarizing this page with the latest context…');
  } else if (action === 'claim-epic') {
    renderSystemMessage('Drafting an epic claim based on this page…');
  }
  sendPrompt(prompt);
}

function buildQuickActionPrompt(action, selection = '') {
  if (action === 'summarize') {
    if (selection) {
      return `Summarize the selected content with clear bullets and a headline. Selection:\n\n${selection}`;
    }
    return 'Read the current page using available tools and produce a crisp summary with highlights and action items if present.';
  }
  if (action === 'claim-epic') {
    let prompt = 'Use the page context to draft a concise epic claim with goal, acceptance criteria, and notable risks or dependencies.';
    if (selection) {
      prompt += `\n\nReference this selection as primary source material:\n${selection}`;
    }
    return prompt;
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

if (!Array.prototype.findLast) {
  Array.prototype.findLast = function(predicate) {
    for (let i = this.length - 1; i >= 0; i -= 1) {
      if (predicate(this[i], i, this)) return this[i];
    }
    return undefined;
  };
}

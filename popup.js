const summarizeBtn = document.getElementById('summarize');
const openSidebarBtn = document.getElementById('open-sidebar');
const recentChatsEl = document.getElementById('recent-chats');

document.addEventListener('DOMContentLoaded', () => {
  ensureSidebarOpen();
  renderRecentChats();
});

summarizeBtn.addEventListener('click', () => triggerQuickAction('summarize'));
openSidebarBtn.addEventListener('click', () => openSidebar({ closeAfter: true }));

async function triggerQuickAction(action) {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  await chrome.runtime.sendMessage({ type: 'queue-quick-action', tabId: tab.id, action, selection: '' }).catch(() => {});
  await chrome.sidePanel.open({ tabId: tab.id }).catch(() => {});
  chrome.tabs.sendMessage(tab.id, { type: 'sidebar-open-request', action });
  window.close();
}

async function ensureSidebarOpen() {
  await openSidebar({ closeAfter: false });
}

async function openSidebar({ closeAfter } = {}) {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  await chrome.sidePanel.open({ tabId: tab.id }).catch(() => {});
  if (closeAfter) {
    window.close();
  }
}

async function renderRecentChats() {
  const { recentChats } = await chrome.storage.local.get('recentChats');
  recentChatsEl.innerHTML = '';
  (recentChats || []).slice(0, 5).forEach(chat => {
    const li = document.createElement('li');
    li.textContent = chat.title;
    li.title = new Date(chat.timestamp).toLocaleString();
    li.addEventListener('click', async () => {
      const tab = await getActiveTab();
      if (!tab?.id) return;
      await chrome.sidePanel.open({ tabId: tab.id }).catch(() => {});
      chrome.runtime.sendMessage({ type: 'trigger-topbar', tabId: tab.id, summary: { line: chat.title, bullets: ['Open the sidebar to continue with HAWA.'] } });
      window.close();
    });
    recentChatsEl.appendChild(li);
  });
  if (!recentChatsEl.children.length) {
    const li = document.createElement('li');
    li.textContent = 'No chats yet. Start a conversation!';
    li.style.cursor = 'default';
    li.addEventListener('click', () => {});
    recentChatsEl.appendChild(li);
  }
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

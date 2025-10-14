const summarizeBtn = document.getElementById('summarize');
const claimEpicBtn = document.getElementById('claim-epic');
const openSidebarBtn = document.getElementById('open-sidebar');
const recentChatsEl = document.getElementById('recent-chats');

summarizeBtn.addEventListener('click', () => triggerQuickAction('summarize'));
claimEpicBtn.addEventListener('click', () => triggerQuickAction('claim-epic'));
openSidebarBtn.addEventListener('click', openSidebar);

renderRecentChats();

async function triggerQuickAction(action) {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  await chrome.sidePanel.open({ tabId: tab.id }).catch(() => {});
  chrome.tabs.sendMessage(tab.id, { type: 'sidebar-open-request', action });
  window.close();
}

async function openSidebar() {
  const tab = await getActiveTab();
  if (!tab?.id) return;
  await chrome.sidePanel.open({ tabId: tab.id }).catch(() => {});
  window.close();
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
      chrome.runtime.sendMessage({ type: 'trigger-topbar', tabId: tab.id, summary: { line: chat.title, bullets: ['Reopen sidebar to continue.'] } });
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

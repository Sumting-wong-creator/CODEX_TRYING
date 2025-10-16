async function init() {
  const buttons = document.querySelectorAll('button[data-action]');
  buttons.forEach((button) => {
    button.addEventListener('click', () => handleAction(button.dataset.action));
  });
}

async function handleAction(action) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) return;
  switch (action) {
    case 'open':
      await chrome.sidePanel.open({ tabId: tab.id }).catch(() => {});
      break;
    case 'summarize':
      await chrome.sidePanel.open({ tabId: tab.id }).catch(() => {});
      chrome.runtime.sendMessage({ type: 'popup-summarize', tabId: tab.id });
      break;
    default:
      break;
  }
  window.close();
}

document.addEventListener('DOMContentLoaded', init);

import { encryptText, decryptText } from './utils/crypto.js';

const apiForm = document.getElementById('api-form');
const apiKeyInput = document.getElementById('api-key');
const enableEncryption = document.getElementById('enable-encryption');
const passphraseField = document.getElementById('passphrase-field');
const passphraseInput = document.getElementById('passphrase');
const unlockBtn = document.getElementById('unlock');
const testBtn = document.getElementById('test-key');
const statusEl = document.getElementById('api-status');
const allowListTextarea = document.getElementById('allow-list');
const saveAllowListBtn = document.getElementById('save-allow-list');
const defaultAllowInstructions = document.getElementById('default-allow-instructions');

let apiKeyData = null;

init();

async function init() {
  const stored = await chrome.storage.local.get(['apiKeyData', 'settings', 'apiKeyCache']);
  apiKeyData = stored.apiKeyData || null;
  if (apiKeyData) {
    if (apiKeyData.encrypted) {
      enableEncryption.checked = true;
      passphraseField.hidden = false;
    } else if (apiKeyData.value) {
      apiKeyInput.value = apiKeyData.value;
    }
  }
  if (stored.settings) {
    allowListTextarea.value = (stored.settings.allowList || []).join('\n');
    defaultAllowInstructions.checked = !!stored.settings.allowInstructions;
  }
  enableEncryption.addEventListener('change', () => {
    passphraseField.hidden = !enableEncryption.checked;
  });
}

apiForm.addEventListener('submit', async event => {
  event.preventDefault();
  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) {
    showStatus('Enter your Gemini API key.', 'error');
    return;
  }
  if (enableEncryption.checked) {
    const passphrase = passphraseInput.value;
    if (!passphrase) {
      showStatus('Passphrase required for encryption.', 'error');
      return;
    }
    try {
      const encrypted = await encryptText(apiKey, passphrase);
      apiKeyData = { encrypted: true, ...encrypted };
      await chrome.storage.local.set({ apiKeyData, apiKeyCache: null });
      showStatus('Encrypted API key saved.', 'success');
      apiKeyInput.value = '';
    } catch (error) {
      showStatus(`Encryption failed: ${error.message}`, 'error');
    }
  } else {
    apiKeyData = { encrypted: false, value: apiKey };
    await chrome.storage.local.set({ apiKeyData });
    showStatus('API key saved.', 'success');
  }
});

unlockBtn.addEventListener('click', async () => {
  if (!apiKeyData || !apiKeyData.encrypted) {
    showStatus('No encrypted key to unlock.', 'error');
    return;
  }
  const passphrase = passphraseInput.value;
  if (!passphrase) {
    showStatus('Enter your passphrase to unlock.', 'error');
    return;
  }
  try {
    const decrypted = await decryptText(apiKeyData, passphrase);
    await chrome.storage.local.set({ apiKeyCache: { value: decrypted, expiresAt: Date.now() + 30 * 60 * 1000 } });
    showStatus('API key unlocked for 30 minutes.', 'success');
  } catch (error) {
    showStatus(`Unlock failed: ${error.message}`, 'error');
  }
});

testBtn.addEventListener('click', async () => {
  const apiKey = await getActiveApiKey();
  if (!apiKey) {
    showStatus('Provide or unlock your API key before testing.', 'error');
    return;
  }
  showStatus('Testing API key...', 'info');
  try {
    const response = await chrome.runtime.sendMessage({ type: 'test-api-key', apiKey });
    if (response?.ok) {
      showStatus('API key test succeeded.', 'success');
    } else {
      showStatus(`API key test failed: ${response?.error || 'unknown error'}`, 'error');
    }
  } catch (error) {
    showStatus(`API key test failed: ${error.message}`, 'error');
  }
});

saveAllowListBtn.addEventListener('click', async () => {
  const domains = allowListTextarea.value
    .split(/\r?\n/)
    .map(line => line.trim().toLowerCase())
    .filter(Boolean);
  const settings = await getSettings();
  settings.allowList = domains;
  await chrome.storage.local.set({ settings });
  showStatus('Allow-list updated.', 'success');
});

defaultAllowInstructions.addEventListener('change', async () => {
  const settings = await getSettings();
  settings.allowInstructions = defaultAllowInstructions.checked;
  await chrome.storage.local.set({ settings });
  showStatus('Safety preference updated.', 'success');
});

async function getSettings() {
  const { settings } = await chrome.storage.local.get('settings');
  return { allowList: [], allowInstructions: false, ...(settings || {}) };
}

async function getActiveApiKey() {
  const inputKey = apiKeyInput.value.trim();
  if (inputKey) return inputKey;
  if (!apiKeyData) return null;
  if (!apiKeyData.encrypted) return apiKeyData.value;
  const { apiKeyCache } = await chrome.storage.local.get('apiKeyCache');
  if (apiKeyCache?.value && apiKeyCache.expiresAt > Date.now()) {
    return apiKeyCache.value;
  }
  return null;
}

function showStatus(message, status) {
  statusEl.textContent = message;
  statusEl.dataset.status = status;
}

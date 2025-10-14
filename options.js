import { encryptText, decryptText } from './utils/crypto.js';
import { verifyGeminiApiKey, persistGeminiApiKey } from './Gemini-Handler.js';

const keyForm = document.getElementById('keyForm');
const apiKeyInput = document.getElementById('apiKey');
const encryptToggle = document.getElementById('encryptToggle');
const passphraseInput = document.getElementById('passphrase');
const keyStatus = document.getElementById('keyStatus');
const unlockBtn = document.getElementById('unlockBtn');
const lockBtn = document.getElementById('lockBtn');
const testBtn = document.getElementById('testBtn');
const allowListArea = document.getElementById('allowList');
const allowStatus = document.getElementById('allowStatus');
const saveAllowBtn = document.getElementById('saveAllowList');

init();

function init() {
  hydrate();
  keyForm.addEventListener('submit', handleSaveKey);
  unlockBtn.addEventListener('click', handleUnlock);
  lockBtn.addEventListener('click', handleLock);
  testBtn.addEventListener('click', handleTest);
  saveAllowBtn.addEventListener('click', handleSaveAllowList);
}

async function hydrate() {
  const { apiKeyData, apiKeyCache, settings } = await chrome.storage.local.get(['apiKeyData', 'apiKeyCache', 'settings']);
  if (apiKeyData) {
    encryptToggle.checked = Boolean(apiKeyData.encrypted);
    if (!apiKeyData.encrypted && apiKeyData.value) {
      apiKeyInput.value = apiKeyData.value;
    }
    if (apiKeyData.encrypted) {
      keyStatus.textContent = 'API key stored encrypted. Provide passphrase to unlock when needed.';
    }
  } else {
    keyStatus.textContent = 'API key not configured.';
  }
  if (apiKeyCache?.value && apiKeyCache.expiresAt > Date.now()) {
    const minutes = Math.round((apiKeyCache.expiresAt - Date.now()) / 60000);
    keyStatus.textContent = `API key unlocked for ${minutes} more minute(s).`;
  }
  if (settings?.allowList?.length) {
    allowListArea.value = settings.allowList.join('\n');
  }
}

async function handleSaveKey(event) {
  event.preventDefault();
  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) {
    keyStatus.textContent = 'Enter an API key first.';
    return;
  }

  keyStatus.textContent = 'Verifying API key…';
  let verifiedKey;
  try {
    const { apiKey: cleanedKey } = await verifyGeminiApiKey(apiKey);
    verifiedKey = cleanedKey;
  } catch (error) {
    keyStatus.textContent = `Verification failed: ${error.message}`;
    return;
  }

  let payload;
  if (encryptToggle.checked) {
    const passphrase = passphraseInput.value.trim();
    if (!passphrase) {
      keyStatus.textContent = 'Enter a passphrase to encrypt the key.';
      return;
    }
    try {
      const encrypted = await encryptText(passphrase, verifiedKey);
      payload = { encrypted: true, payload: encrypted };
    } catch (error) {
      keyStatus.textContent = `Encryption failed: ${error.message}`;
      return;
    }
  } else {
    payload = { encrypted: false, value: verifiedKey };
  }
  try {
    await persistGeminiApiKey(verifiedKey, payload);
    keyStatus.textContent = encryptToggle.checked
      ? 'API key verified, encrypted, and stored. Unlock when you start chatting.'
      : 'API key verified and stored.';
  } catch (error) {
    keyStatus.textContent = `Save failed: ${error.message}`;
  }
}

async function handleUnlock() {
  const { apiKeyData } = await chrome.storage.local.get('apiKeyData');
  if (!apiKeyData) {
    keyStatus.textContent = 'No API key stored yet.';
    return;
  }
  if (!apiKeyData.encrypted) {
    if (apiKeyData.value) {
      await chrome.runtime.sendMessage({ type: 'unlock-api-key', value: apiKeyData.value, ttlMinutes: 30 });
      keyStatus.textContent = 'API key unlocked for 30 minutes.';
    }
    return;
  }
  const passphrase = passphraseInput.value.trim();
  if (!passphrase) {
    keyStatus.textContent = 'Enter the passphrase to unlock the key.';
    return;
  }
  try {
    const decrypted = await decryptText(passphrase, apiKeyData.payload);
    await chrome.runtime.sendMessage({ type: 'unlock-api-key', value: decrypted, ttlMinutes: 30 });
    keyStatus.textContent = 'API key unlocked for 30 minutes.';
  } catch (error) {
    keyStatus.textContent = 'Unable to unlock. Check the passphrase.';
  }
}

async function handleLock() {
  await chrome.runtime.sendMessage({ type: 'lock-api-key' });
  keyStatus.textContent = 'API key cache and environment value cleared.';
}

async function handleTest() {
  const key = apiKeyInput.value.trim();
  if (!key) {
    keyStatus.textContent = 'Enter a key in the field above to test.';
    return;
  }
  keyStatus.textContent = 'Testing…';
  try {
    await verifyGeminiApiKey(key);
    keyStatus.textContent = 'Gemini API responded successfully.';
  } catch (error) {
    keyStatus.textContent = `Test failed: ${error.message}`;
  }
}

async function handleSaveAllowList() {
  const raw = allowListArea.value.split(/\n+/).map((item) => item.trim().toLowerCase()).filter(Boolean);
  const unique = Array.from(new Set(raw));
  await chrome.storage.local.set({ settings: { allowList: unique } });
  allowStatus.textContent = `Saved ${unique.length} domain(s).`;
}

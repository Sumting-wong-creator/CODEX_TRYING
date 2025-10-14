const API_ROOT = 'https://generativelanguage.googleapis.com/v1/models';

function sanitizeKey(rawKey) {
  return rawKey.trim();
}

function isPlausibleKey(key) {
  return /^AIza[0-9A-Za-z_\-]{30,}$/.test(key);
}

export async function verifyGeminiApiKey(rawKey) {
  const apiKey = sanitizeKey(rawKey);
  if (!apiKey) {
    throw new Error('API key is required.');
  }
  if (!isPlausibleKey(apiKey)) {
    throw new Error('API key format looks incorrect.');
  }
  const url = `${API_ROOT}?key=${encodeURIComponent(apiKey)}`;
  const response = await fetch(url, { method: 'GET' });
  if (!response.ok) {
    const text = await response.text().catch(() => '');
    throw new Error(text || `Gemini API rejected the key (${response.status}).`);
  }
  const payload = await response.json().catch(() => null);
  if (!payload || typeof payload !== 'object') {
    throw new Error('Gemini API returned an unexpected response.');
  }
  return { apiKey, payload };
}

export async function persistGeminiApiKey(apiKey, storagePayload) {
  const message = await chrome.runtime.sendMessage({
    type: 'set-gemini-key',
    apiKey,
    payload: storagePayload
  });
  if (!message?.ok) {
    throw new Error(message?.error || 'Failed to save the API key.');
  }
  return message;
}

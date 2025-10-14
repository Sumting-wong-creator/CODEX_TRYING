export async function streamGemini({ apiKey, payload, signal, onToken, onTool, onEnd, onError, model }) {
  const modelPath = model || 'models/gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:streamGenerateContent?key=${encodeURIComponent(apiKey)}`;
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      signal
    });
  } catch (error) {
    onError?.(error);
    throw error;
  }
  if (!response.ok || !response.body) {
    const message = await response.text().catch(() => response.statusText);
    const err = new Error(message || `HTTP ${response.status}`);
    onError?.(err);
    throw err;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let done = false;
  const candidateState = { tokenBuffer: '', finalCandidate: null };

  while (!done) {
    const { value, done: readerDone } = await reader.read();
    done = readerDone;
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let separatorIndex;
    while ((separatorIndex = buffer.indexOf('\n\n')) !== -1) {
      const chunk = buffer.slice(0, separatorIndex).trim();
      buffer = buffer.slice(separatorIndex + 2);
      if (!chunk) continue;
      const dataLines = chunk.split('\n').filter(line => line.startsWith('data:'));
      for (const line of dataLines) {
        const payloadText = line.replace(/^data:\s*/, '');
        if (payloadText === '[DONE]') {
          done = true;
          break;
        }
        try {
          const json = JSON.parse(payloadText);
          await handleEvent(json, onToken, onTool, candidateState);
        } catch (error) {
          console.warn('Failed to parse stream chunk', error);
        }
      }
    }
  }
  onEnd?.(candidateState.finalCandidate);
}

async function handleEvent(json, onToken, onTool, state) {
  const candidates = json?.candidates || [];
  for (const candidate of candidates) {
    state.finalCandidate = candidate;
    const parts = candidate?.content?.parts || [];
    for (const part of parts) {
      if (typeof part.text === 'string') {
        onToken?.(part.text);
      }
      if (part.functionCall) {
        if (onTool) await onTool({ name: part.functionCall.name, args: part.functionCall.args });
      } else if (part.function_call) {
        if (onTool) await onTool({ name: part.function_call.name, args: part.function_call.args });
      }
    }
  }
}

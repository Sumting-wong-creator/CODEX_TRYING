export async function streamGemini({ apiKey, payload, signal, onToken, onTool, onEnd, onError, model }) {
  const modelPath = model || 'models/gemini-2.5-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:streamGenerateContent?key=${encodeURIComponent(apiKey)}`;
  console.debug('[HAWA][stream] starting request', { modelPath, tokenCount: payload?.contents?.length ?? 0 });
  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey
      },
      body: JSON.stringify(payload),
      signal
    });
  } catch (error) {
    if (error?.name === 'AbortError') {
      console.debug('[HAWA][stream] request aborted before response');
      return { aborted: true };
    }
    console.error('[HAWA][stream] network error', error);
    onError?.(error);
    throw error;
  }
  if (!response.ok || !response.body) {
    const message = await response.text().catch(() => response.statusText);
    const err = new Error(message || `HTTP ${response.status}`);
    console.error('[HAWA][stream] bad response', response.status, message);
    onError?.(err);
    throw err;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let done = false;
  const candidateState = {
    finalCandidate: null,
    tokenBuffers: new Map()
  };

  try {
    while (!done) {
      const { value, done: readerDone } = await reader.read();
      done = readerDone;
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let safetyCounter = 0;
      while (true) {
        safetyCounter += 1;
        if (safetyCounter > 5000) {
          console.warn('[HAWA][stream] aborting parser loop due to safety counter');
          break;
        }
        const doubleBreak = findSeparator(buffer, '\n\n');
        const singleBreak = findSeparator(buffer, '\n');
        let useIndex = -1;
        let sepLength = 1;
        if (doubleBreak !== -1) {
          useIndex = doubleBreak;
          sepLength = 2;
        } else if (singleBreak !== -1) {
          useIndex = singleBreak;
          sepLength = 1;
        }
        if (useIndex === -1) break;
        const rawChunk = buffer.slice(0, useIndex).trim();
        buffer = buffer.slice(useIndex + sepLength);
        if (!rawChunk) continue;
        const result = await processChunk(rawChunk, onToken, onTool, candidateState);
        if (result === true) {
          done = true;
          buffer = '';
          break;
        }
        if (result && typeof result === 'object' && result.requeue) {
          buffer = `${result.requeue}\n${buffer}`;
          break;
        }
      }
    }
  } catch (error) {
    if (error?.name === 'AbortError') {
      console.debug('[HAWA][stream] reader aborted');
      return { aborted: true };
    }
    console.error('[HAWA][stream] stream processing error', error);
    onError?.(error);
    throw error;
  }
  const trailing = buffer.trim();
  if (trailing) {
    try {
      await processChunk(trailing, onToken, onTool, candidateState);
    } catch (error) {
      console.warn('[HAWA][stream] trailing chunk parse failed', error, trailing);
    }
  }
  onEnd?.(candidateState.finalCandidate);
  console.debug('[HAWA][stream] request complete');
  return { aborted: false, finalCandidate: candidateState.finalCandidate };
}

function findSeparator(buffer, separator) {
  const idx = buffer.indexOf(separator);
  if (idx === -1) {
    const alt = separator.replace(/\n/g, '\r\n');
    return buffer.indexOf(alt);
  }
  return idx;
}

async function processChunk(chunk, onToken, onTool, state) {
  if (!chunk) return false;
  if (chunk.startsWith('data:')) {
    const lines = chunk.split(/\r?\n/);
    for (const line of lines) {
      if (!line.startsWith('data:')) continue;
      const payloadText = line.replace(/^data:\s*/, '').trim();
      if (!payloadText) continue;
      if (payloadText === '[DONE]') {
        return true;
      }
      try {
        const json = JSON.parse(payloadText);
        await handleEvent(json, onToken, onTool, state);
      } catch (error) {
        console.warn('[HAWA][stream] SSE chunk parse failed', error, payloadText);
      }
    }
    return false;
  }
  if (chunk === '[DONE]') {
    return true;
  }
  try {
    const json = JSON.parse(chunk);
    await handleEvent(json, onToken, onTool, state);
    return false;
  } catch (error) {
    if (error instanceof SyntaxError) {
      return { requeue: chunk };
    }
    throw error;
  }
}

async function handleEvent(json, onToken, onTool, state) {
  const candidates = json?.candidates || [];
  for (const candidate of candidates) {
    state.finalCandidate = candidate;
    const aggregatedText = extractCandidateText(candidate);
    const index = candidate?.index ?? 0;
    const previous = state.tokenBuffers.get(index) || '';
    if (aggregatedText && aggregatedText !== previous) {
      const delta = aggregatedText.startsWith(previous)
        ? aggregatedText.slice(previous.length)
        : aggregatedText;
      if (delta) onToken?.(delta);
      state.tokenBuffers.set(index, aggregatedText);
    } else if (!state.tokenBuffers.has(index)) {
      state.tokenBuffers.set(index, aggregatedText || '');
    }
    const parts = candidate?.content?.parts || [];
    for (const part of parts) {
      if (part.functionCall) {
        if (onTool) await onTool({ name: part.functionCall.name, args: part.functionCall.args });
      } else if (part.function_call) {
        if (onTool) await onTool({ name: part.function_call.name, args: part.function_call.args });
      }
    }
  }
}

function extractCandidateText(candidate) {
  if (!candidate?.content?.parts) return '';
  return candidate.content.parts
    .filter(part => typeof part.text === 'string')
    .map(part => part.text)
    .join('');
}

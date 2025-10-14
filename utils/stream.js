export async function streamGemini({ apiKey, payload, signal, onToken, onTool, onEnd, onError, model }) {
  const modelPath = model && model.startsWith('models/') ? model : `models/${model || 'gemini-2.5-flash'}`;
  const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;
  console.debug('[HAWA][stream] dispatch', { modelPath });

  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'x-goog-api-key': apiKey
      },
      body: JSON.stringify(payload),
      signal
    });
  } catch (error) {
    if (error?.name === 'AbortError') {
      console.debug('[HAWA][stream] aborted before response');
      return { aborted: true };
    }
    console.error('[HAWA][stream] network failure', error);
    onError?.(error);
    throw error;
  }

  if (!response.ok || !response.body) {
    const message = await response.text().catch(() => response.statusText);
    const err = new Error(message || `HTTP ${response.status}`);
    console.error('[HAWA][stream] http error', response.status, message);
    onError?.(err);
    throw err;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  const aggregatedByIndex = new Map();
  let finalCandidate = null;
  let promptFeedback = null;
  let usageMetadata = null;
  let buffer = '';
  let eventLines = [];

  let finished = false;
  try {
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let lineBreakIndex;
      while ((lineBreakIndex = findNextLine(buffer)) !== -1) {
        const nextChar = buffer[lineBreakIndex];
        const sliceOffset = nextChar === '\r' && buffer[lineBreakIndex + 1] === '\n' ? 2 : 1;
        const rawLine = buffer.slice(0, lineBreakIndex);
        buffer = buffer.slice(lineBreakIndex + sliceOffset);
        const line = rawLine.replace(/\r$/, '');

        if (line === '') {
          if (eventLines.length) {
            const result = await flushEvent({
              eventLines,
              aggregatedByIndex,
              onToken,
              onTool,
              onError,
              state: { finalCandidate, promptFeedback, usageMetadata }
            });
            ({ finalCandidate, promptFeedback, usageMetadata, finished } = result);
            eventLines = [];
            if (finished) {
              buffer = '';
              reader.cancel?.().catch(() => {});
              break;
            }
          }
          continue;
        }

        if (line.startsWith('data:')) {
          eventLines.push(line.slice(5).replace(/^\s*/, ''));
        } else if (line.startsWith(':')) {
          continue; // comment line
        } else {
          eventLines.push(line.trim());
        }
      }
    }
  } catch (error) {
    if (error?.name === 'AbortError') {
      console.debug('[HAWA][stream] aborted during read');
      return { aborted: true };
    }
    console.error('[HAWA][stream] reader error', error);
    onError?.(error);
    throw error;
  }

  if (!finished && eventLines.length) {
    const result = await flushEvent({
      eventLines,
      aggregatedByIndex,
      onToken,
      onTool,
      onError,
      state: { finalCandidate, promptFeedback, usageMetadata }
    });
    ({ finalCandidate, promptFeedback, usageMetadata, finished } = result);
    eventLines = [];
  }

  const candidateToReturn = buildFinalCandidate(finalCandidate, aggregatedByIndex, promptFeedback, usageMetadata);
  onEnd?.(candidateToReturn);
  console.debug('[HAWA][stream] complete');
  return { aborted: false, finalCandidate: candidateToReturn };
}

function findNextLine(buffer) {
  const idx = buffer.indexOf('\n');
  if (idx !== -1) return idx;
  return buffer.indexOf('\r');
}

async function handleEvent({ json, aggregatedByIndex, onToken, onTool, finalCandidate, promptFeedback, usageMetadata }) {
  if (json.promptFeedback) {
    promptFeedback = json.promptFeedback;
  }
  if (json.usageMetadata) {
    usageMetadata = json.usageMetadata;
  }

  const candidates = Array.isArray(json.candidates) ? json.candidates : [];
  for (const candidate of candidates) {
    finalCandidate = candidate;
    const index = candidate?.index ?? 0;
    const text = extractText(candidate);
    const previous = aggregatedByIndex.get(index) ?? '';
    if (typeof text === 'string') {
      if (!aggregatedByIndex.has(index) || text !== previous) {
        const delta = text.startsWith(previous) ? text.slice(previous.length) : text;
        if (delta) {
          try {
            onToken?.(delta);
          } catch (error) {
            console.warn('[HAWA][stream] onToken handler failed', error);
          }
        }
        aggregatedByIndex.set(index, text);
      }
    }

    const parts = Array.isArray(candidate?.content?.parts) ? candidate.content.parts : [];
    for (const part of parts) {
      const fn = part.functionCall || part.function_call;
      if (fn && onTool) {
        try {
          await onTool({ name: fn.name, args: fn.args });
        } catch (error) {
          console.warn('[HAWA][stream] onTool handler failed', error);
        }
      }
    }
  }

  return { finalCandidate, promptFeedback, usageMetadata };
}

function extractText(candidate) {
  if (!candidate?.content?.parts) return '';
  return candidate.content.parts
    .filter(part => typeof part.text === 'string')
    .map(part => part.text)
    .join('');
}

async function flushEvent({ eventLines, aggregatedByIndex, onToken, onTool, onError, state }) {
  let { finalCandidate, promptFeedback, usageMetadata } = state;
  const payloadText = eventLines.join('\n').trim();
  if (!payloadText) {
    return { finalCandidate, promptFeedback, usageMetadata, finished: false };
  }
  if (payloadText === '[DONE]') {
    return { finalCandidate, promptFeedback, usageMetadata, finished: true };
  }
  try {
    const json = JSON.parse(payloadText);
    if (json.error) {
      const err = new Error(json.error.message || 'Gemini API error');
      err.details = json.error;
      onError?.(err);
      throw err;
    }
    ({ finalCandidate, promptFeedback, usageMetadata } = await handleEvent({
      json,
      aggregatedByIndex,
      onToken,
      onTool,
      finalCandidate,
      promptFeedback,
      usageMetadata
    }));
  } catch (error) {
    console.warn('[HAWA][stream] event parse failed', error, payloadText);
  }
  return { finalCandidate, promptFeedback, usageMetadata, finished: false };
}

function buildFinalCandidate(finalCandidate, aggregatedByIndex, promptFeedback, usageMetadata) {
  if (!finalCandidate) {
    const text = aggregatedByIndex.get(0) ?? '';
    return {
      content: { parts: text ? [{ text }] : [] },
      promptFeedback,
      usageMetadata
    };
  }
  const index = finalCandidate.index ?? 0;
  const aggregatedText = aggregatedByIndex.get(index);
  if (typeof aggregatedText === 'string') {
    const otherParts = (finalCandidate.content?.parts || []).filter(part => typeof part.text !== 'string');
    finalCandidate = {
      ...finalCandidate,
      content: {
        parts: [
          ...(aggregatedText ? [{ text: aggregatedText }] : []),
          ...otherParts
        ]
      }
    };
  }
  if (promptFeedback && !finalCandidate.promptFeedback) {
    finalCandidate = { ...finalCandidate, promptFeedback };
  }
  if (usageMetadata && !finalCandidate.usageMetadata) {
    finalCandidate = { ...finalCandidate, usageMetadata };
  }
  return finalCandidate;
}

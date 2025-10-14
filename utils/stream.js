const DEFAULT_MODEL = 'gemini-2.5-flash';

export async function streamGemini({
  apiKey,
  payload,
  model = DEFAULT_MODEL,
  signal,
  onOpen,
  onToken,
  onComplete,
  onError
}) {
  if (!apiKey) {
    throw new Error('Missing Gemini API key.');
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?key=${encodeURIComponent(apiKey)}`;
  const toolCalls = [];

  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal
    });

    if (onOpen) onOpen(response);

    if (!response.ok) {
      const message = await safeReadText(response);
      throw new Error(message || `Gemini responded with ${response.status}`);
    }

    if (!response.body) {
      throw new Error('Gemini returned an empty response body.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let eventBuffer = '';
    let lastCandidate = null;
    let promptFeedback = null;
    let finished = false;
    let finalText = '';
    const candidateText = new Map();

    const processEvent = (eventData) => {
      if (!eventData) return;
      const trimmed = eventData.trim();
      if (!trimmed) return;
      if (trimmed === '[DONE]') {
        finished = true;
        return;
      }
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed.error) {
          throw new Error(parsed.error.message || 'Gemini returned an error.');
        }
        if (parsed.promptFeedback) {
          promptFeedback = parsed.promptFeedback;
        }
        const candidates = Array.isArray(parsed.candidates) ? parsed.candidates : [];
        for (const candidate of candidates) {
          lastCandidate = candidate;
          const parts = candidate?.content?.parts || [];
          let candidateTextValue = '';
          for (const part of parts) {
            if (typeof part.text === 'string') {
              candidateTextValue += part.text;
            }
            const fnCall = part.functionCall || part.function_call;
            if (fnCall) {
              toolCalls.push(fnCall);
            }
          }
          const key = candidate?.index ?? 0;
          const previous = candidateText.get(key) || '';
          if (candidateTextValue) {
            let next = '';
            let delta = '';
            if (candidateTextValue.startsWith(previous)) {
              next = candidateTextValue;
              delta = candidateTextValue.slice(previous.length);
            } else {
              next = previous + candidateTextValue;
              delta = candidateTextValue;
            }
            candidateText.set(key, next);
            if (key === 0) {
              finalText = next;
              if (delta && onToken) onToken(delta);
            }
          }
        }
      } catch (error) {
        console.warn('[HAWA][stream] Failed to parse Gemini chunk', error, trimmed);
      }
    };

    while (!finished) {
      const { value, done } = await reader.read();
      if (value) {
        buffer += decoder.decode(value, { stream: !done });
      }
      if (done) {
        buffer += decoder.decode();
      }

      let lineBreakIndex;
      while ((lineBreakIndex = buffer.indexOf('\n')) !== -1) {
        let line = buffer.slice(0, lineBreakIndex);
        buffer = buffer.slice(lineBreakIndex + 1);
        if (line.endsWith('\r')) {
          line = line.slice(0, -1);
        }

        if (line === '') {
          if (eventBuffer) {
            processEvent(eventBuffer);
            eventBuffer = '';
            if (finished) break;
          }
          continue;
        }

        if (line.startsWith('data:')) {
          const payloadSlice = line.slice(5).trimStart();
          if (eventBuffer) {
            eventBuffer += '\n';
          }
          eventBuffer += payloadSlice;
        }
      }

      if (finished) {
        break;
      }

      if (done) {
        if (eventBuffer) {
          processEvent(eventBuffer);
          eventBuffer = '';
        }
        break;
      }
    }

    if (!finalText && candidateText.has(0)) {
      finalText = candidateText.get(0) || '';
    }

    if (onComplete) {
      onComplete({
        text: finalText,
        candidate: lastCandidate,
        promptFeedback,
        toolCalls
      });
    }
  } catch (error) {
    if (onError) {
      onError(error);
    } else {
      throw error;
    }
  }
}

async function safeReadText(response) {
  try {
    return await response.text();
  } catch (_error) {
    return '';
  }
}

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
      headers: {
        'Content-Type': 'application/json'
      },
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
    let aggregateText = '';
    let lastCandidate = null;
    let promptFeedback = null;
    let finished = false;

    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      while (true) {
        const boundary = findBoundary(buffer);
        if (boundary < 0) break;
        const rawEvent = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary);
        const dataPayload = extractData(rawEvent);
        if (!dataPayload) continue;
        if (dataPayload === '[DONE]') {
          finished = true;
          buffer = '';
          break;
        }
        try {
          const parsed = JSON.parse(dataPayload);
          if (parsed.candidates?.length) {
            parsed.candidates.forEach(candidate => {
              lastCandidate = candidate;
              candidate.content?.parts?.forEach(part => {
                if (part.text) {
                  aggregateText += part.text;
                  if (onToken) onToken(part.text);
                }
                if (part.functionCall) {
                  toolCalls.push(part.functionCall);
                }
                if (part.function_call) {
                  toolCalls.push(part.function_call);
                }
              });
            });
          }
          if (parsed.promptFeedback) {
            promptFeedback = parsed.promptFeedback;
          }
        } catch (error) {
          console.warn('[HAWA][stream] Failed to parse chunk', error, dataPayload);
        }
      }
    }

    if (!aggregateText && lastCandidate) {
      aggregateText = lastCandidate.content?.parts?.map(part => part.text || '').join('') || '';
    }

    if (onComplete) {
      onComplete({
        text: aggregateText,
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

function findBoundary(buffer) {
  const carriage = buffer.indexOf('\r\n\r\n');
  const newline = buffer.indexOf('\n\n');
  if (carriage >= 0 && (newline < 0 || carriage < newline)) {
    return carriage + 4;
  }
  return newline >= 0 ? newline + 2 : -1;
}

function extractData(rawEvent) {
  const lines = rawEvent.split(/\r?\n/);
  let data = '';
  for (const line of lines) {
    if (line.startsWith('data:')) {
      data += line.slice(5).trim();
    }
  }
  return data.trim();
}

async function safeReadText(response) {
  try {
    return await response.text();
  } catch (error) {
    return '';
  }
}

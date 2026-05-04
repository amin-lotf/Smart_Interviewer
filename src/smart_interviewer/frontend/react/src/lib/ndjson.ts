interface StreamNdjsonOptions<T> {
  input: RequestInfo | URL;
  init?: RequestInit;
  onMessage: (message: T) => void;
}

async function parseError(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return `${response.status} ${response.statusText}`.trim();
  }

  try {
    const json = JSON.parse(text) as { detail?: string };
    return json.detail || text;
  } catch {
    return text;
  }
}

export async function streamNdjson<T>({
  input,
  init,
  onMessage,
}: StreamNdjsonOptions<T>): Promise<void> {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  if (!response.body) {
    throw new Error("Streaming response body is not available.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }
      onMessage(JSON.parse(trimmed) as T);
    }

    if (done) {
      break;
    }
  }

  const finalLine = buffer.trim();
  if (finalLine) {
    onMessage(JSON.parse(finalLine) as T);
  }
}

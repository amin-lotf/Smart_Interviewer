import type { SessionState, TokenStreamEvent } from "../types";
import { streamNdjson } from "./ndjson";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "");
}

export const API_BASE_URL = normalizeBaseUrl(
  (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).trim(),
);

function withSessionHeaders(sessionId: string, headers?: HeadersInit): Headers {
  const merged = new Headers(headers);
  merged.set("X-Session-Id", sessionId);
  return merged;
}

async function parseError(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return `${response.status} ${response.statusText}`.trim();
  }

  try {
    const payload = JSON.parse(text) as { detail?: string };
    return payload.detail || text;
  } catch {
    return text;
  }
}

async function requestJson<T>(
  path: string,
  init: RequestInit = {},
  sessionId?: string,
): Promise<T> {
  const headers = sessionId
    ? withSessionHeaders(sessionId, init.headers)
    : new Headers(init.headers);

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return (await response.json()) as T;
}

function buildAudioFormData(audio: Blob, filename: string): FormData {
  const formData = new FormData();
  formData.append("audio", audio, filename);
  return formData;
}

export const smartInterviewerApi = {
  health(): Promise<{ status: string; service?: string; version?: string }> {
    return requestJson("/");
  },

  getSessionState(sessionId: string): Promise<SessionState> {
    return requestJson("/v1/session/state", { method: "GET" }, sessionId);
  },

  resetSession(sessionId: string): Promise<{ status: string; session_id: string }> {
    return requestJson("/v1/session/reset", { method: "POST" }, sessionId);
  },

  finishInterview(sessionId: string): Promise<SessionState> {
    return requestJson("/v1/interview/finish", { method: "POST" }, sessionId);
  },

  startInterviewStream(
    sessionId: string,
    onMessage: (event: TokenStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    return streamNdjson<TokenStreamEvent>({
      input: `${API_BASE_URL}/v1/interview/start/stream`,
      init: {
        method: "POST",
        headers: withSessionHeaders(sessionId),
        signal,
      },
      onMessage,
    });
  },

  nextInterviewStream(
    sessionId: string,
    onMessage: (event: TokenStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    return streamNdjson<TokenStreamEvent>({
      input: `${API_BASE_URL}/v1/interview/next/stream`,
      init: {
        method: "POST",
        headers: withSessionHeaders(sessionId),
        signal,
      },
      onMessage,
    });
  },

  answerInterviewStream(
    sessionId: string,
    audio: Blob,
    filename: string,
    onMessage: (event: TokenStreamEvent) => void,
    signal?: AbortSignal,
  ): Promise<void> {
    return streamNdjson<TokenStreamEvent>({
      input: `${API_BASE_URL}/v1/interview/answer/stream`,
      init: {
        method: "POST",
        headers: withSessionHeaders(sessionId),
        body: buildAudioFormData(audio, filename),
        signal,
      },
      onMessage,
    });
  },
};

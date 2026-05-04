import {
  Activity,
  ArrowRight,
  AudioLines,
  BrainCircuit,
  Bug,
  FileJson,
  RefreshCw,
  RotateCcw,
  Server,
  SquareTerminal,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AnswerComposer } from "./components/AnswerComposer";
import { MetricCard } from "./components/MetricCard";
import { Panel } from "./components/Panel";
import { TranscriptFeed } from "./components/TranscriptFeed";
import { useAudioRecorder } from "./hooks/useAudioRecorder";
import { API_BASE_URL, smartInterviewerApi } from "./lib/api";
import type {
  AudioSelection,
  EventLogEntry,
  SessionState,
  StreamingState,
  StreamMode,
  TokenStreamEvent,
  TranscriptEntry,
} from "./types";

const SESSION_STORAGE_KEY = "smart-interviewer.react.session-id";
const INITIAL_STREAMING_STATE: StreamingState = {
  active: false,
  mode: null,
  questionText: "",
  evaluationText: "",
  followupText: "",
};

type HealthState = {
  status: "idle" | "loading" | "ok" | "error";
  message: string;
};

function createId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function createSessionId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `session-${Date.now()}`;
}

function toErrorMessage(caughtError: unknown): string {
  if (caughtError instanceof Error) {
    return caughtError.message;
  }
  return "Unexpected request failure.";
}

function progressLabel(session: SessionState | null): string {
  if (!session) {
    return "Q 0/0";
  }

  const activeIndex = session.interview_done
    ? session.batch_size
    : Math.min(session.batch_index + 1, Math.max(1, session.batch_size));

  return `Q ${activeIndex}/${session.batch_size}`;
}

function makeAudioSelection(file: File): AudioSelection {
  return {
    blob: file,
    fileName: file.name,
    mimeType: file.type || "audio/webm",
    size: file.size,
    source: "upload",
    previewUrl: URL.createObjectURL(file),
  };
}

function decodeBase64ToBlob(base64: string, contentType: string): Blob {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return new Blob([bytes], { type: contentType });
}

function formatDateTime(value: string | undefined): string {
  if (!value) {
    return "Not available";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function hasSummaryAsset(sessionState: SessionState | null): boolean {
  return !!sessionState?.summary?.data_base64;
}

function App() {
  const [sessionId, setSessionId] = useState(() => {
    const saved = window.localStorage.getItem(SESSION_STORAGE_KEY);
    return saved || createSessionId();
  });
  const [sessionState, setSessionState] = useState<SessionState | null>(null);
  const [health, setHealth] = useState<HealthState>({
    status: "idle",
    message: "Unknown",
  });
  const [pendingAction, setPendingAction] = useState<string | null>(null);
  const [streaming, setStreaming] = useState<StreamingState>(INITIAL_STREAMING_STATE);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [eventLog, setEventLog] = useState<EventLogEntry[]>([]);
  const [selectedAudio, setSelectedAudio] = useState<AudioSelection | null>(null);
  const [uiError, setUiError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"events" | "debug">("events");

  const activeAbortController = useRef<AbortController | null>(null);
  const selectedAudioRef = useRef<AudioSelection | null>(null);

  const recorder = useAudioRecorder();

  const setAudioSelection = useCallback((nextSelection: AudioSelection | null) => {
    const previous = selectedAudioRef.current;
    if (previous?.previewUrl) {
      URL.revokeObjectURL(previous.previewUrl);
    }
    selectedAudioRef.current = nextSelection;
    setSelectedAudio(nextSelection);
  }, []);

  useEffect(() => {
    window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }, [sessionId]);

  useEffect(() => {
    return () => {
      if (selectedAudioRef.current?.previewUrl) {
        URL.revokeObjectURL(selectedAudioRef.current.previewUrl);
      }
      activeAbortController.current?.abort();
    };
  }, []);

  const pushTranscriptEntry = useCallback((entry: Omit<TranscriptEntry, "id" | "timestamp">) => {
    setTranscript((current) => [
      ...current,
      {
        ...entry,
        id: createId("transcript"),
        timestamp: new Date().toISOString(),
      },
    ]);
  }, []);

  const pushEvent = useCallback((entry: Omit<EventLogEntry, "id" | "timestamp">) => {
    setEventLog((current) => {
      const nextEntries = [
        ...current,
        {
          ...entry,
          id: createId("event"),
          timestamp: new Date().toISOString(),
        },
      ];
      return nextEntries.slice(-80);
    });
  }, []);

  const abortActiveStream = useCallback(() => {
    if (activeAbortController.current) {
      activeAbortController.current.abort();
      activeAbortController.current = null;
    }
  }, []);

  const refreshHealth = useCallback(async () => {
    setHealth({ status: "loading", message: "Checking API" });

    try {
      const response = await smartInterviewerApi.health();
      setHealth({
        status: "ok",
        message: `${response.service || "SmartInterviewer"} ${response.version || ""}`.trim(),
      });
    } catch (caughtError) {
      setHealth({
        status: "error",
        message: toErrorMessage(caughtError),
      });
    }
  }, []);

  const refreshSessionState = useCallback(async () => {
    setPendingAction("state");
    setUiError(null);

    try {
      const nextState = await smartInterviewerApi.getSessionState(sessionId);
      setSessionState(nextState);
      pushEvent({
        type: "system",
        label: "Session state loaded",
        detail: `Phase ${nextState.phase || "unknown"}`,
      });
    } catch (caughtError) {
      const message = toErrorMessage(caughtError);
      setUiError(message);
      pushEvent({
        type: "error",
        label: "Session state failed",
        detail: message,
      });
    } finally {
      setPendingAction(null);
    }
  }, [pushEvent, sessionId]);

  useEffect(() => {
    void refreshHealth();
    void refreshSessionState();
  }, [refreshHealth, refreshSessionState]);

  const runStreamingAction = useCallback(
    async (
      mode: Exclude<StreamMode, null>,
      request: (signal: AbortSignal, onMessage: (event: TokenStreamEvent) => void) => Promise<void>,
      onFinalState: (
        finalState: SessionState,
        buffers: Pick<StreamingState, "questionText" | "evaluationText" | "followupText">,
      ) => void | Promise<void>,
    ): Promise<SessionState | null> => {
      abortActiveStream();
      setUiError(null);

      const controller = new AbortController();
      activeAbortController.current = controller;

      let questionText = "";
      let evaluationText = "";
      let followupText = "";
      let finalState: SessionState | null = null;
      const seenEventTypes = new Set<string>();

      setStreaming({
        active: true,
        mode,
        questionText: "",
        evaluationText: "",
        followupText: "",
      });

      pushEvent({
        type: "action",
        label: `${mode} stream opened`,
        detail: `Session ${sessionId}`,
      });

      try {
        await request(controller.signal, (event) => {
          if (!seenEventTypes.has(event.type)) {
            seenEventTypes.add(event.type);
            pushEvent({
              type: "stream",
              label: event.type,
              detail: "Streaming event received",
            });
          }

          if (event.type === "question_token") {
            questionText += event.token;
            setStreaming((current) => ({ ...current, questionText }));
            return;
          }

          if (event.type === "evaluation_token") {
            evaluationText += event.token;
            setStreaming((current) => ({ ...current, evaluationText }));
            return;
          }

          if (event.type === "followup_token") {
            followupText += event.token;
            setStreaming((current) => ({ ...current, followupText }));
            return;
          }

          if (event.type === "final_state") {
            const nextState = event.data as SessionState;
            finalState = nextState;
            setSessionState(nextState);
            pushEvent({
              type: "stream",
              label: "final_state",
              detail: `Phase ${nextState.phase || "unknown"}`,
            });
          }
        });

        if (!finalState) {
          finalState = await smartInterviewerApi.getSessionState(sessionId);
          setSessionState(finalState);
        }

        await onFinalState(finalState, {
          questionText,
          evaluationText,
          followupText,
        });
        return finalState;
      } catch (caughtError) {
        if (controller.signal.aborted) {
          pushEvent({
            type: "system",
            label: `${mode} stream aborted`,
            detail: "The active request was cancelled.",
          });
          return null;
        }

        const message = toErrorMessage(caughtError);
        setUiError(message);
        pushTranscriptEntry({
          role: "error",
          title: "Request Failed",
          content: message,
        });
        pushEvent({
          type: "error",
          label: `${mode} stream failed`,
          detail: message,
        });
        return null;
      } finally {
        if (activeAbortController.current === controller) {
          activeAbortController.current = null;
        }
        setStreaming(INITIAL_STREAMING_STATE);
      }
    },
    [abortActiveStream, pushEvent, pushTranscriptEntry, sessionId],
  );

  const clearActiveInterviewView = useCallback(() => {
    setSessionState((current) => {
      if (!current) {
        return current;
      }

      return {
        ...current,
        current_question: "",
        root_question: "",
        assistant_text: "",
        transcript: "",
        current_item_id: "",
        current_context: "",
        current_objective: "",
      };
    });
    setAudioSelection(null);
  }, [setAudioSelection]);

  const handleResetSession = useCallback(async () => {
    abortActiveStream();
    setPendingAction("reset");
    setUiError(null);

    try {
      await smartInterviewerApi.resetSession(sessionId);
      const nextState = await smartInterviewerApi.getSessionState(sessionId);
      setSessionState(nextState);
      setTranscript([]);
      setEventLog([
        {
          id: createId("event"),
          type: "action",
          label: "Session reset",
          detail: `Session ${sessionId}`,
          timestamp: new Date().toISOString(),
        },
      ]);
      setAudioSelection(null);
    } catch (caughtError) {
      const message = toErrorMessage(caughtError);
      setUiError(message);
      pushEvent({
        type: "error",
        label: "Session reset failed",
        detail: message,
      });
    } finally {
      setPendingAction(null);
    }
  }, [abortActiveStream, pushEvent, sessionId, setAudioSelection]);

  const handleNewSession = useCallback(() => {
    abortActiveStream();
    setAudioSelection(null);
    setTranscript([]);
    setEventLog([]);
    setUiError(null);
    setSessionState(null);
    setStreaming(INITIAL_STREAMING_STATE);
    setSessionId(createSessionId());
  }, [abortActiveStream, setAudioSelection]);

  const handleStart = useCallback(async () => {
    clearActiveInterviewView();
    await runStreamingAction(
      "start",
      (signal, onMessage) => smartInterviewerApi.startInterviewStream(sessionId, onMessage, signal),
      (finalState, buffers) => {
        const question = finalState.current_question || buffers.questionText.trim();
        if (question) {
          pushTranscriptEntry({
            role: "assistant",
            title: "Interview Question",
            content: question,
          });
        }
      },
    );
  }, [clearActiveInterviewView, pushTranscriptEntry, runStreamingAction, sessionId]);

  const prepareFinishSummary = useCallback(async () => {
    clearActiveInterviewView();

    return runStreamingAction(
      "next",
      (signal, onMessage) => smartInterviewerApi.nextInterviewStream(sessionId, onMessage, signal),
      async (finalState) => {
        if (hasSummaryAsset(finalState)) {
          pushEvent({
            type: "system",
            label: "Summary ready",
            detail: `Final level ${finalState.final_level}`,
          });
        }
      },
    );
  }, [clearActiveInterviewView, pushEvent, runStreamingAction, sessionId]);

  const handleNext = useCallback(async () => {
    clearActiveInterviewView();
    await runStreamingAction(
      "next",
      (signal, onMessage) => smartInterviewerApi.nextInterviewStream(sessionId, onMessage, signal),
      (finalState, buffers) => {
        if (finalState.interview_done) {
          pushTranscriptEntry({
            role: "system",
            title: "Interview Complete",
            content: `Final level ${finalState.final_level}`,
          });
          return;
        }

        const question = finalState.current_question || buffers.questionText.trim();
        if (question) {
          pushTranscriptEntry({
            role: "assistant",
            title: "Next Question",
            content: question,
          });
        }
      },
    );
  }, [clearActiveInterviewView, pushTranscriptEntry, runStreamingAction, sessionId]);

  const handleStartRecording = useCallback(async () => {
    const started = await recorder.startRecording();
    if (started) {
      pushEvent({
        type: "action",
        label: "Recording started",
        detail: "Capturing microphone audio",
      });
    }
  }, [pushEvent, recorder]);

  const handleStopRecording = useCallback(async () => {
    const recordedAudio = await recorder.stopRecording();
    if (recordedAudio) {
      setAudioSelection(recordedAudio);
      pushEvent({
        type: "action",
        label: "Recording ready",
        detail: recordedAudio.fileName,
      });
    }
  }, [pushEvent, recorder, setAudioSelection]);

  const handleFileSelected = useCallback(
    (file: File | null) => {
      if (!file) {
        return;
      }
      setAudioSelection(makeAudioSelection(file));
      pushEvent({
        type: "action",
        label: "Audio file selected",
        detail: `${file.name} (${file.type || "audio/*"})`,
      });
    },
    [pushEvent, setAudioSelection],
  );

  const handleSubmitAnswer = useCallback(async () => {
    if (!selectedAudio) {
      return;
    }

    pushTranscriptEntry({
      role: "user",
      title: "Candidate Audio",
      content: `${selectedAudio.source === "recording" ? "Recorded" : "Uploaded"} answer submitted.`,
    });

    const finalState = await runStreamingAction(
      "answer",
      (signal, onMessage) =>
        smartInterviewerApi.answerInterviewStream(
          sessionId,
          selectedAudio.blob,
          selectedAudio.fileName,
          onMessage,
          signal,
        ),
      (finalState) => {
        if (finalState.transcript) {
          pushTranscriptEntry({
            role: "transcript",
            title: "Candidate Transcript",
            content: finalState.transcript,
          });
        }

        if (finalState.assistant_text) {
          pushTranscriptEntry({
            role: "evaluation",
            title: "AI Evaluation",
            content: finalState.assistant_text,
          });
        }

        const isFollowup =
          finalState.phase === "AWAITING_ANSWER" &&
          !!finalState.current_question &&
          finalState.current_question !== (finalState.root_question || "");

        if (isFollowup) {
          pushTranscriptEntry({
            role: "followup",
            title: "Follow-up Question",
            content: finalState.current_question,
          });
        }

        if (finalState.interview_done) {
          pushTranscriptEntry({
            role: "system",
            title: "Interview Complete",
            content: `Final level ${finalState.final_level}`,
          });
        }

        setAudioSelection(null);
      },
    );

    if (finalState?.interview_done && !hasSummaryAsset(finalState)) {
      await prepareFinishSummary();
    }
  }, [
    prepareFinishSummary,
    pushTranscriptEntry,
    runStreamingAction,
    selectedAudio,
    sessionId,
    setAudioSelection,
  ]);

  const handleFinish = useCallback(async () => {
    if (sessionState?.interview_done && !hasSummaryAsset(sessionState)) {
      setPendingAction("finish");
      setUiError(null);

      try {
        await prepareFinishSummary();
      } finally {
        setPendingAction(null);
      }
      return;
    }

    setPendingAction("finish");
    setUiError(null);

    try {
      const finalState = await smartInterviewerApi.finishInterview(sessionId);
      setSessionState(finalState);
      pushTranscriptEntry({
        role: "system",
        title: "Interview Finished",
        content: `Final level ${finalState.final_level}`,
      });
      pushEvent({
        type: "action",
        label: "Interview finished",
        detail: `Final level ${finalState.final_level}`,
      });
    } catch (caughtError) {
      const message = toErrorMessage(caughtError);
      setUiError(message);
      pushEvent({
        type: "error",
        label: "Finish failed",
        detail: message,
      });
    } finally {
      setPendingAction(null);
    }
  }, [prepareFinishSummary, pushEvent, pushTranscriptEntry, sessionId, sessionState]);

  const isBusy =
    streaming.active || pendingAction !== null || recorder.status === "processing";
  const canStart = !!sessionState?.allowed_actions.includes("START") && !isBusy;
  const hasSummary = hasSummaryAsset(sessionState);
  const shouldShowFinishAction =
    !isBusy &&
    (!!sessionState?.allowed_actions.includes("FINISH") ||
      (!!sessionState?.interview_done && !!sessionState?.allowed_actions.includes("NEXT")));
  const canNext =
    !!sessionState?.allowed_actions.includes("NEXT") &&
    !!sessionState?.can_proceed &&
    !isBusy &&
    !sessionState?.interview_done;
  const canFinish = shouldShowFinishAction;
  const canSubmitAnswer =
    !!selectedAudio &&
    !!sessionState?.allowed_actions.includes("ANSWER") &&
    !sessionState?.interview_done &&
    !isBusy &&
    !hasSummary;

  const summaryDownload = useMemo(() => {
    const summary = sessionState?.summary;
    if (!summary?.data_base64) {
      return null;
    }

    const blob = decodeBase64ToBlob(summary.data_base64, summary.content_type || "application/json");
    return {
      fileName: summary.filename || "interview_summary.json",
      url: URL.createObjectURL(blob),
    };
  }, [sessionState?.summary]);

  useEffect(() => {
    return () => {
      if (summaryDownload?.url) {
        URL.revokeObjectURL(summaryDownload.url);
      }
    };
  }, [summaryDownload]);

  const liveQuestion = streaming.questionText || sessionState?.current_question || "";
  const liveEvaluation =
    streaming.active && streaming.mode === "answer"
      ? "Evaluating your answer..."
      : sessionState?.assistant_text || "";
  const showSummaryView = hasSummary;
  const showInterviewFlowPanels = !showSummaryView;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8 space-y-5">
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-xs font-medium uppercase tracking-[0.18em] text-cyan-200">
              <Activity className="h-3.5 w-3.5" />
              Smart Interviewer
            </div>
            <div>
              <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Adaptive interview flow with streamed questions and evaluation
              </h1>
              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-400">
                This React frontend stays thin by driving the existing FastAPI session model and
                NDJSON stream endpoints directly.
              </p>
            </div>
          </div>

          <section className="rounded-3xl border border-white/10 bg-slate-900/75 p-5 shadow-[0_24px_80px_rgba(15,23,42,0.35)] backdrop-blur">
            <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(24rem,0.9fr)]">
              <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-5">
                <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
                  Product Demo
                </p>
                <p className="mt-3 text-lg font-semibold text-white">
                  FastAPI interview engine, streamed prompts, browser audio capture, and a thin
                  React client on top of the existing session model.
                </p>
                <div className="mt-5 grid gap-3 sm:grid-cols-3">
                  <div className="rounded-xl border border-white/8 bg-slate-950/40 px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Phase</p>
                    <p className="mt-2 text-sm font-medium text-slate-100">
                      {sessionState?.phase || "Unknown"}
                    </p>
                  </div>
                  <div className="rounded-xl border border-white/8 bg-slate-950/40 px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Turn</p>
                    <p className="mt-2 text-sm font-medium text-slate-100">
                      {sessionState?.turn ?? 0}
                    </p>
                  </div>
                  <div className="rounded-xl border border-white/8 bg-slate-950/40 px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Mode</p>
                    <p className="mt-2 text-sm font-medium text-slate-100">
                      {hasSummary ? "Summary" : streaming.active ? "Streaming" : "Interactive"}
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-2xl border border-cyan-400/15 bg-cyan-400/[0.06] p-5">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="flex items-center gap-2 text-sm font-medium text-slate-100">
                        <Server className="h-4 w-4 text-cyan-300" />
                        API Status
                      </p>
                      <p className="mt-2 break-all text-sm leading-6 text-slate-400">
                        {API_BASE_URL}
                      </p>
                    </div>
                    <span
                      className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${
                        health.status === "ok"
                          ? "bg-emerald-400/12 text-emerald-200"
                          : health.status === "error"
                            ? "bg-rose-400/12 text-rose-200"
                            : "bg-white/10 text-slate-300"
                      }`}
                    >
                      {health.status}
                    </span>
                  </div>
                  <p className="mt-4 min-h-10 text-sm text-slate-400">{health.message}</p>
                  <button
                    type="button"
                    onClick={() => void refreshHealth()}
                    disabled={health.status === "loading"}
                    className="mt-4 inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:bg-white/[0.04] disabled:cursor-not-allowed disabled:text-slate-500"
                  >
                    <RefreshCw className={`h-4 w-4 ${health.status === "loading" ? "animate-spin" : ""}`} />
                    Refresh Health
                  </button>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
                  <p className="text-sm font-medium text-slate-100">Session Control</p>
                  <p className="mt-2 text-xs uppercase tracking-[0.14em] text-slate-500">
                    Active session
                  </p>
                  <p className="mt-2 break-all text-sm leading-6 text-slate-300">{sessionId}</p>
                  <div className="mt-4 grid gap-2 sm:grid-cols-2">
                    <button
                      type="button"
                      onClick={() => void refreshSessionState()}
                      disabled={isBusy}
                      className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:bg-white/[0.04] disabled:cursor-not-allowed disabled:text-slate-500"
                    >
                      <RefreshCw className="h-4 w-4" />
                      Reload
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleResetSession()}
                      disabled={isBusy}
                      className="inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:bg-white/[0.04] disabled:cursor-not-allowed disabled:text-slate-500"
                    >
                      <RotateCcw className="h-4 w-4" />
                      Reset
                    </button>
                    <button
                      type="button"
                      onClick={handleNewSession}
                      disabled={isBusy}
                      className="sm:col-span-2 inline-flex items-center justify-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-slate-200 transition hover:bg-white/[0.04] disabled:cursor-not-allowed disabled:text-slate-500"
                    >
                      <ArrowRight className="h-4 w-4" />
                      New Session
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </header>

        {uiError ? (
          <div className="mb-6 rounded-2xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
            {uiError}
          </div>
        ) : null}

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.7fr)_22rem]">
          <main className="space-y-6">
            <Panel
              title="Interview Controls"
              subtitle="Start, advance, finish, or reset the active interview session."
              actions={
                <div className="flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => void handleStart()}
                    disabled={!canStart}
                    className="rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-4 py-2 text-sm font-medium text-cyan-100 transition hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/[0.05] disabled:text-slate-500"
                  >
                    Start
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleNext()}
                    disabled={!canNext}
                    hidden={!canNext}
                    className="rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:text-slate-500"
                  >
                    Next
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleFinish()}
                    disabled={!canFinish}
                    hidden={!shouldShowFinishAction}
                    className="rounded-xl border border-violet-400/30 bg-violet-400/12 px-4 py-2 text-sm font-medium text-violet-100 transition hover:bg-violet-400/20 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/[0.05] disabled:text-slate-500"
                  >
                    Finish
                  </button>
                </div>
              }
            >
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <MetricCard label="Testing Level" value={String(sessionState?.current_level ?? 1)} />
                <MetricCard label="Progress" value={progressLabel(sessionState)} />
                <MetricCard
                  label="Correct This Level"
                  value={String(sessionState?.batch_correct ?? 0)}
                />
                <MetricCard
                  label="Last Passed"
                  value={String(sessionState?.last_passed_level ?? 0)}
                />
              </div>
            </Panel>

            {showSummaryView ? (
              <Panel
                title="Interview Summary"
                subtitle="The interview is complete. Review the final state and download the summary artifact."
              >
                <div className="space-y-4">
                  <div className="rounded-2xl border border-emerald-400/20 bg-emerald-400/10 p-5">
                    <p className="text-sm font-medium text-emerald-100">
                      Final level: {sessionState?.final_level ?? sessionState?.last_passed_level ?? 0}
                    </p>
                    <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-emerald-50">
                      {sessionState?.assistant_text || "Interview complete."}
                    </p>
                  </div>
                  {summaryDownload ? (
                    <a
                      href={summaryDownload.url}
                      download={summaryDownload.fileName}
                      className="inline-flex items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/12 px-4 py-2.5 text-sm font-medium text-emerald-100 transition hover:bg-emerald-400/18"
                    >
                      Download Summary JSON
                    </a>
                  ) : null}
                </div>
              </Panel>
            ) : null}

            {showInterviewFlowPanels ? (
              <>
                <Panel
                  title="Current Question"
                  subtitle="Question text is updated progressively while the backend stream is open."
                >
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
                    <div className="flex items-center gap-3 text-sm text-slate-400">
                      <AudioLines className="h-4 w-4 text-cyan-300" />
                      <span>
                        {streaming.active && (streaming.mode === "start" || streaming.mode === "next")
                          ? "Streaming question"
                          : "Latest prompt"}
                      </span>
                    </div>
                    <p className="mt-4 whitespace-pre-wrap text-lg leading-8 text-slate-100">
                      {liveQuestion || "Start the interview to request the first question."}
                    </p>
                  </div>
                  {streaming.followupText ? (
                    <div className="mt-4 rounded-2xl border border-amber-400/20 bg-amber-400/10 p-4 text-sm text-amber-100">
                      {streaming.followupText}
                    </div>
                  ) : null}
                </Panel>

                <Panel
                  title="Answer Submission"
                  subtitle="Record from the browser or upload an audio file, then stream the evaluation response."
                >
                  <AnswerComposer
                    selectedAudio={selectedAudio}
                    busy={isBusy}
                    canSubmit={canSubmitAnswer}
                    recordingSupported={recorder.supported}
                    recorderStatus={recorder.status}
                    recorderError={recorder.error}
                    onStartRecording={handleStartRecording}
                    onStopRecording={handleStopRecording}
                    onFileSelected={handleFileSelected}
                    onClearSelection={() => setAudioSelection(null)}
                    onSubmit={handleSubmitAnswer}
                  />
                </Panel>

                <Panel
                  title="Evaluation Stream"
                  subtitle="Streaming feedback from the backend evaluation node."
                >
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-5">
                    <div className="flex items-center gap-3 text-sm text-slate-400">
                      <BrainCircuit className="h-4 w-4 text-emerald-300" />
                      <span>
                        {streaming.active && streaming.mode === "answer"
                          ? "Evaluation in progress"
                          : "Latest backend feedback"}
                      </span>
                    </div>
                    <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-slate-100">
                      {liveEvaluation || "Submit an answer to stream evaluation feedback here."}
                    </p>
                  </div>
                </Panel>

                <Panel
                  title="Transcript & History"
                  subtitle="Frontend-kept session history for questions, transcripts, evaluation, and follow-ups."
                >
                  <TranscriptFeed
                    entries={transcript}
                    emptyTitle="No transcript entries yet"
                    emptyBody="The interview feed fills in as soon as you start the session and send audio answers."
                  />
                </Panel>
              </>
            ) : null}
          </main>

          <aside className="space-y-6">
            <Panel title="Session State" subtitle="High-level session diagnostics from the API.">
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
                  <span className="text-slate-400">Phase</span>
                  <span className="font-medium text-slate-100">{sessionState?.phase || "Unknown"}</span>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
                  <span className="text-slate-400">Turn</span>
                  <span className="font-medium text-slate-100">{sessionState?.turn ?? 0}</span>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
                  <span className="text-slate-400">Allowed Actions</span>
                  <span className="text-right font-medium text-slate-100">
                    {sessionState?.allowed_actions.join(", ") || "None"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
                  <span className="text-slate-400">Can Proceed</span>
                  <span className="font-medium text-slate-100">
                    {sessionState?.can_proceed ? "Yes" : "No"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
                  <span className="text-slate-400">Interview Done</span>
                  <span className="font-medium text-slate-100">
                    {sessionState?.interview_done ? "Yes" : "No"}
                  </span>
                </div>
              </div>
            </Panel>

            <Panel title="Question Metadata" subtitle="Fields already present in the interview state.">
              <div className="space-y-4 text-sm text-slate-300">
                <div>
                  <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Item ID</p>
                  <p className="mt-2 break-words text-slate-100">
                    {sessionState?.current_item_id || "Not exposed yet"}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Objective</p>
                  <p className="mt-2 whitespace-pre-wrap text-slate-100">
                    {sessionState?.current_objective || "No objective on the current state."}
                  </p>
                </div>
                <div>
                  <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Context</p>
                  <p className="mt-2 whitespace-pre-wrap text-slate-100">
                    {sessionState?.current_context || "No context on the current state."}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Follow-ups</p>
                    <p className="mt-2 text-base font-semibold text-slate-100">
                      {sessionState?.followups_used ?? 0}/{sessionState?.max_followups ?? 0}
                    </p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-white/[0.03] px-4 py-3">
                    <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Last Result</p>
                    <p className="mt-2 text-base font-semibold text-slate-100">
                      {typeof sessionState?.last_correct === "boolean"
                        ? sessionState.last_correct
                          ? "Correct"
                          : "Incorrect"
                        : "Pending"}
                    </p>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="Evaluation Summary" subtitle="Latest evaluation and session timing.">
              <div className="space-y-4 text-sm">
                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                  <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Last Reason</p>
                  <p className="mt-2 whitespace-pre-wrap text-slate-100">
                    {sessionState?.last_reason || "No evaluation reason available yet."}
                  </p>
                </div>
                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                  <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Started</p>
                  <p className="mt-2 text-slate-100">{formatDateTime(sessionState?.started_at)}</p>
                </div>
                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                  <p className="text-xs uppercase tracking-[0.16em] text-slate-500">Finished</p>
                  <p className="mt-2 text-slate-100">{formatDateTime(sessionState?.finished_at)}</p>
                </div>
                {summaryDownload ? (
                  <a
                    href={summaryDownload.url}
                    download={summaryDownload.fileName}
                    className="inline-flex w-full items-center justify-center rounded-xl border border-emerald-400/30 bg-emerald-400/12 px-4 py-2.5 text-sm font-medium text-emerald-100 transition hover:bg-emerald-400/18"
                  >
                    Download Summary JSON
                  </a>
                ) : null}
              </div>
            </Panel>

            <Panel
              title="Development Panels"
              subtitle="Use the tabs below for stream events and raw session state."
            >
              <div className="mb-4 inline-flex rounded-xl border border-white/10 bg-white/[0.03] p-1">
                <button
                  type="button"
                  onClick={() => setActiveTab("events")}
                  className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                    activeTab === "events"
                      ? "bg-white/[0.08] text-slate-100"
                      : "text-slate-400"
                  }`}
                >
                  <SquareTerminal className="h-4 w-4" />
                  Events
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab("debug")}
                  className={`inline-flex items-center gap-2 rounded-lg px-3 py-2 text-sm ${
                    activeTab === "debug"
                      ? "bg-white/[0.08] text-slate-100"
                      : "text-slate-400"
                  }`}
                >
                  <Bug className="h-4 w-4" />
                  Debug
                </button>
              </div>

              {activeTab === "events" ? (
                <div className="max-h-[28rem] space-y-3 overflow-y-auto pr-1">
                  {eventLog.length ? (
                    eventLog
                      .slice()
                      .reverse()
                      .map((entry) => (
                        <div
                          key={entry.id}
                          className="rounded-xl border border-white/10 bg-white/[0.03] p-4"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2 text-sm font-medium text-slate-100">
                              <SquareTerminal className="h-4 w-4 text-cyan-300" />
                              {entry.label}
                            </div>
                            <span className="text-xs text-slate-500">
                              {new Intl.DateTimeFormat(undefined, {
                                hour: "2-digit",
                                minute: "2-digit",
                                second: "2-digit",
                              }).format(new Date(entry.timestamp))}
                            </span>
                          </div>
                          <p className="mt-2 text-sm text-slate-400">{entry.detail}</p>
                        </div>
                      ))
                  ) : (
                    <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] px-4 py-10 text-center text-sm text-slate-400">
                      No client-side stream events captured yet.
                    </div>
                  )}
                </div>
              ) : (
                <div className="max-h-[28rem] space-y-4 overflow-y-auto pr-1">
                  <div className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
                      <FileJson className="h-4 w-4 text-cyan-300" />
                      Raw session state
                    </div>
                    <pre className="max-h-[28rem] overflow-auto whitespace-pre-wrap break-words text-xs leading-6 text-slate-300">
                      {JSON.stringify(sessionState, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </Panel>
          </aside>
        </div>
      </div>
    </div>
  );
}

export default App;

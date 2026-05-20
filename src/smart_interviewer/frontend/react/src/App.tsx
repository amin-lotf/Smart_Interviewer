import {
  Activity,
  ArrowRight,
  AudioLines,
  BrainCircuit,
  Bug,
  CircleHelp,
  FileJson,
  Flag,
  ListChecks,
  MessageSquareText,
  Play,
  RefreshCw,
  RotateCcw,
  Server,
  Sparkles,
  SquareTerminal,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { AnswerComposer } from "./components/AnswerComposer";
import { Drawer } from "./components/Drawer";
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
const QUESTION_VOICE_STORAGE_KEY = "smart-interviewer.react.voice-enabled";
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

type SecondaryPanel = "api" | "session" | "transcript" | "debug" | "question";

interface DetailFieldProps {
  label: string;
  value: string;
  multiline?: boolean;
}

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

function formatClockTime(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function hasSummaryAsset(sessionState: SessionState | null): boolean {
  return !!sessionState?.summary?.data_base64;
}

function DetailField({ label, value, multiline = false }: DetailFieldProps) {
  return (
    <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
      <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p
        className={`mt-2 text-sm leading-7 text-slate-700 ${
          multiline ? "whitespace-pre-wrap" : "break-words"
        }`}
      >
        {value}
      </p>
    </div>
  );
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
  const [activePanel, setActivePanel] = useState<SecondaryPanel | null>(null);
  const [voiceEnabled, setVoiceEnabled] = useState(() => {
    const saved = window.localStorage.getItem(QUESTION_VOICE_STORAGE_KEY);
    return saved ? saved === "true" : true;
  });

  const activeAbortController = useRef<AbortController | null>(null);
  const selectedAudioRef = useRef<AudioSelection | null>(null);
  const questionAudioRef = useRef<HTMLAudioElement | null>(null);
  const questionAudioUrlRef = useRef<string | null>(null);
  const questionAudioAbortRef = useRef<AbortController | null>(null);
  const lastSpokenQuestionKeyRef = useRef<string>("");

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
    window.localStorage.setItem(QUESTION_VOICE_STORAGE_KEY, String(voiceEnabled));
  }, [voiceEnabled]);

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

  const stopQuestionAudio = useCallback(() => {
    if (questionAudioAbortRef.current) {
      questionAudioAbortRef.current.abort();
      questionAudioAbortRef.current = null;
    }

    if (questionAudioRef.current) {
      questionAudioRef.current.pause();
      questionAudioRef.current.src = "";
      questionAudioRef.current = null;
    }

    if (questionAudioUrlRef.current) {
      URL.revokeObjectURL(questionAudioUrlRef.current);
      questionAudioUrlRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (selectedAudioRef.current?.previewUrl) {
        URL.revokeObjectURL(selectedAudioRef.current.previewUrl);
      }
      activeAbortController.current?.abort();
      stopQuestionAudio();
    };
  }, [stopQuestionAudio]);

  useEffect(() => {
    if (!voiceEnabled) {
      stopQuestionAudio();
    }
  }, [stopQuestionAudio, voiceEnabled]);

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

  const playQuestionAudio = useCallback(
    async ({
      enabled,
      question,
      questionKey,
      label,
    }: {
      enabled: boolean;
      question: string;
      questionKey: string;
      label: string;
    }) => {
      const trimmedQuestion = question.trim();
      if (!voiceEnabled || !enabled || !trimmedQuestion || !questionKey) {
        return;
      }

      if (lastSpokenQuestionKeyRef.current === questionKey) {
        return;
      }

      stopQuestionAudio();

      const controller = new AbortController();
      questionAudioAbortRef.current = controller;

      try {
        const audioBlob = await smartInterviewerApi.fetchQuestionSpeech(
          sessionId,
          trimmedQuestion,
          controller.signal,
        );
        if (controller.signal.aborted || !audioBlob) {
          return;
        }

        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        questionAudioUrlRef.current = audioUrl;
        questionAudioRef.current = audio;

        audio.onended = () => {
          if (questionAudioRef.current === audio) {
            questionAudioRef.current = null;
          }
          if (questionAudioUrlRef.current === audioUrl) {
            URL.revokeObjectURL(audioUrl);
            questionAudioUrlRef.current = null;
          }
        };

        await audio.play();
        lastSpokenQuestionKeyRef.current = questionKey;
        pushEvent({
          type: "system",
          label: "Question audio",
          detail: `${label} playback started`,
        });
      } catch (caughtError) {
        if (controller.signal.aborted) {
          return;
        }

        const message = toErrorMessage(caughtError);
        stopQuestionAudio();
        pushEvent({
          type: "error",
          label: "Question audio failed",
          detail: message,
        });
      } finally {
        if (questionAudioAbortRef.current === controller) {
          questionAudioAbortRef.current = null;
        }
      }
    },
    [pushEvent, sessionId, stopQuestionAudio, voiceEnabled],
  );

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
    stopQuestionAudio();
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
  }, [setAudioSelection, stopQuestionAudio]);

  const handleResetSession = useCallback(async () => {
    abortActiveStream();
    stopQuestionAudio();
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
      lastSpokenQuestionKeyRef.current = "";
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
  }, [abortActiveStream, pushEvent, sessionId, setAudioSelection, stopQuestionAudio]);

  const handleNewSession = useCallback(() => {
    abortActiveStream();
    stopQuestionAudio();
    setAudioSelection(null);
    setTranscript([]);
    setEventLog([]);
    setUiError(null);
    setSessionState(null);
    setStreaming(INITIAL_STREAMING_STATE);
    setSessionId(createSessionId());
    setActivePanel(null);
    lastSpokenQuestionKeyRef.current = "";
  }, [abortActiveStream, setAudioSelection, stopQuestionAudio]);

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
          void playQuestionAudio({
            enabled: !!finalState.tts_enabled,
            question,
            questionKey: `${finalState.turn}:${finalState.followups_used ?? 0}:${question}`,
            label: "Interview question",
          });
        }
      },
    );
  }, [clearActiveInterviewView, playQuestionAudio, pushTranscriptEntry, runStreamingAction, sessionId]);

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
          void playQuestionAudio({
            enabled: !!finalState.tts_enabled,
            question,
            questionKey: `${finalState.turn}:${finalState.followups_used ?? 0}:${question}`,
            label: "Next question",
          });
        }
      },
    );
  }, [clearActiveInterviewView, playQuestionAudio, pushTranscriptEntry, runStreamingAction, sessionId]);

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

    stopQuestionAudio();

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
          void playQuestionAudio({
            enabled: !!finalState.tts_enabled,
            question: finalState.current_question,
            questionKey: `${finalState.turn}:${finalState.followups_used ?? 0}:${finalState.current_question}`,
            label: "Follow-up question",
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
    playQuestionAudio,
    pushTranscriptEntry,
    runStreamingAction,
    selectedAudio,
    sessionId,
    setAudioSelection,
    stopQuestionAudio,
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

  const isBusy = streaming.active || pendingAction !== null || recorder.status === "processing";
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
      ? streaming.evaluationText
      : sessionState?.assistant_text || "";
  const showSummaryView = hasSummary;
  const voiceAvailable = !!sessionState?.tts_enabled;
  const progress = progressLabel(sessionState);
  const lastResult =
    typeof sessionState?.last_correct === "boolean"
      ? sessionState.last_correct
        ? "Correct"
        : "Incorrect"
      : "Pending";
  const helperButtonBase =
    "inline-flex min-h-11 items-center gap-2 rounded-full border px-4 py-2.5 text-sm font-medium transition";
  const panelToggleBase =
    "inline-flex items-center gap-2 rounded-full px-4 py-2.5 text-sm font-medium transition";
  const helperButtons = [
    { key: "api" as const, icon: Server, label: "API Status" },
    { key: "session" as const, icon: ListChecks, label: "Session State" },
    { key: "transcript" as const, icon: MessageSquareText, label: "Transcript" },
    { key: "debug" as const, icon: Bug, label: "Debug Details" },
  ];

  const drawerTitle =
    activePanel === "api"
      ? "API Status"
      : activePanel === "session"
        ? "Session State"
        : activePanel === "transcript"
          ? "Transcript"
          : activePanel === "debug"
            ? "Debug Details"
            : activePanel === "question"
              ? "Question Details"
              : "";

  const drawerSubtitle =
    activePanel === "api"
      ? "Health checks and endpoint details stay here instead of in the main interview flow."
      : activePanel === "session"
        ? "Session progress, timing, and state details from the FastAPI backend."
        : activePanel === "transcript"
          ? "Questions, transcripts, evaluations, and follow-ups captured by the React client."
          : activePanel === "debug"
            ? "Stream events and raw payloads for troubleshooting."
            : activePanel === "question"
              ? "Metadata for the current prompt, hidden by default during the demo."
              : "";

  const renderDrawerContent = () => {
    if (activePanel === "api") {
      return (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <DetailField label="Endpoint" value={API_BASE_URL} />
            <DetailField
              label="Health"
              value={
                health.status === "ok"
                  ? `Connected: ${health.message}`
                  : health.status === "error"
                    ? `Unavailable: ${health.message}`
                    : health.status === "loading"
                      ? "Checking API status"
                      : "Unknown"
              }
            />
            <DetailField
              label="Question Voice"
              value={
                voiceAvailable
                  ? voiceEnabled
                    ? "Available and enabled"
                    : "Available and muted"
                  : "Not available for this session"
              }
            />
            <DetailField
              label="Current Mode"
              value={showSummaryView ? "Summary" : streaming.active ? "Streaming" : "Interactive"}
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void refreshHealth()}
              disabled={health.status === "loading"}
              className="inline-flex min-h-11 items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
            >
              <RefreshCw className={`h-4 w-4 ${health.status === "loading" ? "animate-spin" : ""}`} />
              Refresh Health
            </button>
          </div>
        </div>
      );
    }

    if (activePanel === "session") {
      return (
        <div className="space-y-6">
          <div className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
            <p className="text-xs font-medium uppercase tracking-[0.18em] text-slate-500">
              Session ID
            </p>
            <p className="mt-3 break-all text-sm leading-7 text-slate-700">{sessionId}</p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <MetricCard label="Phase" value={sessionState?.phase || "Unknown"} />
            <MetricCard label="Progress" value={progress} />
            <MetricCard label="Turn" value={String(sessionState?.turn ?? 0)} />
            <MetricCard label="Current Level" value={String(sessionState?.current_level ?? 1)} />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <DetailField
              label="Allowed Actions"
              value={sessionState?.allowed_actions.join(", ") || "None"}
              multiline
            />
            <DetailField
              label="Flow State"
              value={`Can proceed: ${sessionState?.can_proceed ? "Yes" : "No"}\nInterview done: ${
                sessionState?.interview_done ? "Yes" : "No"
              }`}
              multiline
            />
            <DetailField label="Started" value={formatDateTime(sessionState?.started_at)} />
            <DetailField label="Finished" value={formatDateTime(sessionState?.finished_at)} />
            <DetailField label="Last Result" value={lastResult} />
            <DetailField
              label="Latest Reason"
              value={sessionState?.last_reason || "No evaluation reason available yet."}
              multiline
            />
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void refreshSessionState()}
              disabled={isBusy}
              className="inline-flex min-h-11 items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
            >
              <RefreshCw className="h-4 w-4" />
              Reload Session
            </button>
            {summaryDownload ? (
              <a
                href={summaryDownload.url}
                download={summaryDownload.fileName}
                className="inline-flex min-h-11 items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-4 py-2.5 text-sm font-medium text-emerald-700 transition hover:bg-emerald-100"
              >
                Download Summary JSON
              </a>
            ) : null}
          </div>
        </div>
      );
    }

    if (activePanel === "transcript") {
      return (
        <TranscriptFeed
          entries={transcript}
          emptyTitle="No transcript entries yet"
          emptyBody="The interview feed fills in after you start the session and submit candidate audio."
        />
      );
    }

    if (activePanel === "debug") {
      return (
        <div className="space-y-4">
          <div className="inline-flex rounded-full border border-slate-200 bg-slate-100 p-1">
            <button
              type="button"
              onClick={() => setActiveTab("events")}
              className={`${panelToggleBase} ${
                activeTab === "events"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              <SquareTerminal className="h-4 w-4" />
              Events
            </button>
            <button
              type="button"
              onClick={() => setActiveTab("debug")}
              className={`${panelToggleBase} ${
                activeTab === "debug"
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              <FileJson className="h-4 w-4" />
              Raw State
            </button>
          </div>

          {activeTab === "events" ? (
            <div className="space-y-3">
              {eventLog.length ? (
                eventLog
                  .slice()
                  .reverse()
                  .map((entry) => (
                    <div key={entry.id} className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                          <SquareTerminal className="h-4 w-4 text-sky-600" />
                          {entry.label}
                        </div>
                        <span className="text-xs text-slate-500">{formatClockTime(entry.timestamp)}</span>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-slate-600">{entry.detail}</p>
                    </div>
                  ))
              ) : (
                <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-4 py-10 text-center text-sm text-slate-600">
                  No client-side stream events captured yet.
                </div>
              )}
            </div>
          ) : (
            <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
              <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-slate-900">
                <FileJson className="h-4 w-4 text-sky-600" />
                Raw session state
              </div>
              <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap break-words rounded-2xl bg-white p-4 text-xs leading-6 text-slate-700">
                {JSON.stringify(sessionState, null, 2)}
              </pre>
            </div>
          )}
        </div>
      );
    }

    if (activePanel === "question") {
      return (
        <div className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <MetricCard label="Level" value={String(sessionState?.current_level ?? 1)} />
            <MetricCard
              label="Follow-ups"
              value={`${sessionState?.followups_used ?? 0}/${sessionState?.max_followups ?? 0}`}
            />
            <MetricCard label="Last Result" value={lastResult} />
            <DetailField label="Item ID" value={sessionState?.current_item_id || "Not available"} />
          </div>

          <DetailField
            label="Objective"
            value={sessionState?.current_objective || "No objective on the current state."}
            multiline
          />
          <DetailField
            label="Context"
            value={sessionState?.current_context || "No context on the current state."}
            multiline
          />
          <DetailField
            label="Root Question"
            value={sessionState?.root_question || "No root question available yet."}
            multiline
          />
          <DetailField
            label="Latest Reason"
            value={sessionState?.last_reason || "No evaluation reason available yet."}
            multiline
          />
        </div>
      );
    }

    return null;
  };

  return (
    <div className="min-h-screen text-slate-900">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
        <header className="mb-8">
          <div className="rounded-[32px] border border-amber-100 bg-white/88 p-6 shadow-[0_24px_80px_rgba(148,163,184,0.14)] backdrop-blur sm:p-8">
            <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
              <div className="space-y-4">
                <div className="inline-flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-amber-700">
                  <Sparkles className="h-3.5 w-3.5" />
                  Smart Interviewer
                </div>
                <div>
                  <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
                    A cleaner interview flow for demos
                  </h1>
                  <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-600">
                    Start or continue the interview, focus on the current question, capture the
                    answer, and keep supporting details one click away.
                  </p>
                </div>
                <div className="flex flex-wrap gap-3 text-sm">
                  <span className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-sky-50 px-3 py-1.5 font-medium text-sky-700">
                    <Activity className="h-4 w-4" />
                    {progress}
                  </span>
                  <span className="inline-flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 font-medium text-amber-700">
                    Level {sessionState?.current_level ?? 1}
                  </span>
                  <span
                    className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 font-medium ${
                      health.status === "ok"
                        ? "border border-emerald-200 bg-emerald-50 text-emerald-700"
                        : health.status === "error"
                          ? "border border-rose-200 bg-rose-50 text-rose-700"
                          : "border border-slate-200 bg-slate-50 text-slate-600"
                    }`}
                  >
                    <Server className="h-4 w-4" />
                    {health.status === "ok"
                      ? "API connected"
                      : health.status === "error"
                        ? "API unavailable"
                        : "API checking"}
                  </span>
                </div>
              </div>

              <div className="flex flex-wrap gap-2 lg:max-w-xl lg:justify-end">
                {helperButtons.map(({ key, icon: Icon, label }) => (
                  <button
                    key={key}
                    type="button"
                    onClick={() => setActivePanel(key)}
                    className={`${helperButtonBase} ${
                      activePanel === key
                        ? "border-sky-200 bg-sky-50 text-sky-700"
                        : "border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    {label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </header>

        {uiError ? (
          <div className="mb-6 rounded-3xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {uiError}
          </div>
        ) : null}

        <main className="space-y-6">
          <Panel
            title="Interview Controls"
            subtitle="Keep the main actions together so the interview flow stays easy to follow."
          >
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => void handleStart()}
                disabled={!canStart}
                className="inline-flex min-h-12 items-center gap-2 rounded-full border border-emerald-200 bg-emerald-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-600 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-200 disabled:text-slate-500"
              >
                <Play className="h-4 w-4" />
                Start
              </button>
              <button
                type="button"
                onClick={() => void handleNext()}
                disabled={!canNext}
                className="inline-flex min-h-12 items-center gap-2 rounded-full border border-sky-200 bg-sky-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-600 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-200 disabled:text-slate-500"
              >
                <ArrowRight className="h-4 w-4" />
                Continue
              </button>
              <button
                type="button"
                onClick={() => void handleFinish()}
                disabled={!canFinish}
                className="inline-flex min-h-12 items-center gap-2 rounded-full border border-violet-200 bg-violet-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-violet-600 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-200 disabled:text-slate-500"
              >
                <Flag className="h-4 w-4" />
                Finish
              </button>
              <button
                type="button"
                onClick={() => void handleResetSession()}
                disabled={isBusy}
                className="inline-flex min-h-12 items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
              >
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
              <button
                type="button"
                onClick={handleNewSession}
                disabled={isBusy}
                className="inline-flex min-h-12 items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
              >
                <ArrowRight className="h-4 w-4" />
                New Session
              </button>
            </div>

            <div className="mt-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard label="Current Level" value={String(sessionState?.current_level ?? 1)} />
              <MetricCard label="Progress" value={progress} />
              <MetricCard label="Correct This Level" value={String(sessionState?.batch_correct ?? 0)} />
              <MetricCard
                label="Interview State"
                value={sessionState?.interview_done ? "Complete" : sessionState?.phase || "Ready"}
              />
            </div>
          </Panel>

          {showSummaryView ? (
            <Panel
              title="Interview Summary"
              subtitle="The interview is complete. Review the final result or download the backend summary."
            >
              <div className="space-y-4">
                <div className="rounded-[32px] border border-emerald-200 bg-gradient-to-br from-white to-emerald-50 p-6">
                  <p className="text-sm font-semibold text-emerald-700">
                    Final level: {sessionState?.final_level ?? sessionState?.last_passed_level ?? 0}
                  </p>
                  <p className="mt-4 whitespace-pre-wrap text-sm leading-8 text-slate-700">
                    {sessionState?.assistant_text || "Interview complete."}
                  </p>
                </div>
                {summaryDownload ? (
                  <a
                    href={summaryDownload.url}
                    download={summaryDownload.fileName}
                    className="inline-flex min-h-12 items-center justify-center rounded-full border border-emerald-200 bg-emerald-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-600"
                  >
                    Download Summary JSON
                  </a>
                ) : null}
              </div>
            </Panel>
          ) : (
            <>
              <Panel
                title="Current Question"
                subtitle="The question remains the visual focus, with details available only when needed."
                actions={
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => setActivePanel("question")}
                      className="inline-flex min-h-11 items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2.5 text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                    >
                      <CircleHelp className="h-4 w-4" />
                      Details
                    </button>
                    <button
                      type="button"
                      onClick={() => setVoiceEnabled((current) => !current)}
                      disabled={!voiceAvailable}
                      className={`inline-flex min-h-11 items-center gap-2 rounded-full border px-4 py-2.5 text-sm font-medium transition ${
                        voiceAvailable
                          ? "border-sky-200 bg-sky-50 text-sky-700 hover:bg-sky-100"
                          : "cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400"
                      }`}
                    >
                      {voiceEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                      {voiceAvailable ? (voiceEnabled ? "Voice On" : "Voice Off") : "Voice Unavailable"}
                    </button>
                  </div>
                }
              >
                <div className="rounded-[32px] border border-sky-100 bg-gradient-to-br from-white to-sky-50 p-6 sm:p-8">
                  <div className="flex flex-wrap items-center gap-3 text-sm">
                    <span className="inline-flex items-center gap-2 rounded-full border border-sky-200 bg-white px-3 py-1.5 font-medium text-sky-700">
                      <AudioLines className="h-4 w-4" />
                      {streaming.active && (streaming.mode === "start" || streaming.mode === "next")
                        ? "Streaming question"
                        : liveQuestion
                          ? "Current prompt"
                          : "Ready to start"}
                    </span>
                    <span className="inline-flex rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 font-medium text-amber-700">
                      {progress}
                    </span>
                  </div>
                  <p className="mt-6 whitespace-pre-wrap text-2xl leading-10 text-slate-900 sm:text-[2rem]">
                    {liveQuestion || "Start the interview to request the first question."}
                  </p>
                </div>

                {!voiceAvailable ? (
                  <p className="mt-4 text-sm leading-6 text-slate-500">
                    Question audio playback is unavailable for this session.
                  </p>
                ) : null}

                {streaming.followupText ? (
                  <div className="mt-4 rounded-3xl border border-amber-200 bg-amber-50 p-4">
                    <p className="text-sm font-semibold text-amber-800">Follow-up in progress</p>
                    <p className="mt-2 whitespace-pre-wrap text-sm leading-7 text-amber-900">
                      {streaming.followupText}
                    </p>
                  </div>
                ) : null}
              </Panel>

              <Panel
                title="Answer"
                subtitle="Record or upload the candidate response, then submit it through the existing audio API."
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
                title="Evaluation Feedback"
                subtitle="Backend feedback stays readable here while the evaluation stream is active."
              >
                <div className="rounded-[32px] border border-emerald-100 bg-gradient-to-br from-white to-emerald-50 p-6">
                  <div className="flex flex-wrap items-center gap-3 text-sm">
                    <span className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-white px-3 py-1.5 font-medium text-emerald-700">
                      <BrainCircuit className="h-4 w-4" />
                      {streaming.active && streaming.mode === "answer"
                        ? "Streaming evaluation"
                        : "Latest feedback"}
                    </span>
                  </div>
                  <p
                    className={`mt-4 whitespace-pre-wrap text-sm leading-8 ${
                      liveEvaluation ? "text-slate-700" : "text-slate-500"
                    }`}
                  >
                    {liveEvaluation ||
                      (streaming.active && streaming.mode === "answer"
                        ? "Evaluation feedback will appear here as it streams."
                        : "Submit an answer to view evaluation feedback here.")}
                  </p>
                </div>
              </Panel>
            </>
          )}
        </main>
      </div>

      <Drawer
        open={activePanel !== null}
        title={drawerTitle}
        subtitle={drawerSubtitle}
        onClose={() => setActivePanel(null)}
      >
        {renderDrawerContent()}
      </Drawer>
    </div>
  );
}

export default App;

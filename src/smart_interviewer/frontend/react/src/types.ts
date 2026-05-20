export type ClientAction = "START" | "NEXT" | "ANSWER" | "RETRY" | "FINISH" | string;

export interface SummaryAsset {
  filename: string;
  content_type: string;
  data_base64: string;
}

export interface TurnAttempt {
  question: string;
  answer: string;
  verdict: "correct" | "incorrect" | "needs_more" | string;
  reason: string;
  is_followup: boolean;
}

export interface TurnLog {
  level: number;
  turn: number;
  item_id: string;
  context: string;
  objective: string;
  root_question: string;
  attempts: TurnAttempt[];
  correct: boolean;
  final_reason: string;
}

export interface SessionState {
  status: string;
  phase: string;
  turn: number;
  current_level: number;
  last_passed_level: number;
  batch_level: number;
  batch_index: number;
  batch_size: number;
  batch_correct: number;
  interview_done: boolean;
  final_level: number;
  current_question: string;
  root_question?: string;
  assistant_text: string;
  transcript: string;
  current_item_id?: string;
  current_context?: string;
  current_objective?: string;
  followups_used?: number;
  max_followups?: number;
  last_correct?: boolean;
  last_reason?: string;
  started_at?: string;
  finished_at?: string;
  turns_log?: TurnLog[];
  can_proceed: boolean;
  allowed_actions: ClientAction[];
  summary?: SummaryAsset | null;
  tts_enabled?: boolean;
}

export type TokenStreamEvent =
  | {
      type: "question_token" | "evaluation_token" | "followup_token";
      token: string;
    }
  | {
      type: "final_state";
      data: SessionState;
    }
  | {
      type: string;
      token?: string;
      data?: unknown;
    };

export type TranscriptRole =
  | "assistant"
  | "user"
  | "transcript"
  | "evaluation"
  | "followup"
  | "system"
  | "error";

export interface TranscriptEntry {
  id: string;
  role: TranscriptRole;
  title: string;
  content: string;
  timestamp: string;
}

export interface EventLogEntry {
  id: string;
  type: "action" | "stream" | "error" | "system";
  label: string;
  detail: string;
  timestamp: string;
}

export type StreamMode = "start" | "answer" | "next" | null;

export interface StreamingState {
  active: boolean;
  mode: StreamMode;
  questionText: string;
  evaluationText: string;
  followupText: string;
}

export interface AudioSelection {
  blob: Blob;
  fileName: string;
  mimeType: string;
  size: number;
  source: "recording" | "upload";
  previewUrl: string;
}

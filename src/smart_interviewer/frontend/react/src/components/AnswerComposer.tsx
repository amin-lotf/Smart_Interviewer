import { LoaderCircle, Mic, Square, Trash2, Upload } from "lucide-react";
import { useRef } from "react";

import type { AudioSelection } from "../types";

interface AnswerComposerProps {
  selectedAudio: AudioSelection | null;
  busy: boolean;
  canSubmit: boolean;
  recordingSupported: boolean;
  recorderStatus: "idle" | "recording" | "processing";
  recorderError: string | null;
  onStartRecording: () => Promise<void>;
  onStopRecording: () => Promise<void>;
  onFileSelected: (file: File | null) => void;
  onClearSelection: () => void;
  onSubmit: () => Promise<void>;
}

function formatBytes(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function AnswerComposer({
  selectedAudio,
  busy,
  canSubmit,
  recordingSupported,
  recorderStatus,
  recorderError,
  onStartRecording,
  onStopRecording,
  onFileSelected,
  onClearSelection,
  onSubmit,
}: AnswerComposerProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          onClick={() => void onStartRecording()}
          disabled={busy || !recordingSupported || recorderStatus === "recording"}
          className="inline-flex items-center gap-2 rounded-xl border border-cyan-400/30 bg-cyan-400/10 px-4 py-2 text-sm font-medium text-cyan-100 transition hover:bg-cyan-400/15 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/5 disabled:text-slate-500"
        >
          <Mic className="h-4 w-4" />
          Record
        </button>
        <button
          type="button"
          onClick={() => void onStopRecording()}
          disabled={busy || recorderStatus !== "recording"}
          className="inline-flex items-center gap-2 rounded-xl border border-amber-400/30 bg-amber-400/10 px-4 py-2 text-sm font-medium text-amber-100 transition hover:bg-amber-400/15 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/5 disabled:text-slate-500"
        >
          {recorderStatus === "processing" ? (
            <LoaderCircle className="h-4 w-4 animate-spin" />
          ) : (
            <Square className="h-4 w-4" />
          )}
          Stop
        </button>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={busy}
          className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/[0.08] disabled:cursor-not-allowed disabled:text-slate-500"
        >
          <Upload className="h-4 w-4" />
          Upload Audio
        </button>
        <button
          type="button"
          onClick={onClearSelection}
          disabled={busy || !selectedAudio}
          className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2 text-sm font-medium text-slate-300 transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:text-slate-500"
        >
          <Trash2 className="h-4 w-4" />
          Clear
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          hidden
          onChange={(event) => onFileSelected(event.target.files?.[0] || null)}
        />
      </div>

      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
        <span className="rounded-full border border-white/10 px-3 py-1">
          Recorder: {recordingSupported ? recorderStatus : "upload only"}
        </span>
        <span className="rounded-full border border-white/10 px-3 py-1">
          Backend expects audio uploads for answer submission
        </span>
      </div>

      {recorderError ? (
        <div className="rounded-2xl border border-rose-400/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
          {recorderError}
        </div>
      ) : null}

      {selectedAudio ? (
        <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-slate-100">{selectedAudio.fileName}</p>
              <p className="mt-1 text-sm text-slate-400">
                {selectedAudio.source} • {selectedAudio.mimeType || "audio/*"} •{" "}
                {formatBytes(selectedAudio.size)}
              </p>
            </div>
          </div>
          <audio className="mt-4 w-full" controls src={selectedAudio.previewUrl} />
        </div>
      ) : (
        <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] px-4 py-8 text-center text-sm text-slate-400">
          Capture an answer from the microphone or upload an existing recording before sending it to
          the API.
        </div>
      )}

      <div className="flex items-center justify-between gap-4">
        <p className="max-w-xl text-sm text-slate-400">
          The React client mirrors the current backend contract. It does not synthesize text-answer
          requests because the API only accepts audio on the answer endpoint today.
        </p>
        <button
          type="button"
          onClick={() => void onSubmit()}
          disabled={!canSubmit}
          className="inline-flex items-center gap-2 rounded-xl border border-emerald-400/30 bg-emerald-400/12 px-5 py-2.5 text-sm font-medium text-emerald-100 transition hover:bg-emerald-400/18 disabled:cursor-not-allowed disabled:border-white/10 disabled:bg-white/[0.05] disabled:text-slate-500"
        >
          {busy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
          Submit Answer
        </button>
      </div>
    </div>
  );
}

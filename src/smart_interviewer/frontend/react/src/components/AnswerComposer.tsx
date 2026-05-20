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
  const isRecording = recorderStatus === "recording";
  const showProcessing = recorderStatus === "processing";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          onClick={() => void onStartRecording()}
          disabled={busy || !recordingSupported || isRecording}
          className="inline-flex min-h-12 items-center gap-2 rounded-full border border-rose-200 bg-rose-50 px-5 py-3 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400"
        >
          <Mic className="h-4 w-4" />
          Record
        </button>
        <button
          type="button"
          onClick={() => void onStopRecording()}
          disabled={busy || !isRecording}
          className="inline-flex min-h-12 items-center gap-2 rounded-full border border-slate-200 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400"
        >
          {showProcessing ? (
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
          className="inline-flex min-h-12 items-center gap-2 rounded-full border border-sky-200 bg-white px-5 py-3 text-sm font-semibold text-sky-700 transition hover:bg-sky-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400"
        >
          <Upload className="h-4 w-4" />
          Upload Audio
        </button>
        <button
          type="button"
          onClick={onClearSelection}
          disabled={busy || !selectedAudio}
          className="inline-flex min-h-12 items-center gap-2 rounded-full border border-slate-200 bg-slate-50 px-5 py-3 text-sm font-semibold text-slate-600 transition hover:bg-slate-100 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-400"
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

      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-600">
        <span className="rounded-full border border-amber-100 bg-amber-50 px-3 py-1.5">
          Recorder: {recordingSupported ? recorderStatus : "upload only"}
        </span>
        <span className="rounded-full border border-slate-200 bg-white px-3 py-1.5">
          Submit a recorded or uploaded response
        </span>
      </div>

      {recorderError ? (
        <div className="rounded-3xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {recorderError}
        </div>
      ) : null}

      {selectedAudio ? (
        <div className="rounded-3xl border border-slate-200 bg-slate-50 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-slate-900">{selectedAudio.fileName}</p>
              <p className="mt-1 text-sm text-slate-600">
                {selectedAudio.source} • {selectedAudio.mimeType || "audio/*"} •{" "}
                {formatBytes(selectedAudio.size)}
              </p>
            </div>
          </div>
          <audio className="mt-4 w-full" controls src={selectedAudio.previewUrl} />
        </div>
      ) : (
        <div className="rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-4 py-8 text-center text-sm leading-6 text-slate-600">
          Capture an answer from the microphone or upload an existing recording before submitting it.
        </div>
      )}

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <p className="max-w-xl text-sm leading-6 text-slate-600">
          Keep the candidate response short and clear. The current API flow still submits audio only.
        </p>
        <button
          type="button"
          onClick={() => void onSubmit()}
          disabled={!canSubmit}
          className="inline-flex min-h-12 items-center justify-center gap-2 rounded-full border border-emerald-200 bg-emerald-500 px-6 py-3 text-sm font-semibold text-white transition hover:bg-emerald-600 disabled:cursor-not-allowed disabled:border-slate-200 disabled:bg-slate-200 disabled:text-slate-500"
        >
          {busy ? <LoaderCircle className="h-4 w-4 animate-spin" /> : null}
          Submit Answer
        </button>
      </div>
    </div>
  );
}

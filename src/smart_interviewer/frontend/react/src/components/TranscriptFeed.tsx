import {
  AlertCircle,
  BrainCircuit,
  FileAudio,
  MessageCircleReply,
  MessageSquareText,
  NotebookText,
  Sparkles,
} from "lucide-react";

import type { TranscriptEntry, TranscriptRole } from "../types";

interface TranscriptFeedProps {
  entries: TranscriptEntry[];
  emptyTitle: string;
  emptyBody: string;
}

function roleStyle(role: TranscriptRole) {
  switch (role) {
    case "user":
      return {
        icon: FileAudio,
        accent: "from-rose-50 to-white text-slate-900 ring-rose-100",
        badge: "bg-rose-100 text-rose-700",
      };
    case "transcript":
      return {
        icon: NotebookText,
        accent: "from-slate-50 to-white text-slate-900 ring-slate-200",
        badge: "bg-slate-200 text-slate-700",
      };
    case "evaluation":
      return {
        icon: BrainCircuit,
        accent: "from-emerald-50 to-white text-slate-900 ring-emerald-100",
        badge: "bg-emerald-100 text-emerald-700",
      };
    case "followup":
      return {
        icon: MessageCircleReply,
        accent: "from-amber-50 to-white text-slate-900 ring-amber-100",
        badge: "bg-amber-100 text-amber-700",
      };
    case "system":
      return {
        icon: Sparkles,
        accent: "from-violet-50 to-white text-slate-900 ring-violet-100",
        badge: "bg-violet-100 text-violet-700",
      };
    case "error":
      return {
        icon: AlertCircle,
        accent: "from-rose-50 to-white text-slate-900 ring-rose-100",
        badge: "bg-rose-100 text-rose-700",
      };
    case "assistant":
    default:
      return {
        icon: MessageSquareText,
        accent: "from-sky-50 to-white text-slate-900 ring-sky-100",
        badge: "bg-sky-100 text-sky-700",
      };
  }
}

function formatTimestamp(timestamp: string): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(timestamp));
}

export function TranscriptFeed({
  entries,
  emptyTitle,
  emptyBody,
}: TranscriptFeedProps) {
  if (!entries.length) {
    return (
      <div className="flex min-h-64 flex-col items-center justify-center rounded-3xl border border-dashed border-slate-200 bg-slate-50 px-6 py-10 text-center">
        <p className="text-base font-medium text-slate-800">{emptyTitle}</p>
        <p className="mt-2 max-w-md text-sm leading-6 text-slate-600">{emptyBody}</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {entries.map((entry) => {
        const style = roleStyle(entry.role);
        const Icon = style.icon;

        return (
          <article
            key={entry.id}
            className={`rounded-3xl border bg-gradient-to-br ${style.accent} p-4 ring-1`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-white shadow-sm">
                  <Icon className="h-4 w-4" />
                </span>
                <div>
                  <p className="text-sm font-semibold">{entry.title}</p>
                  <span
                    className={`mt-1 inline-flex rounded-full px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.14em] ${style.badge}`}
                  >
                    {entry.role}
                  </span>
                </div>
              </div>
              <time className="text-xs text-slate-500">{formatTimestamp(entry.timestamp)}</time>
            </div>
            <p className="mt-4 whitespace-pre-wrap text-sm leading-7 text-slate-700">{entry.content}</p>
          </article>
        );
      })}
    </div>
  );
}

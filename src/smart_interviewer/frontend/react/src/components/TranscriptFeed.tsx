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
        accent: "from-cyan-400/25 to-cyan-500/10 text-cyan-200 ring-cyan-400/20",
        badge: "bg-cyan-400/12 text-cyan-200",
      };
    case "transcript":
      return {
        icon: NotebookText,
        accent: "from-slate-400/20 to-slate-500/10 text-slate-200 ring-white/10",
        badge: "bg-slate-400/10 text-slate-200",
      };
    case "evaluation":
      return {
        icon: BrainCircuit,
        accent: "from-emerald-400/25 to-emerald-500/10 text-emerald-100 ring-emerald-400/20",
        badge: "bg-emerald-400/12 text-emerald-200",
      };
    case "followup":
      return {
        icon: MessageCircleReply,
        accent: "from-amber-400/25 to-amber-500/10 text-amber-100 ring-amber-400/20",
        badge: "bg-amber-400/12 text-amber-200",
      };
    case "system":
      return {
        icon: Sparkles,
        accent: "from-violet-400/25 to-violet-500/10 text-violet-100 ring-violet-400/20",
        badge: "bg-violet-400/12 text-violet-200",
      };
    case "error":
      return {
        icon: AlertCircle,
        accent: "from-rose-400/25 to-rose-500/10 text-rose-100 ring-rose-400/20",
        badge: "bg-rose-400/12 text-rose-200",
      };
    case "assistant":
    default:
      return {
        icon: MessageSquareText,
        accent: "from-indigo-400/25 to-indigo-500/10 text-indigo-100 ring-indigo-400/20",
        badge: "bg-indigo-400/12 text-indigo-200",
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
      <div className="flex min-h-64 flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-white/[0.02] px-6 py-10 text-center">
        <p className="text-base font-medium text-slate-200">{emptyTitle}</p>
        <p className="mt-2 max-w-md text-sm text-slate-400">{emptyBody}</p>
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
            className={`rounded-2xl border border-white/10 bg-gradient-to-br ${style.accent} p-4 ring-1`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-3">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-950/40">
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
              <time className="text-xs text-slate-400">{formatTimestamp(entry.timestamp)}</time>
            </div>
            <p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-slate-100">{entry.content}</p>
          </article>
        );
      })}
    </div>
  );
}

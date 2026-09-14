"use client";

import type { Cut, CutStats } from "@/lib/curation";
import { Crosshair, Trash2 } from "lucide-react";
import { useState } from "react";

/* Aside panel listing a clip's cut ranges: each editable (timestamps + note,
   committed on blur/Enter), removable, and seekable. Timestamps are frame
   ids ("HH:MM:SS.f"); the parent normalises + persists. */

const TS_RE = /^\d{2}:\d{2}:\d{2}\.\d$/;

function TsInput({
  value,
  label,
  onCommit,
}: {
  value: string;
  label: string;
  onCommit(v: string): void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? value;
  const bad = draft != null && !TS_RE.test(draft);
  return (
    <input
      type="text"
      aria-label={label}
      value={shown}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={() => {
        if (draft != null && TS_RE.test(draft) && draft !== value) onCommit(draft);
        setDraft(null);
      }}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        if (e.key === "Escape") setDraft(null);
      }}
      spellCheck={false}
      className={`w-[5.6rem] rounded-sm border bg-surface-1 px-1 py-0.5 font-mono text-[11px] tabular-nums focus:outline-none ${bad ? "border-status-serious" : "border-border focus:border-seeblau-100"}`}
    />
  );
}

export function CutsPanel({
  cuts,
  stats,
  saving,
  error,
  onChange,
  onSeekTs,
}: {
  cuts: Cut[];
  stats: CutStats;
  saving: boolean;
  error: string | null;
  onChange(next: Cut[]): void;
  onSeekTs(ts: string): void;
}) {
  const update = (i: number, patch: Partial<Cut>) =>
    onChange(cuts.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const fmt = (s: number) => (s >= 600 ? `${Math.round(s / 60)} min` : `${(s / 60).toFixed(1)} min`);
  return (
    <div className="space-y-2 text-xs">
      <div className="font-mono text-[11px] tabular-nums text-muted" data-cut-stats>
        {stats.kept} kept · {stats.cut} cut · {fmt(stats.keptSeconds)} kept / {fmt(stats.cutSeconds)} cut
        {saving && <span className="ml-2 text-subtle">saving…</span>}
      </div>
      {error && <div className="font-mono text-[11px] text-status-serious">{error}</div>}
      {cuts.length === 0 && (
        <p className="leading-relaxed text-subtle">
          No cuts. Mark <span className="font-mono">in</span> and <span className="font-mono">out</span> on the
          transport (keys <span className="font-mono">i</span>/<span className="font-mono">o</span>) to trim a range.
          Cut frames stay on disk and keep their ids; every consumer skips them.
        </p>
      )}
      <ul className="space-y-1.5">
        {cuts.map((c, i) => (
          <li key={i} data-cut-row={i} className="rounded-sm border border-border bg-surface-2 p-1.5">
            <div className="flex items-center gap-1">
              <span className="w-4 font-mono text-[10px] text-subtle">{i + 1}</span>
              <TsInput value={c.start_ts} label={`Cut ${i + 1} start`} onCommit={(v) => update(i, { start_ts: v })} />
              <span className="text-subtle">–</span>
              <TsInput value={c.end_ts} label={`Cut ${i + 1} end`} onCommit={(v) => update(i, { end_ts: v })} />
              <button
                type="button"
                onClick={() => onSeekTs(c.start_ts)}
                aria-label={`Go to cut ${i + 1}`}
                title="Go to the start of this cut"
                className="ml-auto rounded-sm p-1 text-subtle hover:bg-surface-3 hover:text-foreground"
              >
                <Crosshair size={12} aria-hidden />
              </button>
              <button
                type="button"
                onClick={() => onChange(cuts.filter((_, j) => j !== i))}
                aria-label={`Remove cut ${i + 1}`}
                title="Remove this cut (frames come back)"
                className="rounded-sm p-1 text-subtle hover:bg-surface-3 hover:text-status-serious"
              >
                <Trash2 size={12} aria-hidden />
              </button>
            </div>
            <input
              type="text"
              aria-label={`Cut ${i + 1} note`}
              placeholder="note (why)"
              defaultValue={c.note ?? ""}
              onBlur={(e) => {
                const v = e.target.value.trim();
                if (v !== (c.note ?? "")) update(i, { note: v || undefined });
              }}
              onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
              className="mt-1 w-full rounded-sm border border-transparent bg-transparent px-1 py-0.5 text-[11px] text-muted placeholder:text-subtle focus:border-border focus:outline-none"
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

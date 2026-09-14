"use client";

import { CopyFetchLink } from "@/components/catalogue/CopyFetchLink";
import { StreamChips } from "@/components/catalogue/StreamChips";
import {
  applyCuration,
  cutRangeSeconds,
  daysRemaining,
  getCurationBackend,
  isDeleted,
  type CurationRecord,
} from "@/lib/curation";
import { mintAndCopyFetchLink, mintPending, type MintResult } from "@/lib/exportLink";
import { formatActivityRange, formatScene } from "@/lib/format";
import { getProvider, listCuratedClips } from "@/lib/data";
import { clipKeyFromId, type ClipSummary } from "@/lib/types";
import { Check, RotateCcw, Scissors, Trash2 } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";

/* eslint-disable @next/next/no-img-element -- bundle/storage assets, not optimizer targets */

const CAPTURE_FPS = 3;

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-surface-1 px-4 py-3">
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
        {label}
      </div>
      <div className="mt-1 font-mono text-xl tabular-nums tracking-tight">
        {value}
      </div>
    </div>
  );
}

function fmtMin(seconds: number): string {
  const m = seconds / 60;
  return m >= 10 ? `${Math.round(m)} min` : `${m.toFixed(1)} min`;
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  const dd = String(d.getDate()).padStart(2, "0");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  return `${dd}-${mm}-${d.getFullYear()}`;
}

export default function ClipsPage() {
  const [clips, setClips] = useState<ClipSummary[] | null>(null);
  const [records, setRecords] = useState<Map<string, CurationRecord>>(new Map());
  const [view, setView] = useState<"active" | "deleted">("active");
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [minting, setMinting] = useState(false);
  const [toast, setToast] = useState<MintResult | null>(null);
  const [confirmKey, setConfirmKey] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<string | null>(null);
  // wall clock for "days until prune", sampled when the catalogue loads
  // (a render must not call Date.now())
  const [now, setNow] = useState(0);
  const toastTimer = useRef<number | null>(null);

  function showToast(r: MintResult) {
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    setToast(r);
    // a pending toast stays until the outcome replaces it
    if (r.kind !== "pending") {
      toastTimer.current = window.setTimeout(() => setToast(null), 6000);
    }
  }

  function toggleSelected(key: string) {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  async function mintSelected() {
    if (minting || selected.size === 0) return;
    setMinting(true);
    showToast(mintPending(selected.size));
    showToast(await mintAndCopyFetchLink([...selected].sort()));
    setMinting(false);
  }

  async function setDeleted(key: string, deleted: boolean) {
    if (busyKey) return;
    setBusyKey(key);
    setConfirmKey(null);
    try {
      const rec = await getCurationBackend().setDeleted(key, deleted);
      setRecords((m) => new Map(m).set(key, rec));
      setSelected((s) => {
        const next = new Set(s);
        next.delete(key);
        return next;
      });
      showToast({
        kind: "ok",
        message: deleted
          ? `Deleted ${key}. Restore any time from the Deleted view; the bundle is kept 30 days.`
          : `Restored ${key}.`,
      });
    } catch (e) {
      showToast({ kind: "error", message: `${deleted ? "Delete" : "Restore"} failed: ${String(e)}` });
    } finally {
      setBusyKey(null);
    }
  }

  useEffect(() => {
    listCuratedClips()
      .then(({ active, deleted, records }) => {
        // reverse chronological — ISO timestamps sort lexicographically
        setClips(
          [...active, ...deleted.map((d) => d.clip)].sort((a, b) =>
            b.triplet_ts.localeCompare(a.triplet_ts),
          ),
        );
        setRecords(records);
        setNow(Date.now());
      })
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const { active, deleted } = useMemo(
    () => (clips ? applyCuration(clips, records) : { active: [], deleted: [] }),
    [clips, records],
  );
  const shown = view === "active" ? active : deleted.map((d) => d.clip);

  if (error) {
    return (
      <div className="p-8 font-mono text-sm text-status-serious">
        Failed to load the clip catalogue: {error}
      </div>
    );
  }

  const totalFrames = active.reduce((acc, c) => acc + c.n_frames, 0);
  const totalLabelled = active.reduce((acc, c) => acc + (c.n_labelled ?? 0), 0);
  const provider = getProvider();

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
        Institution One · the lake
      </div>
      <div className="mt-1 flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">
          Mission clips
        </h1>
        <span className="flex items-center gap-2">
          <Link
            href="/offline"
            className="rounded-sm border border-border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-muted hover:border-border-strong hover:text-foreground"
          >
            offline packs
          </Link>
          <Link
            href="/upload"
            className="rounded-sm border border-border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-muted hover:border-border-strong hover:text-foreground"
          >
            upload session
          </Link>
        </span>
      </div>
      <p className="mt-2 max-w-2xl text-sm text-muted">
        Fused fisheye + thermal + mmWave radar captures from the Vessel A
        obstacle-detection payload, with segmentation, typed detections and
        learned-scorer sectors where available.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-sm border border-border bg-border sm:grid-cols-4">
        <Kpi label="clips" value={clips ? String(active.length) : "…"} />
        <Kpi label="frames" value={clips ? String(totalFrames) : "…"} />
        <Kpi label="labelled frames" value={clips ? String(totalLabelled) : "…"} />
        <Kpi label="source" value={provider.mode} />
      </div>

      <div className="mt-6 flex items-center gap-2" role="tablist" aria-label="Catalogue view">
        {(
          [
            ["active", `Active (${active.length})`],
            ["deleted", `Deleted (${deleted.length})`],
          ] as const
        ).map(([v, label]) => (
          <button
            key={v}
            type="button"
            role="tab"
            aria-selected={view === v}
            onClick={() => setView(v)}
            className={`rounded-sm border px-3 py-1 font-mono text-[11px] uppercase tracking-wider ${
              view === v
                ? "border-seeblau-100 bg-seeblau-100/10 text-seeblau-deep"
                : "border-border text-muted hover:text-foreground"
            }`}
          >
            {label}
          </button>
        ))}
        {view === "deleted" && (
          <span className="ml-2 text-[11px] text-subtle">
            Deleted sets leave the catalogue, planner, data links and coverage map.
            Bundles are kept 30 days (restore is instant), then{" "}
            <code className="bg-surface-3 px-1">pnpm curation:prune</code> may
            remove them. Raw captures are never touched.
          </span>
        )}
      </div>

      <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-2">
        {clips === null && (
          <div className="font-mono text-sm text-subtle">loading clips…</div>
        )}
        {clips !== null && shown.length === 0 && (
          <div className="font-mono text-sm text-subtle">
            {view === "deleted" ? "no deleted sets" : "no clips"}
          </div>
        )}
        {shown.map((clip) => {
          const key = clipKeyFromId(clip.clip_id);
          const rec = records.get(key);
          const del = rec && isDeleted(rec);
          const cutS = rec && !del ? cutRangeSeconds(rec.cuts) : 0;
          const totalS = clip.n_frames / CAPTURE_FPS;
          return (
            <Link
              key={clip.clip_id}
              href={`/clips/${key}`}
              data-clip-key={key}
              className={`group overflow-hidden rounded-sm border border-border bg-surface-1 hover:border-border-strong hover:bg-surface-2 ${del ? "opacity-80" : ""}`}
            >
              {clip.thumb_ts && (
                <div className="aspect-[4/3] w-full overflow-hidden border-b border-border bg-surface-3">
                  <img
                    src={provider.frameUrl(key, "frames", clip.thumb_ts)}
                    alt={`First frame of ${clip.title}`}
                    className={`h-full w-full object-cover ${del ? "grayscale" : ""}`}
                    loading="lazy"
                  />
                </div>
              )}
              <div className="p-4">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-sm font-semibold tracking-tight">
                    {clip.title}
                  </h2>
                  <span className="flex shrink-0 items-center gap-2">
                    <span className="font-mono text-xs tabular-nums text-subtle">
                      {clip.n_frames} frames
                    </span>
                    {!del && <CopyFetchLink clipKey={key} onResult={showToast} />}
                    {!del && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          if (confirmKey === key) void setDeleted(key, true);
                          else setConfirmKey(key);
                        }}
                        onBlur={() => confirmKey === key && setConfirmKey(null)}
                        disabled={busyKey === key}
                        title={confirmKey === key ? "Click again to delete this set (restorable for 30 days)" : "Delete this set"}
                        aria-label={confirmKey === key ? `Confirm delete ${key}` : `Delete ${key}`}
                        className={`flex shrink-0 items-center gap-1 rounded-sm border p-1 font-mono text-[10px] uppercase ${
                          confirmKey === key
                            ? "border-status-serious bg-status-serious/10 px-1.5 text-status-serious"
                            : "border-border text-subtle hover:border-status-serious hover:text-status-serious"
                        }`}
                      >
                        <Trash2 size={13} aria-hidden />
                        {confirmKey === key && "confirm"}
                      </button>
                    )}
                    {!del && (
                      <button
                        type="button"
                        role="checkbox"
                        aria-checked={selected.has(key)}
                        onClick={(e) => {
                          // inside the card's <Link>: keep the click ours.
                          // A button, not a native checkbox: preventDefault on
                          // a controlled checkbox inside an anchor leaves its
                          // tick one click behind.
                          e.preventDefault();
                          e.stopPropagation();
                          toggleSelected(key);
                        }}
                        title="Select for a combined data link"
                        aria-label={`Select ${clip.title} for a combined data link`}
                        className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-[3px] border ${
                          selected.has(key)
                            ? "border-seeblau-100 bg-seeblau-100 text-inverse"
                            : "border-border-strong bg-surface-1 text-transparent hover:border-seeblau-100"
                        }`}
                      >
                        <Check size={11} strokeWidth={3} aria-hidden />
                      </button>
                    )}
                  </span>
                </div>
                <div className="mt-0.5 flex flex-wrap items-baseline gap-2 font-mono text-[11px] tabular-nums text-subtle">
                  <span className="text-muted">
                    {formatActivityRange(clip.triplet_ts, clip.end_ts)}
                  </span>
                  <span>· {formatScene(clip.scene)}</span>
                  {(clip.chunks?.length ?? 1) > 1 && (
                    <span className="rounded-sm border border-border px-1">
                      {clip.chunks!.length} chunks
                    </span>
                  )}
                  {!del && rec && rec.cuts.length > 0 && (
                    <span
                      className="flex items-center gap-1 rounded-sm border border-status-warn/60 px-1 text-status-warn"
                      title={`${rec.cuts.length} cut${rec.cuts.length > 1 ? "s" : ""}: ${fmtMin(Math.max(totalS - cutS, 0))} kept, ${fmtMin(cutS)} cut`}
                    >
                      <Scissors size={10} aria-hidden />
                      trimmed · {fmtMin(Math.max(totalS - cutS, 0))} kept / {fmtMin(cutS)} cut
                    </span>
                  )}
                </div>
                {del && rec?.deleted_at && (
                  <div className="mt-2 flex flex-wrap items-center gap-2 rounded-sm border border-status-serious/40 bg-status-serious/5 px-2 py-1.5 font-mono text-[11px] text-status-serious">
                    <span>
                      Deleted on {fmtDate(rec.deleted_at)} ·{" "}
                      {rec.purged_at
                        ? "bundle purged"
                        : `${daysRemaining(rec.deleted_at, now)} days until prune`}
                    </span>
                    {!rec.purged_at && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.preventDefault();
                          e.stopPropagation();
                          void setDeleted(key, false);
                        }}
                        disabled={busyKey === key}
                        aria-label={`Restore ${key}`}
                        className="ml-auto flex items-center gap-1 rounded-sm border border-status-good px-2 py-0.5 uppercase tracking-wider text-status-good hover:bg-status-good/10 disabled:opacity-50"
                      >
                        <RotateCcw size={11} aria-hidden /> restore
                      </button>
                    )}
                  </div>
                )}
                <p className="mt-2 text-xs leading-relaxed text-muted">
                  {clip.description}
                </p>
                <div className="mt-3">
                  <StreamChips streams={clip.streams} />
                </div>
              </div>
            </Link>
          );
        })}
      </div>

      {(selected.size > 0 || toast) && (
        <div className="pointer-events-none fixed inset-x-0 bottom-6 z-40 flex flex-col items-center gap-2 px-4">
          {toast && (
            <div
              role="status"
              className={`pointer-events-auto max-w-xl rounded-sm border bg-surface-1 px-4 py-2 font-mono text-xs shadow-lg ${
                toast.kind === "ok"
                  ? "border-status-good text-foreground"
                  : toast.kind === "pending"
                    ? "border-status-warn text-status-warn"
                    : "border-status-serious text-status-serious"
              }`}
            >
              {toast.message}
            </div>
          )}
          {selected.size > 0 && (
            <div className="pointer-events-auto flex items-center gap-3 rounded-sm border border-border-strong bg-surface-2 px-4 py-2 shadow-lg">
              <span className="font-mono text-xs tabular-nums text-muted">
                {selected.size} clip{selected.size > 1 ? "s" : ""} selected
              </span>
              <button
                type="button"
                onClick={() => void mintSelected()}
                disabled={minting}
                className="rounded-sm bg-seeblau-100 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-inverse hover:bg-seeblau-deep disabled:opacity-50"
              >
                {minting ? "minting…" : "copy data link"}
              </button>
              <button
                type="button"
                onClick={() => setSelected(new Set())}
                className="font-mono text-[11px] uppercase tracking-wider text-subtle hover:text-foreground"
              >
                clear
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

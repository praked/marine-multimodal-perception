"use client";

import { useViewerStore } from "@/lib/stores/viewer";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  Pause,
  Play,
  Scissors,
} from "lucide-react";
import { useRef } from "react";

/** Curation controls the transport renders when a clip can be trimmed. */
export interface CutControls {
  /** cut regions as timeline fractions (lib/curation cutRuns) */
  runs: { start: number; width: number }[];
  /** per cut: first/last frame index, for the draggable edge handles */
  edges: { cut: number; startIdx: number; endIdx: number }[];
  inIndex: number | null;
  isCurrentCut: boolean;
  nCuts: number;
  showCutFrames: boolean;
  onMarkIn(): void;
  onMarkOut(): void;
  onClearIn(): void;
  onToggleShowCutFrames(): void;
  /** live edge drag: cut i, which edge, new frame index (not yet saved) */
  onDragEdge(cut: number, edge: "start" | "end", frameIndex: number): void;
  /** pointer released: persist the dragged cut */
  onDragEnd(): void;
}

interface Props {
  /** playback is holding the current frame while the next one decodes */
  buffering?: boolean;
  index: number;
  total: number;
  ts: string;
  playing: boolean;
  onSeek(index: number): void;
  onTogglePlay(): void;
  cuts?: CutControls;
}

const SPEEDS = [0.5, 1, 2, 4];
const JUMP = 5;

function IconButton({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick(): void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="flex h-8 w-8 items-center justify-center rounded-sm text-muted hover:bg-surface-3 hover:text-foreground"
    >
      {children}
    </button>
  );
}

/** One draggable cut edge on the scrub bar. Owns its pointer capture so
    the parent never reads a ref during render. */
function CutHandle({
  cut,
  edge,
  left,
  frameAt,
  onDragEdge,
  onDragEnd,
}: {
  cut: number;
  edge: "start" | "end";
  left: number; // percent
  frameAt(clientX: number): number;
  onDragEdge(cut: number, edge: "start" | "end", frameIndex: number): void;
  onDragEnd(): void;
}) {
  const dragging = useRef(false);
  return (
    <button
      type="button"
      data-cut-handle={`${cut}-${edge}`}
      aria-label={`Drag ${edge} of cut ${cut + 1}`}
      title={`drag to move the ${edge} of cut ${cut + 1}`}
      onPointerDown={(e) => {
        e.preventDefault();
        e.stopPropagation();
        dragging.current = true;
        (e.target as Element).setPointerCapture?.(e.pointerId);
      }}
      onPointerMove={(e) => {
        if (dragging.current) onDragEdge(cut, edge, frameAt(e.clientX));
      }}
      onPointerUp={() => {
        if (!dragging.current) return;
        dragging.current = false;
        onDragEnd();
      }}
      className="absolute top-1/2 z-10 h-4 w-2 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize rounded-[2px] border border-status-warn bg-surface-1 hover:bg-status-warn"
      style={{ left: `${left}%` }}
    />
  );
}

/** Transport bar: play/pause at footage rate (~3 fps × speed), single-step,
    5-frame jumps, scrub bar, timestamp readout. Keyboard: space, ←/→,
    shift+←/→, i/o in-out points (bound in ClipViewer). Cut regions render
    greyed on the scrub bar with draggable edge handles. */
export function Transport({
  index,
  total,
  ts,
  playing,
  onSeek,
  onTogglePlay,
  cuts,
  buffering = false,
}: Props) {
  const speed = useViewerStore((s) => s.speed);
  const setSpeed = useViewerStore((s) => s.setSpeed);
  const clamp = (i: number) => Math.min(Math.max(i, 0), total - 1);
  const barRef = useRef<HTMLDivElement>(null);

  function frameAt(clientX: number): number {
    const el = barRef.current;
    if (!el || total <= 1) return 0;
    const r = el.getBoundingClientRect();
    const f = Math.min(Math.max((clientX - r.left) / r.width, 0), 1);
    return Math.round(f * (total - 1));
  }

  const denom = Math.max(total - 1, 1);

  return (
    <div className="flex items-center gap-2 border-t border-border bg-surface-1 px-3 py-2">
      <IconButton label={`Back ${JUMP} frames (shift+←)`} onClick={() => onSeek(clamp(index - JUMP))}>
        <ChevronsLeft size={16} aria-hidden />
      </IconButton>
      <IconButton label="Previous frame (←)" onClick={() => onSeek(clamp(index - 1))}>
        <ChevronLeft size={16} aria-hidden />
      </IconButton>
      <button
        type="button"
        aria-label={playing ? "Pause (space)" : "Play (space)"}
        title={playing ? "Pause (space)" : "Play (space)"}
        onClick={onTogglePlay}
        className="flex h-8 w-8 items-center justify-center rounded-sm bg-seeblau-100 text-inverse hover:bg-seeblau-deep"
      >
        {playing ? (
          <Pause size={16} aria-hidden />
        ) : (
          <Play size={16} aria-hidden />
        )}
      </button>
      <IconButton label="Next frame (→)" onClick={() => onSeek(clamp(index + 1))}>
        <ChevronRight size={16} aria-hidden />
      </IconButton>
      <IconButton label={`Forward ${JUMP} frames (shift+→)`} onClick={() => onSeek(clamp(index + JUMP))}>
        <ChevronsRight size={16} aria-hidden />
      </IconButton>

      <div ref={barRef} className="relative min-w-0 flex-1 py-1">
        {cuts && cuts.runs.length > 0 && (
          <div className="pointer-events-none absolute inset-x-0 top-1/2 h-3 -translate-y-1/2" aria-hidden>
            {cuts.runs.map((r, i) => (
              <div
                key={i}
                data-cut-region
                className="absolute h-full rounded-sm"
                style={{
                  left: `${r.start * 100}%`,
                  width: `${Math.max(r.width * 100, 0.4)}%`,
                  background: "repeating-linear-gradient(135deg, #8a8f94 0 2px, #c5c9cc 2px 4px)",
                  opacity: 0.85,
                }}
              />
            ))}
          </div>
        )}
        {cuts && cuts.inIndex != null && (
          <div
            className="pointer-events-none absolute top-0 h-full w-[2px] bg-status-warn"
            style={{ left: `${(cuts.inIndex / denom) * 100}%` }}
            aria-hidden
            data-in-marker
          />
        )}
        <input
          type="range"
          aria-label="Scrub through frames"
          min={0}
          max={Math.max(total - 1, 0)}
          step={1}
          value={index}
          onChange={(e) => onSeek(Number(e.target.value))}
          className="relative block w-full accent-(--seeblau-100)"
        />
        {cuts && cuts.edges.map(({ cut, startIdx, endIdx }) =>
          (["start", "end"] as const).map((edge) => (
            <CutHandle
              key={`${cut}-${edge}`}
              cut={cut}
              edge={edge}
              left={((edge === "start" ? startIdx : endIdx) / denom) * 100}
              frameAt={frameAt}
              onDragEdge={cuts.onDragEdge}
              onDragEnd={cuts.onDragEnd}
            />
          )),
        )}
      </div>

      {playing && buffering && (
        <span
          className="h-2 w-2 animate-pulse rounded-full bg-status-warn"
          title="Buffering: holding this frame while the next one loads"
          aria-label="buffering"
        />
      )}
      <span className="font-mono text-xs tabular-nums text-muted">
        {index + 1}/{total}
      </span>
      <span className="w-24 text-right font-mono text-xs tabular-nums text-foreground">
        {ts}
      </span>
      {cuts && (
        <span className="flex items-center gap-1 border-l border-border pl-2">
          <button
            type="button"
            onClick={cuts.onMarkIn}
            aria-pressed={cuts.inIndex === index}
            title="Mark the IN point of a cut here (i)"
            className={`rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${cuts.inIndex != null ? "border-status-warn text-status-warn" : "border-border text-muted hover:text-foreground"}`}
          >
            in{cuts.inIndex != null ? ` ${cuts.inIndex + 1}` : ""}
          </button>
          <button
            type="button"
            onClick={cuts.onMarkOut}
            title={cuts.inIndex != null ? "Mark the OUT point here and create the cut (o)" : "Cut this single frame (o); set an IN point first for a range"}
            className="flex items-center gap-1 rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted hover:border-status-warn hover:text-status-warn"
          >
            <Scissors size={11} aria-hidden /> out
          </button>
          {cuts.inIndex != null && (
            <button
              type="button"
              onClick={cuts.onClearIn}
              title="Clear the IN point"
              className="font-mono text-[10px] uppercase text-subtle hover:text-foreground"
            >
              ×
            </button>
          )}
          {cuts.nCuts > 0 && (
            <button
              type="button"
              role="switch"
              aria-checked={cuts.showCutFrames}
              onClick={cuts.onToggleShowCutFrames}
              title="Show cut frames while playing and stepping (default: skipped)"
              className={`ml-1 rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${cuts.showCutFrames ? "border-seeblau-100 bg-seeblau-100/10 text-seeblau-deep" : "border-border text-muted hover:text-foreground"}`}
            >
              show cut frames
            </button>
          )}
          {cuts.isCurrentCut && (
            <span className="rounded-sm bg-surface-3 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-muted" data-current-cut>
              cut
            </span>
          )}
        </span>
      )}
      <label className="flex items-center gap-1 font-mono text-[10px] uppercase tracking-wider text-subtle">
        speed
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="rounded-sm border border-border bg-surface-1 px-1 py-0.5 font-mono text-xs text-foreground"
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

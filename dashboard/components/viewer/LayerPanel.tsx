"use client";

import {
  useViewerStore,
  type LayerState,
  type ScoreSource,
} from "@/lib/stores/viewer";
import type { Streams } from "@/lib/types";
import { Eye, EyeOff } from "lucide-react";

interface LayerDef {
  key: keyof LayerState;
  label: string;
  needs?: (streams: Streams) => boolean;
}

const LAYERS: LayerDef[] = [
  { key: "seg", label: "Segmentation tint", needs: (s) => s.seg },
  { key: "waterEdge", label: "Water edge", needs: (s) => s.seg },
  { key: "typedBoxes", label: "Detection boxes" },
  { key: "instances", label: "Instance masks" },
  { key: "labelBoxes", label: "Label boxes" },
  { key: "clutterFilter", label: "Radar clutter filter (y ≥ 0.5 m)", needs: (s) => s.radar },
  { key: "radarOverlay", label: "Radar → fisheye projection", needs: (s) => s.radar },
  { key: "confirmedMarkers", label: "Radar-confirmed markers", needs: (s) => s.sectors },
  { key: "targetTracks", label: "Target tracks", needs: (s) => s.sectors },
];

interface Props {
  streams: Streams;
  hasScorer: boolean;
}

export function LayerPanel({ streams, hasScorer }: Props) {
  const layers = useViewerStore((s) => s.layers);
  const setLayer = useViewerStore((s) => s.setLayer);
  const threshold = useViewerStore((s) => s.threshold);
  const setThreshold = useViewerStore((s) => s.setThreshold);
  const scoreSource = useViewerStore((s) => s.scoreSource);
  const setScoreSource = useViewerStore((s) => s.setScoreSource);
  const sectorWidthDeg = useViewerStore((s) => s.sectorWidthDeg);
  const setSectorWidthDeg = useViewerStore((s) => s.setSectorWidthDeg);

  return (
    <div className="space-y-3">
      <ul className="space-y-0.5">
        {LAYERS.map(({ key, label, needs }) => {
          const available = needs ? needs(streams) : true;
          const on = layers[key] && available;
          return (
            <li key={key}>
              <button
                type="button"
                disabled={!available}
                onClick={() => setLayer(key, !layers[key])}
                aria-pressed={on}
                className="flex w-full items-center gap-2 rounded-sm px-1.5 py-1 text-left text-xs hover:bg-surface-3 disabled:cursor-not-allowed disabled:opacity-40"
              >
                {on ? (
                  <Eye size={14} className="text-accent-strong" aria-hidden />
                ) : (
                  <EyeOff size={14} className="text-subtle" aria-hidden />
                )}
                <span className={on ? "" : "text-muted"}>{label}</span>
                {!available && (
                  <span className="ml-auto font-mono text-[10px] text-subtle">
                    n/a
                  </span>
                )}
              </button>
            </li>
          );
        })}
      </ul>

      <div>
        <label
          htmlFor="threshold"
          className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-subtle"
        >
          <span>hit threshold</span>
          <span className="tabular-nums text-foreground">
            {threshold.toFixed(2)}
          </span>
        </label>
        <input
          id="threshold"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
          className="mt-1 w-full accent-(--seeblau-100)"
        />
      </div>

      <div>
        <label
          htmlFor="sector-width"
          className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-subtle"
        >
          <span>sector width</span>
          <span className="tabular-nums text-foreground">{sectorWidthDeg}°</span>
        </label>
        <input
          id="sector-width"
          type="range"
          min={5}
          max={45}
          step={5}
          value={sectorWidthDeg}
          onChange={(e) => setSectorWidthDeg(Number(e.target.value))}
          className="mt-1 w-full accent-(--seeblau-100)"
        />
        <p className="mt-0.5 text-[10px] leading-snug text-subtle">
          resamples the record&apos;s native bins (15° current, 10° legacy)
          conservatively: max score, min range. Below the native width,
          sectors subdivide the same bin.
        </p>
      </div>

      <div>
        <span className="font-mono text-[10px] uppercase tracking-wider text-subtle">
          sector value
        </span>
        <div className="mt-1 flex gap-1" role="radiogroup" aria-label="Sector value source">
          {(
            [
              ["scores", "fused"],
              ["p_obstacle", "p(obstacle)"],
              ["threat", "threat"],
            ] as [ScoreSource, string][]
          ).map(([value, label]) => {
            const disabled = value !== "scores" && !hasScorer;
            const active = scoreSource === value;
            return (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={active}
                disabled={disabled}
                onClick={() => setScoreSource(value)}
                className={
                  active
                    ? "rounded-sm border border-accent bg-surface-2 px-2 py-0.5 font-mono text-[11px] text-accent-strong"
                    : "rounded-sm border border-border px-2 py-0.5 font-mono text-[11px] text-muted hover:bg-surface-3 disabled:cursor-not-allowed disabled:opacity-40"
                }
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

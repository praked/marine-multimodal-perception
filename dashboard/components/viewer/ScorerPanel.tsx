"use client";

import { formatBearing } from "@/lib/bins";
import { valuesFor, type SectorView } from "@/lib/sectors";
import type { ScoreSource } from "@/lib/stores/viewer";
import {
  arcPath,
  polarPoint,
  SCORER_VIEW,
  sectorPath,
  spokePath,
  toPx,
} from "@/lib/viewer/polar";
import { Compass } from "lucide-react";

const R = 9.0; // p = 1.0 spans the radar's 9 m envelope (poster geometry)
const GAP = 0.9;

interface Props {
  view: SectorView | null;
  scoreSource: ScoreSource;
  showConfirmed: boolean;
}

/** Polar per-sector score display, 1:1 with the poster's "learned scorer"
    panel: one wedge per sector (width follows the sector-width slider),
    radius ∝ value, radar-confirmed ▾ markers, p=0.5/1.0 arcs. */
export function ScorerPanel({ view: sv, scoreSource, showConfirmed }: Props) {
  const view = SCORER_VIEW;

  if (!sv) {
    return (
      <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 text-subtle">
        <Compass size={20} aria-hidden />
        <span className="font-mono text-[11px] uppercase tracking-[0.28em]">
          sectors · none
        </span>
        <span className="text-xs">no fusion output for this clip</span>
      </div>
    );
  }

  const values = valuesFor(sv, scoreSource);
  const maxIdx = values.reduce((best, v, i) => (v > values[best]! ? i : best), 0);
  const edges = sv.centers.map((c, i) => c - sv.halfWidths[i]!);
  edges.push(sv.centers[sv.centers.length - 1]! + sv.halfWidths[sv.halfWidths.length - 1]!);

  return (
    <svg
      viewBox={`0 0 ${view.width} ${view.height}`}
      className="h-full w-full"
      role="img"
      aria-label={`Per-sector ${scoreSource} polar chart`}
    >
      {/* spokes at every sector edge */}
      {edges.map((deg) => (
        <path key={deg} d={spokePath(deg, R, view)} stroke="var(--border)"
              strokeWidth={1} fill="none" />
      ))}
      {/* reference arcs: solid at 1.0, dashed at 0.5 */}
      <path d={arcPath(R, edges[0]!, edges[edges.length - 1]!, view)}
            stroke="var(--border-strong)" strokeWidth={1.2} fill="none" />
      <path d={arcPath(R / 2, edges[0]!, edges[edges.length - 1]!, view)}
            stroke="var(--border-strong)" strokeWidth={1}
            strokeDasharray="5 4" fill="none" />
      {(
        [
          [R, "1.0"],
          [R / 2, "0.5"],
        ] as const
      ).map(([r, label]) => {
        const [x, y] = polarPoint(edges[0]! - 4, r);
        const [px, py] = toPx(x, y, view);
        return (
          <text key={label} x={px} y={py} textAnchor="end" fontSize={11}
                fontFamily="var(--font-mono)" fill="var(--subtle)">
            {scoreSource === "threat" ? label : `p = ${label}`}
          </text>
        );
      })}
      {/* sector wedges */}
      {sv.centers.map((c, i) => {
        const v = Math.min(Math.max(values[i] ?? 0, 0), 1);
        if (v <= 0.005) return null;
        return (
          <path key={c} d={sectorPath(c, sv.halfWidths[i]!, R * v, view, GAP)}
                fill="var(--seeblau-deep)" stroke="var(--background)"
                strokeWidth={1.2}>
            <title>{`${formatBearing(c)} (±${sv.halfWidths[i]}°) · ${scoreSource} ${v.toFixed(2)}${
              sv.confirmed?.[i] ? " · radar-confirmed" : ""
            }`}</title>
          </path>
        );
      })}
      {/* radar-confirmed markers just past each wedge tip */}
      {showConfirmed &&
        sv.confirmed?.map((conf, i) => {
          if (!conf) return null;
          const v = Math.min(Math.max(values[i] ?? 0, 0), 1);
          const [x, y] = polarPoint(sv.centers[i]!, R * v + 0.55);
          const [px, py] = toPx(x, y, view);
          return (
            <path key={i}
                  d={`M${px - 5},${py - 4} L${px + 5},${py - 4} L${px},${py + 4} Z`}
                  fill="var(--seeblau-navy)">
              <title>{`${formatBearing(sv.centers[i]!)} · radar-confirmed (object-level match)`}</title>
            </path>
          );
        })}
      {/* strongest sector gets a direct value label */}
      {values[maxIdx]! > 0.05 &&
        (() => {
          const [x, y] = polarPoint(sv.centers[maxIdx]!, R * values[maxIdx]! + 1.35);
          const [px, py] = toPx(x, y, view);
          return (
            <text x={px} y={py} textAnchor="middle" fontSize={13}
                  fontWeight={600} fontFamily="var(--font-mono)"
                  fill="var(--foreground)">
              {values[maxIdx]!.toFixed(2)}
            </text>
          );
        })()}
      {/* bearing labels every 20° along the outer rim */}
      {[-40, -20, 0, 20, 40].map((deg) => {
        const [x, y] = polarPoint(deg, 9.85);
        const [px, py] = toPx(x, y, view);
        return (
          <text key={deg} x={px} y={py} textAnchor="middle" fontSize={11}
                fontFamily="var(--font-mono)" fill="var(--subtle)">
            {formatBearing(deg)}
          </text>
        );
      })}
      <text x={view.width / 2} y={view.height - 4} textAnchor="middle"
            fontSize={11} fontFamily="var(--font-mono)" fill="var(--subtle)">
        bearing (0° = dead ahead)
      </text>
    </svg>
  );
}

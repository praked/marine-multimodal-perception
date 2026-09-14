"use client";

import { formatBearing } from "@/lib/bins";
import type { SectorRecord } from "@/lib/types";

interface Props {
  record: SectorRecord | null;
}

const W = 264;
const H = 72;

/** Heading cost curve + recommendation, the web version of the Tkinter
    "Heading cost" panel. Green dashed = recommended heading; the smoothed
    value (what the autopilot would steer) is printed alongside. */
export function HeadingPanel({ record }: Props) {
  const candidates = record?.heading_candidates_deg;
  const curve = record?.heading_cost_curve;
  if (!record || !candidates || !curve || candidates.length < 2) {
    return (
      <p className="px-1 py-2 text-xs text-subtle">No heading data.</p>
    );
  }

  const maxCost = Math.max(0.05, ...curve) * 1.15;
  const minDeg = candidates[0]!;
  const maxDeg = candidates[candidates.length - 1]!;
  const sx = (deg: number) => ((deg - minDeg) / (maxDeg - minDeg)) * W;
  const sy = (cost: number) => H - (cost / maxCost) * H;
  const path = candidates
    .map(
      (deg, i) =>
        `${i === 0 ? "M" : "L"}${sx(deg).toFixed(1)},${sy(curve[i] ?? 0).toFixed(1)}`,
    )
    .join(" ");

  const rec = record.recommended_heading_deg;
  const abstained = rec == null;

  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Heading cost curve over candidate headings"
      >
        <line
          x1={sx(0)}
          y1={0}
          x2={sx(0)}
          y2={H}
          stroke="var(--border)"
          strokeWidth={1}
        />
        <path d={path} stroke="var(--seeblau-deep)" strokeWidth={2} fill="none" />
        {!abstained && (
          <line
            x1={sx(rec)}
            y1={0}
            x2={sx(rec)}
            y2={H}
            stroke="var(--status-good)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
        )}
      </svg>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-subtle">
        <span>{formatBearing(minDeg)}</span>
        <span>0°</span>
        <span>{formatBearing(maxDeg)}</span>
      </div>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-[11px] tabular-nums">
        <dt className="text-subtle">recommended</dt>
        <dd className={abstained ? "text-status-serious" : ""}>
          {abstained
            ? `abstain (${record.heading_reason ?? "?"})`
            : formatBearing(Math.round(rec))}
        </dd>
        <dt className="text-subtle">smoothed</dt>
        <dd>
          {record.smoothed_heading_deg == null
            ? "–"
            : formatBearing(Math.round(record.smoothed_heading_deg))}
        </dd>
        <dt className="text-subtle">reason</dt>
        <dd>{record.heading_reason ?? "–"}</dd>
      </dl>
    </div>
  );
}

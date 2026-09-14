"use client";

import type { Target } from "@/lib/types";
import { arcPath, RADAR_VIEW, spokePath, toPx } from "@/lib/viewer/polar";
import {
  TARGET_STATE_VAR,
  targetArrowEnd,
  targetCpaXY,
  targetLabel,
  targetXY,
} from "@/lib/viewer/targets";
import { RadioTower } from "lucide-react";
import { useMemo } from "react";

const DEFAULT_SPOKES = [-55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55];
const ARCS = [3, 6, 9];

interface Props {
  /** newest radar ts when the age gate emptied the trail (radar silent) */
  silentSince?: string | null;
  /** Radar groups of the clip in time order; [] when the stream is down. */
  trail: number[][][]; // last N groups, oldest first; last entry = current
  down: boolean;
  yMin: number; // self-clutter display filter
  /** Wedge-grid spoke bearings (follows the sector-width slider). */
  spokeEdges?: number[];
  /** Protocol-v1.2 tracked targets for the current frame (optional). */
  targets?: Target[] | null;
  /** "Target tracks" layer toggle; markers render only when true. */
  showTargets?: boolean;
}

/** Bird's-eye radar view, 1:1 with the poster panel: ±7.5 m lateral,
    0–9.4 m forward, 10° spoke grid over ±55°, range arcs at 3/6/9 m.
    Tracked targets (protocol v1.2) draw on top: motion-state-coloured
    ring + velocity arrow (∝ speed, capped) + dashed CPA ghost. */
export function RadarPanel({
  trail,
  down,
  yMin,
  spokeEdges,
  targets,
  showTargets = false,
  silentSince = null,
}: Props) {
  const view = RADAR_VIEW;
  const spokes = spokeEdges ?? DEFAULT_SPOKES;
  const kept = useMemo(() => {
    const current = trail.length > 0 ? trail[trail.length - 1]! : [];
    return current.filter((p) => (p[1] ?? 0) >= yMin);
  }, [trail, yMin]);
  const shownTargets = showTargets && targets?.length ? targets : [];

  if (down) {
    return (
      <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 text-subtle">
        <RadioTower size={20} aria-hidden />
        <span className="font-mono text-[11px] uppercase tracking-[0.28em]">
          radar · down
        </span>
        <span className="text-xs">no mmWave stream in this clip</span>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      {silentSince != null && trail.length === 0 && (
        <div className="pointer-events-none absolute inset-x-0 top-2 z-10 flex justify-center">
          <span className="rounded-sm border border-status-warn/60 bg-surface-1/90 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-status-warn">
            radar silent{silentSince !== "start" ? ` since ${silentSince}` : ""}
          </span>
        </div>
      )}
    <svg
      viewBox={`0 0 ${view.width} ${view.height}`}
      className="h-full w-full"
      role="img"
      aria-label={
        `Radar bird's-eye view, ${kept.length} returns` +
        (shownTargets.length > 0
          ? `, ${shownTargets.length} tracked target${shownTargets.length === 1 ? "" : "s"}`
          : "")
      }
    >
      {/* wedge grid */}
      {spokes.map((deg) => (
        <path
          key={deg}
          d={spokePath(deg, 9, view)}
          stroke="var(--border)"
          strokeWidth={1}
          fill="none"
        />
      ))}
      {ARCS.map((r) => (
        <path
          key={r}
          d={arcPath(r, -55, 55, view)}
          stroke="var(--border)"
          strokeWidth={1}
          fill="none"
        />
      ))}
      {/* range labels along the 0° spoke */}
      {ARCS.map((r) => {
        const [px, py] = toPx(0.15, r + 0.12, view);
        return (
          <text
            key={r}
            x={px}
            y={py}
            fontSize={11}
            fontFamily="var(--font-mono)"
            fill="var(--subtle)"
          >
            {r} m
          </text>
        );
      })}
      {/* fading trail (older = fainter) */}
      {trail.slice(0, -1).map((group, gi) => {
        const alpha = ((gi + 1) / trail.length) * 0.45;
        return group
          .filter((p) => (p[1] ?? 0) >= yMin)
          .map((p, pi) => {
            const [px, py] = toPx(p[0]!, p[1]!, view);
            return (
              <circle
                key={`${gi}-${pi}`}
                cx={px}
                cy={py}
                r={2.6}
                fill="var(--viz-trail)"
                fillOpacity={alpha}
              />
            );
          });
      })}
      {/* current returns */}
      {kept.map((p, i) => {
        const [px, py] = toPx(p[0]!, p[1]!, view);
        return (
          <circle
            key={i}
            cx={px}
            cy={py}
            r={4}
            fill="var(--seeblau-deep)"
            fillOpacity={0.85}
          >
            <title>
              {`x ${p[0]!.toFixed(2)} m · y ${p[1]!.toFixed(2)} m` +
                (p.length > 3 ? ` · v ${p[3]!.toFixed(2)} m/s` : "")}
            </title>
          </circle>
        );
      })}
      {/* tracked targets (protocol v1.2), on top of the raw returns */}
      {shownTargets.map((t) => {
        const [px, py] = toPx(...targetXY(t), view);
        const colour = TARGET_STATE_VAR[t.motion_state];
        const arrowEnd = targetArrowEnd(t);
        const cpa = targetCpaXY(t);
        const label = targetLabel(t);
        let arrow: React.ReactNode = null;
        if (arrowEnd) {
          const [ax, ay] = toPx(arrowEnd[0], arrowEnd[1], view);
          const a = Math.atan2(ay - py, ax - px);
          const h = 6; // arrowhead px
          const head = [
            `${ax.toFixed(1)},${ay.toFixed(1)}`,
            `${(ax - h * Math.cos(a - 0.45)).toFixed(1)},${(ay - h * Math.sin(a - 0.45)).toFixed(1)}`,
            `${(ax - h * Math.cos(a + 0.45)).toFixed(1)},${(ay - h * Math.sin(a + 0.45)).toFixed(1)}`,
          ].join(" ");
          arrow = (
            <>
              <line
                x1={px}
                y1={py}
                x2={ax}
                y2={ay}
                stroke={colour}
                strokeWidth={2}
                data-target-arrow
              />
              <polygon points={head} fill={colour} />
            </>
          );
        }
        return (
          <g key={t.id} role="img" aria-label={label} data-target-id={t.id}>
            {cpa && (
              <>
                <line
                  x1={px}
                  y1={py}
                  x2={toPx(cpa[0], cpa[1], view)[0]}
                  y2={toPx(cpa[0], cpa[1], view)[1]}
                  stroke={colour}
                  strokeWidth={1.2}
                  strokeDasharray="4 3"
                  opacity={0.6}
                  data-target-cpa-connector
                />
                <circle
                  cx={toPx(cpa[0], cpa[1], view)[0]}
                  cy={toPx(cpa[0], cpa[1], view)[1]}
                  r={5}
                  fill="none"
                  stroke={colour}
                  strokeWidth={1.4}
                  strokeDasharray="3 2"
                  opacity={0.7}
                  data-target-cpa-ghost
                />
              </>
            )}
            {arrow}
            <circle
              cx={px}
              cy={py}
              r={6.5}
              fill="none"
              stroke={colour}
              strokeWidth={2.2}
              data-target-marker
            />
            <title>{label}</title>
          </g>
        );
      })}
      {/* axis labels */}
      <text
        x={view.width / 2}
        y={view.height - 4}
        textAnchor="middle"
        fontSize={11}
        fontFamily="var(--font-mono)"
        fill="var(--subtle)"
      >
        lateral (m)
      </text>
      <text
        x={10}
        y={view.height / 2}
        fontSize={11}
        fontFamily="var(--font-mono)"
        fill="var(--subtle)"
        transform={`rotate(-90 10 ${view.height / 2})`}
        textAnchor="middle"
      >
        forward (m)
      </text>
    </svg>
    </div>
  );
}

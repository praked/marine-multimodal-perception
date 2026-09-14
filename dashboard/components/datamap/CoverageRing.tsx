"use client";

import type { Band } from "@/lib/datamap";
import { useState } from "react";

/* Generic coverage ring: a donut whose angular extent maps a value domain
   (hours of day, % cloud cover, wind speed…) and whose segments are shaded
   by captured-frame density. Gaps in the data read as literal gaps in the
   ring. Portable: domain + bands + labels only. */

const SIZE = 200;
const C = SIZE / 2;
const R_OUT = 78;
const R_IN = 56;

function polar(r: number, frac: number, startDeg: number, sweepDeg: number): [number, number] {
  const a = ((startDeg + frac * sweepDeg - 90) * Math.PI) / 180;
  return [C + r * Math.cos(a), C + r * Math.sin(a)];
}

function segmentPath(
  f0: number,
  f1: number,
  startDeg: number,
  sweepDeg: number,
): string {
  const [x0o, y0o] = polar(R_OUT, f0, startDeg, sweepDeg);
  const [x1o, y1o] = polar(R_OUT, f1, startDeg, sweepDeg);
  const [x0i, y0i] = polar(R_IN, f0, startDeg, sweepDeg);
  const [x1i, y1i] = polar(R_IN, f1, startDeg, sweepDeg);
  const large = (f1 - f0) * Math.abs(sweepDeg) > 180 ? 1 : 0;
  const sweep = sweepDeg > 0 ? 1 : 0;
  return [
    `M${x0o},${y0o}`,
    `A${R_OUT},${R_OUT} 0 ${large} ${sweep} ${x1o},${y1o}`,
    `L${x1i},${y1i}`,
    `A${R_IN},${R_IN} 0 ${large} ${1 - sweep} ${x0i},${y0i}`,
    "Z",
  ].join(" ");
}

const RAMP = ["#cceef9", "#a6e1f4", "#59c7eb", "#00a9e0", "#1487b8"];

export function CoverageRing({
  title,
  unit,
  bands,
  domain,
  tickEvery,
  startDeg = 0,
  sweepDeg = 360,
  format = (v: number) => String(v),
  goals,
  framesToHours,
}: {
  title: string;
  unit: string;
  bands: Band[];
  domain: [number, number];
  tickEvery: number;
  startDeg?: number;
  sweepDeg?: number;
  format?: (v: number) => string;
  /** target hours per band (aligned with `bands`) — under-target bands get
      an amber outline and the hover bubble shows progress toward the goal */
  goals?: number[];
  framesToHours?: (frames: number) => number;
}) {
  const [d0, d1] = domain;
  const span = d1 - d0;
  const maxV = Math.max(1, ...bands.map((b) => b.value));
  const covered = bands.filter((b) => b.value > 0).length;
  const [hovered, setHovered] = useState<number | null>(null);

  const ticks: number[] = [];
  for (let v = d0; v <= d1; v += tickEvery) ticks.push(v);
  // on a full circle the domain endpoints coincide — drop the duplicate
  if (Math.abs(sweepDeg) >= 360 && ticks.length > 1) ticks.pop();

  const hb = hovered != null ? bands[hovered] : null;
  const hGoal = hovered != null ? goals?.[hovered] : undefined;
  const toH = framesToHours ?? ((f: number) => f / (3 * 3600));
  return (
    <figure className="relative flex flex-col items-center">
      {hb && (
        <div className="pointer-events-none absolute bottom-full left-1/2 z-10 w-56 -translate-x-1/2 rounded-sm border border-border bg-surface-1 p-2 text-left shadow-sm">
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-subtle">
            <span>{title}</span>
            <span>{format(hb.from)}–{format(hb.to)} {unit}</span>
          </div>
          <div className="mt-0.5 font-mono text-sm tabular-nums">
            {Math.round(hb.value).toLocaleString("en-GB")} frames
          </div>
          {hGoal != null && hGoal > 0 && (
            <div
              className={`font-mono text-[11px] tabular-nums ${toH(hb.value) >= hGoal ? "text-status-good" : "text-status-warn"}`}
            >
              {toH(hb.value).toFixed(1)}h of {hGoal}h goal
              {toH(hb.value) >= hGoal ? " ✓" : ""}
            </div>
          )}
          {hb.clips.length > 0 ? (
            <ul className="mt-1 space-y-0.5 border-t border-border pt-1 text-[11px]">
              {hb.clips.slice(0, 6).map((c) => (
                <li key={c.clipId} className="flex justify-between gap-2">
                  <span className="truncate text-muted">{c.title}</span>
                  <span className="shrink-0 font-mono tabular-nums">
                    {c.value.toLocaleString("en-GB")}f
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="mt-1 border-t border-border pt-1 text-[11px] text-status-warn">
              gap — no clips in this band
            </div>
          )}
        </div>
      )}
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-44" role="img"
           aria-label={`${title} coverage ring`}>
        {bands.map((b, i) => {
          const f0 = (b.from - d0) / span;
          const f1 = (b.to - d0) / span;
          const shade =
            b.value === 0
              ? "var(--surface-3)"
              : RAMP[Math.min(RAMP.length - 1,
                  Math.floor((b.value / maxV) * RAMP.length))]!;
          return (
            <path key={i} d={segmentPath(f0 + 0.002, f1 - 0.002, startDeg, sweepDeg)}
                  fill={shade} stroke="var(--background)" strokeWidth={1}
                  className="cursor-pointer"
                  opacity={hovered != null && hovered !== i ? 0.45 : 1}
                  onMouseEnter={() => setHovered(i)}
                  onMouseLeave={() => setHovered(null)}>
              <title>
                {`${format(b.from)}–${format(b.to)} ${unit}: ${Math.round(b.value)} frames` +
                  (b.clips.length
                    ? "\n" +
                      b.clips
                        .slice(0, 6)
                        .map((c) => ` · ${c.title}: ${c.value.toLocaleString("en-GB")}f`)
                        .join("\n")
                    : "\n · no clips in this band")}
              </title>
            </path>
          );
        })}
        {ticks.map((v) => {
          const f = (v - d0) / span;
          const [x, y] = polar(R_OUT + 12, f, startDeg, sweepDeg);
          return (
            <text key={v} x={x} y={y + 3} textAnchor="middle" fontSize={9}
                  fontFamily="var(--font-mono)" fill="var(--subtle)">
              {format(v)}
            </text>
          );
        })}
        <text x={C} y={C - 4} textAnchor="middle" fontSize={20} fontWeight={600}
              fontFamily="var(--font-mono)" fill="var(--foreground)">
          {covered}/{bands.length}
        </text>
        <text x={C} y={C + 12} textAnchor="middle" fontSize={9}
              fontFamily="var(--font-mono)" fill="var(--subtle)">
          bands covered
        </text>
      </svg>
      <figcaption className="mt-1 text-center">
        <div className="text-xs font-medium">{title}</div>
        <div className="font-mono text-[10px] text-subtle">{unit}</div>
      </figcaption>
    </figure>
  );
}

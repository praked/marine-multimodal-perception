"use client";

import type { EntityCoverage } from "@/lib/datamap";

/* Entity-vs-target gauges: one arc per class, sweep = progress toward the
   dataset target, in the class's established colour. A hollow arc with a
   red-tinted label = "we've seen none of these yet" — the loudest gap. */

const SIZE = 108;
const C = SIZE / 2;
const R = 42;
const START = -210; // degrees; arc spans -210..30 (240° gauge)
const SWEEP = 240;

function arcPath(frac: number): string {
  const a0 = ((START - 90) * Math.PI) / 180;
  const a1 = ((START + SWEEP * Math.min(frac, 1) - 90) * Math.PI) / 180;
  const large = SWEEP * Math.min(frac, 1) > 180 ? 1 : 0;
  return `M${C + R * Math.cos(a0)},${C + R * Math.sin(a0)} A${R},${R} 0 ${large} 1 ${C + R * Math.cos(a1)},${C + R * Math.sin(a1)}`;
}

export function TargetArcs({
  items,
  colour,
}: {
  items: EntityCoverage[];
  colour: (cls: string) => string;
}) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((e) => {
        const frac = e.target > 0 ? e.instances / e.target : 1;
        const done = e.target > 0 && e.instances >= e.target;
        return (
          <figure key={e.cls} className="group relative flex flex-col items-center">
            {e.contributions.length > 0 && (
              <div className="pointer-events-none absolute bottom-full left-1/2 z-10 hidden w-60 -translate-x-1/2 rounded-sm border border-border bg-surface-1 p-2 shadow-sm group-hover:block">
                <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-subtle">
                  {e.cls} · provenance
                </div>
                <ul className="space-y-0.5 text-[11px]">
                  {e.contributions.slice(0, 5).map((c) => (
                    <li key={c.clipId} className="flex justify-between gap-2">
                      <span className="truncate text-muted">{c.title}</span>
                      <span className="shrink-0 font-mono tabular-nums">
                        {c.instances.toLocaleString("en-GB")}
                      </span>
                    </li>
                  ))}
                </ul>
                <div className="mt-1 border-t border-border pt-1 font-mono text-[10px] text-subtle">
                  {Object.entries(
                    e.contributions.reduce<Record<string, number>>((acc, c) => {
                      for (const [s2, n] of Object.entries(c.sources)) acc[s2] = (acc[s2] ?? 0) + n;
                      return acc;
                    }, {}),
                  )
                    .map(([s2, n]) => `${s2}: ${n.toLocaleString("en-GB")}`)
                    .join(" · ")}
                </div>
                <div className="mt-0.5 text-[10px] leading-snug text-subtle">
                  headline = max per clip across streams (never summed)
                </div>
              </div>
            )}
            <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-24" role="img"
                 aria-label={`${e.cls}: ${e.instances} of ${e.target} target instances`}>
              <path d={arcPath(1)} fill="none" stroke="var(--surface-3)"
                    strokeWidth={9} strokeLinecap="round" />
              {e.instances > 0 && (
                <path d={arcPath(frac)} fill="none" stroke={colour(e.cls)}
                      strokeWidth={9} strokeLinecap="round">
                  <title>{`${e.cls}: ${e.instances} instances over ${e.frames} frames (target ${e.target})`}</title>
                </path>
              )}
              <text x={C} y={C - 2} textAnchor="middle" fontSize={17}
                    fontWeight={600} fontFamily="var(--font-mono)"
                    fill={e.instances === 0 ? "var(--status-serious)" : "var(--foreground)"}>
                {e.instances >= 10000
                  ? `${Math.round(e.instances / 1000)}k`
                  : e.instances}
              </text>
              <text x={C} y={C + 14} textAnchor="middle" fontSize={9}
                    fontFamily="var(--font-mono)" fill="var(--subtle)">
                / {e.target || "–"}
              </text>
              {done && (
                <text x={C} y={C + 34} textAnchor="middle" fontSize={10}
                      fill="var(--status-good)">✓</text>
              )}
            </svg>
            <figcaption className="mt-0.5 flex items-center gap-1.5 text-xs">
              <span aria-hidden className="h-2 w-2 rounded-full"
                    style={{ background: colour(e.cls) }} />
              {e.cls}
            </figcaption>
          </figure>
        );
      })}
    </div>
  );
}

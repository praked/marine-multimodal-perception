"use client";

import type { EntityConditionMatrix } from "@/lib/datamap";
import { useState } from "react";

/* Advanced mode: entity-class x capture-condition intersection matrix.
   Answers "under which conditions have we actually seen each class?" —
   a class detected only in the rain is a deployment risk in the sun.
   Cell shading = instance count; amber ring = GAP (class exists in the
   corpus, condition has footage, but the two never intersect). */

const RAMP = ["#cceef9", "#a6e1f4", "#59c7eb", "#00a9e0", "#1487b8"];

export function CrossMatrix({
  matrix,
  colour,
}: {
  matrix: EntityConditionMatrix;
  colour: (cls: string) => string;
}) {
  const [hovered, setHovered] = useState<[string, string] | null>(null);
  const { classes, conditions, cells } = matrix;
  const max = Math.max(
    1,
    ...classes.flatMap((cls) =>
      conditions.map((c) => cells[cls]?.[c.key]?.instances ?? 0),
    ),
  );
  const groups: { group: string; span: number }[] = [];
  for (const c of conditions) {
    const last = groups[groups.length - 1];
    if (last && last.group === c.group) last.span += 1;
    else groups.push({ group: c.group, span: 1 });
  }
  const hb = hovered ? cells[hovered[0]]?.[hovered[1]] : null;
  const hCond = hovered ? conditions.find((c) => c.key === hovered[1]) : null;

  return (
    <div className="relative overflow-x-auto">
      {hovered && hb && hCond && (
        <div className="pointer-events-none absolute left-1/2 top-0 z-10 w-64 -translate-x-1/2 rounded-sm border border-border bg-surface-1 p-2 shadow-sm">
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-subtle">
            <span>{hovered[0]}</span>
            <span>
              {hCond.group} · {hCond.label}
            </span>
          </div>
          <div className="mt-0.5 font-mono text-sm tabular-nums">
            {hb.instances.toLocaleString("en-GB")} instances
          </div>
          {hb.clips.length > 0 ? (
            <ul className="mt-1 space-y-0.5 border-t border-border pt-1 text-[11px]">
              {hb.clips.slice(0, 5).map((c) => (
                <li key={c.clipId} className="flex justify-between gap-2">
                  <span className="truncate text-muted">{c.title}</span>
                  <span className="shrink-0 font-mono tabular-nums">
                    {c.value.toLocaleString("en-GB")}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="mt-1 border-t border-border pt-1 text-[11px] text-status-warn">
              {hb.gap
                ? "gap — class never captured under this condition"
                : "no footage under this condition yet"}
            </div>
          )}
        </div>
      )}
      <table className="border-collapse font-mono text-[10px]">
        <thead>
          <tr>
            <th />
            {groups.map((g) => (
              <th
                key={g.group}
                colSpan={g.span}
                className="border-b border-border px-1 pb-1 text-left font-medium uppercase tracking-wider text-muted"
              >
                {g.group}
              </th>
            ))}
          </tr>
          <tr>
            <th />
            {conditions.map((c) => (
              <th
                key={c.key}
                className="px-0.5 pb-1 text-center font-normal text-subtle"
              >
                <span className="inline-block max-w-12 truncate align-bottom">
                  {c.label}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {classes.map((cls) => (
            <tr key={cls}>
              <td className="whitespace-nowrap py-0.5 pr-2 text-right text-[11px]">
                <span
                  aria-hidden
                  className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle"
                  style={{ background: colour(cls) }}
                />
                {cls}
              </td>
              {conditions.map((cond) => {
                const cell = cells[cls]?.[cond.key];
                const n = cell?.instances ?? 0;
                const shade =
                  n === 0
                    ? "var(--surface-2)"
                    : RAMP[
                        Math.min(
                          RAMP.length - 1,
                          Math.floor((n / max) * RAMP.length),
                        )
                      ];
                return (
                  <td key={cond.key} className="p-0.5">
                    <div
                      role="img"
                      aria-label={`${cls} under ${cond.group} ${cond.label}: ${n} instances${cell?.gap ? " (gap)" : ""}`}
                      onMouseEnter={() => setHovered([cls, cond.key])}
                      onMouseLeave={() => setHovered(null)}
                      className="flex h-7 w-10 cursor-pointer items-center justify-center rounded-[2px] tabular-nums"
                      style={{
                        background: shade,
                        boxShadow: cell?.gap
                          ? "inset 0 0 0 1.5px var(--viz-water-edge)"
                          : undefined,
                        color:
                          n / max > 0.55 ? "#ffffff" : "var(--muted)",
                      }}
                    >
                      {n > 0
                        ? n >= 10000
                          ? `${Math.round(n / 1000)}k`
                          : n.toLocaleString("en-GB")
                        : cell?.gap
                          ? "!"
                          : "·"}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 font-mono text-[10px] text-subtle">
        cell = instances of the class captured under the condition (within-clip
        max across detector streams, summed across clips) ·{" "}
        <span className="text-status-warn">amber ring</span> = class exists in
        the corpus but never under this condition · &quot;·&quot; = no footage
        there at all
      </div>
    </div>
  );
}

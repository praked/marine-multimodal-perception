"use client";

import type { HealthRun } from "@/lib/health";

/* Discreet per-sensor health timeline: a slim pastel bar under the feed —
   green where the sensor delivered, red where it dropped. Curation cut
   regions (frames trimmed in the viewer) render greyed over the bar so
   the timeline shows what the corpus will actually keep. */

export function HealthBar({
  runs,
  playhead,
  label,
  note,
  cuts = [],
}: {
  runs: HealthRun[];
  playhead: number; // fraction 0..1
  label: string;
  note?: string;
  /** cut regions as timeline fractions (lib/curation cutRuns) */
  cuts?: { start: number; width: number }[];
}) {
  if (runs.length === 0) return null;
  const okFrac = runs.filter((r) => r.state === "ok").reduce((a, r) => a + r.width, 0);
  const degFrac = runs.filter((r) => r.state === "degraded").reduce((a, r) => a + r.width, 0);
  const cutFrac = cuts.reduce((a, c) => a + c.width, 0);
  return (
    <div
      className="relative mx-1.5 mb-1 h-[5px] overflow-hidden rounded-full"
      role="img"
      aria-label={`${label} health: ${(okFrac * 100).toFixed(0)}% healthy${degFrac > 0 ? `, ${(degFrac * 100).toFixed(0)}% degraded` : ""}${cutFrac > 0 ? `, ${(cutFrac * 100).toFixed(0)}% cut` : ""}`}
      title={`${label} health — ${(okFrac * 100).toFixed(0)}% healthy${degFrac > 0 ? ` · ${(degFrac * 100).toFixed(0)}% present but degraded (below quality floors)` : ""}${cutFrac > 0 ? ` · ${(cutFrac * 100).toFixed(0)}% cut by curation (greyed)` : ""}${note ? ` · ${note}` : ""}`}
    >
      {runs.map((r, i) => (
        <div
          key={i}
          className="absolute h-full"
          style={{
            left: `${r.start * 100}%`,
            width: `${r.width * 100}%`,
            background: r.state === "ok" ? "#b9e2c4"
              : r.state === "degraded" ? "#f3d9a4" // pastel amber
              : "#f2b8b1", // pastel green / amber / red
          }}
        />
      ))}
      {cuts.map((c, i) => (
        <div
          key={`cut-${i}`}
          data-cut-run
          className="absolute h-full"
          style={{
            left: `${c.start * 100}%`,
            width: `${c.width * 100}%`,
            background: "repeating-linear-gradient(135deg, #8a8f94 0 2px, #c5c9cc 2px 4px)",
            opacity: 0.9,
          }}
          aria-hidden
        />
      ))}
      <div
        className="absolute h-full w-[2px]"
        style={{ left: `${playhead * 100}%`, background: "var(--seeblau-navy)", opacity: 0.6 }}
        aria-hidden
      />
    </div>
  );
}

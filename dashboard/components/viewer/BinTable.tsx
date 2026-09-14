"use client";

import { formatBearing } from "@/lib/bins";
import { cn } from "@/lib/cn";
import { valuesFor, type SectorView } from "@/lib/sectors";

interface Props {
  view: SectorView | null;
  threshold: number;
  scoreSource?: "scores" | "p_obstacle" | "threat";
}

function fmt(v: number | null | undefined, digits = 1, suffix = ""): string {
  if (v == null) return "–";
  return `${v.toFixed(digits)}${suffix}`;
}

const SENSOR_LABELS = ["F", "T", "M"] as const;
const SENSOR_VARS = [
  "var(--viz-fisheye)",
  "var(--viz-thermal)",
  "var(--viz-radar)",
] as const;

/** Per-sector fusion table (follows the sector-width slider): score,
    per-sensor hits, min range, closing speed, TTC, confirmation.
    Row states: blocked (score ≥ 2×threshold) and urgent (ttc ≤ 3 s). */
export function BinTable({ view, threshold, scoreSource = "scores" }: Props) {
  if (!view) {
    return (
      <p className="px-1 py-2 text-xs text-subtle">
        No fusion record for this frame.
      </p>
    );
  }
  const blockedAt = Math.min(2 * threshold, 0.66);
  const shown = valuesFor(view, scoreSource);
  return (
    <table className="w-full border-collapse font-mono text-[11px] tabular-nums">
      <thead>
        <tr className="border-b border-border text-left text-[10px] uppercase tracking-wider text-subtle">
          <th className="py-1 pr-1 font-medium">sector</th>
          <th className="py-1 pr-1 font-medium">
            {scoreSource === "p_obstacle" ? "p(obs)" : scoreSource === "threat" ? "threat" : "score"}
          </th>
          <th className="py-1 pr-1 font-medium">f·t·m</th>
          <th className="py-1 pr-1 text-right font-medium">range</th>
          <th className="py-1 pr-1 text-right font-medium">v</th>
          <th className="py-1 pr-1 text-right font-medium">ttc</th>
          <th className="py-1 font-medium">✓</th>
        </tr>
      </thead>
      <tbody>
        {view.centers.map((c, i) => {
          const score = shown[i] ?? 0;
          const ttc = view.ttc?.[i];
          const urgent = ttc != null && ttc <= 3 && score >= threshold;
          const blocked = score >= blockedAt;
          const hits = view.sensorHits[i] ?? [0, 0, 0];
          return (
            <tr
              key={c}
              className={cn(
                "border-b border-border/60",
                blocked && "bg-surface-3 font-semibold",
                urgent && "bg-status-serious/10",
              )}
              title={
                urgent
                  ? "Urgent: closing with TTC ≤ 3 s"
                  : blocked
                    ? "Blocked at the nav threshold"
                    : undefined
              }
            >
              <td className="py-1 pr-1">
                {formatBearing(c)}
                {view.halfWidths[i]! > 5 && (
                  <span className="text-subtle">±{view.halfWidths[i]}</span>
                )}
              </td>
              <td className="py-1 pr-1">{score.toFixed(2)}</td>
              <td className="py-1 pr-1">
                <span className="inline-flex items-center gap-1">
                  {hits.map((hit, s) => (
                    <span
                      key={s}
                      aria-label={`${SENSOR_LABELS[s]} ${hit ? "hit" : "no hit"}`}
                      className="inline-block h-2 w-2 rounded-full border"
                      style={{
                        background: hit ? SENSOR_VARS[s] : "transparent",
                        borderColor: SENSOR_VARS[s],
                        opacity: hit ? 1 : 0.35,
                      }}
                    />
                  ))}
                </span>
              </td>
              <td className="py-1 pr-1 text-right">
                {fmt(view.minRange[i], 1, " m")}
              </td>
              <td className="py-1 pr-1 text-right">
                {fmt(view.velocity?.[i], 2)}
              </td>
              <td className="py-1 pr-1 text-right">{fmt(ttc, 1, " s")}</td>
              <td className="py-1">
                {view.confirmed?.[i] ? (
                  <span className="text-accent-strong" title="radar-confirmed">
                    ▾
                  </span>
                ) : (
                  <span className="text-subtle/50">·</span>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

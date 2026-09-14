"use client";

import type { Target } from "@/lib/types";
import { TARGET_STATE_VAR, targetLabel } from "@/lib/viewer/targets";

/* Compact per-target chip row under the radar bird's-eye: one mono chip
   per tracked target (protocol v1.2), nearest first as emitted — state,
   range, closing speed, CPA. Dot colour matches the panel's motion-state
   colouring so the row indexes the markers above it. */

function fmt(v: number | null, digits: number, suffix = ""): string {
  return v == null ? "–" : `${v.toFixed(digits)}${suffix}`;
}

export function TargetChips({ targets }: { targets: Target[] }) {
  if (targets.length === 0) return null;
  return (
    <ul
      aria-label="Tracked targets"
      className="mx-1.5 mb-1 flex flex-wrap gap-1"
    >
      {targets.map((t) => (
        <li
          key={t.id}
          title={targetLabel(t)}
          className="flex items-center gap-1.5 rounded-sm border border-border px-1.5 py-0.5 font-mono text-[10px] tabular-nums text-muted"
        >
          <span
            aria-hidden
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: TARGET_STATE_VAR[t.motion_state] }}
          />
          <span className="text-foreground">#{t.id}</span>
          <span>{t.motion_state}</span>
          <span>{t.range_m.toFixed(1)} m</span>
          <span>{fmt(t.closing_mps, 2, " m/s")}</span>
          <span>
            cpa{" "}
            {t.cpa_m != null && t.t_cpa_s != null
              ? `${t.cpa_m.toFixed(1)} m/${t.t_cpa_s.toFixed(0)} s`
              : "–"}
          </span>
        </li>
      ))}
    </ul>
  );
}

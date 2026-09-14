"use client";

/* Luminosity spectrum: a dark-to-bright strip whose bar heights show how
   many sampled frames fall in each luminance bin. Empty bins = literal
   holes in the captured light conditions. Generic: hist + domain labels. */

import type { Band } from "@/lib/datamap";
import { useState } from "react";

export function SpectrumStrip({
  hist,
  bins,
  leftLabel = "dark",
  rightLabel = "bright",
  title = "Luminosity coverage",
}: {
  hist: number[];
  bins?: Band[];
  leftLabel?: string;
  rightLabel?: string;
  title?: string;
}) {
  const max = Math.max(1, ...hist);
  const [hovered, setHovered] = useState<number | null>(null);
  const hb = hovered != null ? bins?.[hovered] : null;
  const hv = hovered != null ? hist[hovered] : null;
  return (
    <figure className="relative">
      {hovered != null && (
        <div
          className="pointer-events-none absolute bottom-full z-10 w-56 -translate-x-1/2 rounded-sm border border-border bg-surface-1 p-2 shadow-sm"
          style={{ left: `${((hovered + 0.5) / hist.length) * 100}%` }}
        >
          <div className="flex justify-between font-mono text-[10px] uppercase tracking-wider text-subtle">
            <span>luminance</span>
            <span>
              {Math.round((hovered * 256) / hist.length)}–
              {Math.round(((hovered + 1) * 256) / hist.length)} luma
            </span>
          </div>
          <div className="mt-0.5 font-mono text-sm tabular-nums">
            {(hv ?? 0).toLocaleString("en-GB")} sampled frames
          </div>
          {hb && hb.clips.length > 0 ? (
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
              {hv === 0 ? "gap — no frames at this light level" : "attribution: bins hold frames from clips whose mean luma lies elsewhere"}
            </div>
          )}
        </div>
      )}
      <figcaption className="mb-1.5 text-xs font-medium">{title}</figcaption>
      <div
        className="flex h-20 items-end gap-px overflow-hidden rounded-sm border border-border p-1"
        style={{
          background:
            "linear-gradient(90deg, #0a1418 0%, #35505e 45%, #cfe6f2 100%)",
        }}
        role="img"
        aria-label="Luminance histogram of sampled frames, dark to bright"
      >
        {hist.map((v, i) => (
          <div
            key={i}
            className="flex-1 cursor-pointer rounded-t-[2px]"
            onMouseEnter={() => setHovered(i)}
            onMouseLeave={() => setHovered(null)}
            style={{
              height: v === 0 ? "2px" : `${8 + (v / max) * 88}%`,
              background: v === 0 ? "rgba(242,163,60,0.55)" : "var(--seeblau-65)",
              opacity: v === 0 ? 1 : 0.92,
            }}
            title={
              `luma ${Math.round((i * 256) / hist.length)}–${Math.round(((i + 1) * 256) / hist.length)}: ${v} sampled frames${v === 0 ? " — GAP" : ""}` +
              (bins?.[i]?.clips.length
                ? "\n" +
                  bins[i]!.clips
                    .slice(0, 5)
                    .map((c) => ` · ${c.title}`)
                    .join("\n")
                : "")
            }
          />
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-subtle">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </figure>
  );
}

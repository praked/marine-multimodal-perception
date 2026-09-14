"use client";

import type { SunArc } from "@/lib/datamap";
import { useMemo, useState } from "react";

/* The data-map hero: a sky-dome semicircle. Azimuth sweeps across the
   dome (E -> S -> W), elevation climbs from the horizon rim toward the
   zenith point; each clip is drawn as its actual sun path. The seasonal
   envelope (solstice sun paths for the site latitude) frames what's
   physically capturable; the band below the horizon is night.

   Generic: consumes SunArc[] + latitude only — portable to any project. */

const W = 900;
const H = 470;
const CX = W / 2;
const CY = 430;
const R = 380;
const NIGHT_H = 34;

function project(elevDeg: number, azDeg: number): [number, number] {
  // Classic solar-path chart: azimuth (E..S..W) maps across, elevation maps
  // up — sun paths read as natural rise-peak-set arcs. The dome outline is
  // aesthetic framing (drawing clipped to it).
  const t = Math.min(Math.max((azDeg - 60) / 240, 0), 1);
  const x = CX - R + 2 * R * t;
  const y = CY - (Math.min(Math.max(elevDeg, 0), 90) / 90) * (R * 0.985);
  return [x, y];
}

/** Sun path for a solar declination at given latitude — pure geometry,
    no clock/DST involved. Used for the solstice envelope. */
export function declinationPath(
  latDeg: number,
  declDeg: number,
  steps = 60,
): [number, number][] {
  const lat = (latDeg * Math.PI) / 180;
  const dec = (declDeg * Math.PI) / 180;
  const out: [number, number][] = [];
  for (let i = 0; i <= steps; i++) {
    const Hdeg = -120 + (240 * i) / steps; // hour angle
    const Ha = (Hdeg * Math.PI) / 180;
    const elev = Math.asin(
      Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(Ha),
    );
    if (elev < 0) continue;
    const az =
      Math.atan2(
        -Math.sin(Ha),
        Math.tan(dec) * Math.cos(lat) - Math.sin(lat) * Math.cos(Ha),
      ) *
      (180 / Math.PI);
    out.push([(elev * 180) / Math.PI, (az + 360) % 360]);
  }
  return out;
}

function pathD(points: [number, number][]): string {
  return points
    .map(([e, a], i) => {
      const [x, y] = project(e, a);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

const ELEV_RINGS = [15, 30, 45, 60, 75];
const AZ_TICKS: [number, string][] = [
  [90, "E"],
  [135, "SE"],
  [180, "S"],
  [225, "SW"],
  [270, "W"],
];

export function SkyDome({
  arcs,
  latitude,
  nightFrames = 0,
}: {
  arcs: SunArc[];
  latitude: number;
  nightFrames?: number;
}) {
  const envelope = useMemo(
    () => ({
      summer: declinationPath(latitude, 23.44),
      winter: declinationPath(latitude, -23.44),
    }),
    [latitude],
  );
  const maxFrames = Math.max(1, ...arcs.map((a) => a.frames));
  const [hovered, setHovered] = useState<string | null>(null);
  const ha = arcs.find((a) => a.id === hovered);

  return (
    <div className="relative">
      {ha && (
        <div className="pointer-events-none absolute left-1/2 top-2 z-10 w-64 -translate-x-1/2 rounded-sm border border-border bg-surface-1 p-2 shadow-sm">
          <div className="font-mono text-[10px] uppercase tracking-wider text-subtle">
            sun-path arc · provenance
          </div>
          <div className="mt-0.5 truncate text-sm font-medium">{ha.label}</div>
          <div className="mt-1 grid grid-cols-2 gap-x-3 font-mono text-[11px] tabular-nums text-muted">
            <span>{ha.frames.toLocaleString("en-GB")} frames</span>
            <span>mean luma {ha.luminance ?? "?"}</span>
            <span>elev {Math.min(...ha.path.map((p) => p[0])).toFixed(0)}–{Math.max(...ha.path.map((p) => p[0])).toFixed(0)}°</span>
            <span>az {Math.min(...ha.path.map((p) => p[1])).toFixed(0)}–{Math.max(...ha.path.map((p) => p[1])).toFixed(0)}°</span>
          </div>
        </div>
      )}
    <svg
      viewBox={`0 0 ${W} ${H + NIGHT_H + 26}`}
      className="w-full"
      role="img"
      aria-label="Sky dome showing captured sun-path coverage"
    >
      <defs>
        <radialGradient id="skygrad" cx="50%" cy="92%" r="85%">
          <stop offset="0%" stopColor="var(--seeblau-100)" stopOpacity="0.35" />
          <stop offset="55%" stopColor="var(--seeblau-65)" stopOpacity="0.22" />
          <stop offset="100%" stopColor="var(--seeblau-20)" stopOpacity="0.30" />
        </radialGradient>
        <clipPath id="domeclip">
          <path d={`M${CX - R},${CY} A${R},${R} 0 0 1 ${CX + R},${CY} Z`} />
        </clipPath>
      </defs>

      {/* dome */}
      <path
        d={`M${CX - R},${CY} A${R},${R} 0 0 1 ${CX + R},${CY} Z`}
        fill="url(#skygrad)"
        stroke="var(--border-strong)"
        strokeWidth={1.5}
      />
      {/* elevation gridlines + labels (horizontal, clipped to the dome) */}
      <g clipPath="url(#domeclip)">
        {ELEV_RINGS.map((e) => {
          const [, y] = project(e, 180);
          return (
            <line key={e} x1={CX - R} y1={y} x2={CX + R} y2={y}
                  stroke="var(--border)" strokeWidth={1} strokeDasharray="2 5" />
          );
        })}
      </g>
      {ELEV_RINGS.map((e) => {
        const [, y] = project(e, 180);
        // label just inside the dome edge at that height
        const half = Math.sqrt(Math.max(R * R - (CY - y) * (CY - y), 0));
        return (
          <text key={e} x={CX - half + 8} y={y - 3} fontSize={10}
                fontFamily="var(--font-mono)" fill="var(--subtle)">
            {e}°
          </text>
        );
      })}
      {/* azimuth ticks */}
      {AZ_TICKS.map(([az, label]) => {
        const [x0] = project(0, az);
        return (
          <g key={az}>
            <line x1={x0} y1={CY} x2={x0} y2={CY - 7}
                  stroke="var(--border-strong)" strokeWidth={1.5} />
            <text x={x0} y={CY + 16} textAnchor="middle" fontSize={12}
                  fontFamily="var(--font-mono)" fill="var(--muted)">
              {label}
            </text>
          </g>
        );
      })}
      {/* seasonal envelope */}
      <g clipPath="url(#domeclip)">
        <path d={pathD(envelope.summer)} fill="none" stroke="var(--viz-water-edge)"
              strokeWidth={1.2} strokeDasharray="6 5" opacity={0.55} />
        <path d={pathD(envelope.winter)} fill="none" stroke="var(--viz-water-edge)"
              strokeWidth={1.2} strokeDasharray="6 5" opacity={0.55} />
        {/* captured sun arcs */}
        {arcs.map((arc) => {
          const width = 2 + 5 * Math.sqrt(arc.frames / maxFrames);
          const above = arc.path.filter(([e]) => e >= 0);
          if (above.length < 2) return null;
          const [ex, ey] = project(above[above.length - 1]![0], above[above.length - 1]![1]);
          return (
            <g key={arc.id}>
              <path
                d={pathD(above)}
                fill="none"
                stroke="var(--seeblau-deep)"
                strokeWidth={width + 4}
                strokeLinecap="round"
                opacity={0}
                className="cursor-pointer"
                onMouseEnter={() => setHovered(arc.id)}
                onMouseLeave={() => setHovered(null)}
              />
              <path
                d={pathD(above)}
                fill="none"
                stroke="var(--seeblau-deep)"
                strokeWidth={width}
                strokeLinecap="round"
                opacity={hovered && hovered !== arc.id ? 0.35 : 0.85}
                pointerEvents="none"
              />
              <circle cx={ex} cy={ey} r={width / 2 + 2.5}
                      fill="var(--viz-water-edge)" stroke="var(--background)" strokeWidth={1.5}>
                <title>{`${arc.label} — end of capture`}</title>
              </circle>
            </g>
          );
        })}
      </g>
      {/* envelope labels at each curve's apex */}
      <text x={CX} y={project(90 - latitude + 23.44, 180)[1] - 8}
            textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--viz-water-edge)">
        summer solstice
      </text>
      <text x={CX} y={project(90 - latitude - 23.44, 180)[1] - 8}
            textAnchor="middle" fontSize={10} fontFamily="var(--font-mono)" fill="var(--viz-water-edge)">
        winter solstice
      </text>
      {/* horizon + night band */}
      <line x1={CX - R} y1={CY} x2={CX + R} y2={CY}
            stroke="var(--border-strong)" strokeWidth={2} />
      <rect x={CX - R} y={CY + 22} width={2 * R} height={NIGHT_H}
            rx={4} fill="var(--seeblau-navy)" opacity={0.9} />
      <text x={CX} y={CY + 22 + NIGHT_H / 2 + 4} textAnchor="middle"
            fontSize={12} fontFamily="var(--font-mono)"
            fill={nightFrames > 0 ? "#ffffff" : "var(--viz-water-edge)"}>
        {nightFrames > 0
          ? `night · ${nightFrames} frames`
          : "night · no data yet"}
      </text>
    </svg>
    </div>
  );
}

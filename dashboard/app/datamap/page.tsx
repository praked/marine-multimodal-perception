"use client";

import { CoverageRing } from "@/components/datamap/CoverageRing";
import { SkyDome } from "@/components/datamap/SkyDome";
import { SpectrumStrip } from "@/components/datamap/SpectrumStrip";
import { TargetArcs } from "@/components/datamap/TargetArcs";
import { CrossMatrix } from "@/components/datamap/CrossMatrix";
import { GoalsEditor } from "@/components/datamap/GoalsEditor";
import {
  coverageGaps,
  degToCompass,
  daypartTotals,
  enriched,
  entityCoverage,
  hourlyDensity,
  luminanceBins,
  luminanceHist,
  sunArcs,
  weatherBands,
  goalGaps,
  entityConditionMatrix,
} from "@/lib/datamap";
import {
  defaultGoals,
  framesToHours,
  getGoalsBackend,
  goalsForBands,
  WEATHER_BAND_DEFS,
  type CoverageGoals,
} from "@/lib/goals";
import { listCuratedClips } from "@/lib/data";
import { colourForClass } from "@/lib/palette";
import type { ClipSummary } from "@/lib/types";
import {
  AlertTriangle,
  Cloud,
  CloudRain,
  ChevronDown,
  OctagonAlert,
  Pencil,
  Thermometer,
  Wind,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import type { Gap } from "@/lib/datamap";

/** "Open gaps" KPI with a click-open dropdown listing every gap succinctly,
    so the headline number is inspectable without scrolling to the bottom. */
function GapsKpi({ gaps }: { gaps: Gap[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="relative bg-surface-1">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full px-4 py-3 text-left hover:bg-surface-2"
        aria-expanded={open}
      >
        <div className="flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
          open gaps
          <ChevronDown
            size={12}
            aria-hidden
            className={`transition-transform ${open ? "rotate-180" : ""}`}
          />
        </div>
        <div className="mt-1 font-mono text-xl tabular-nums tracking-tight">
          {gaps.length}
        </div>
      </button>
      {open && (
        <div className="absolute right-0 top-full z-20 mt-1 max-h-80 w-80 overflow-y-auto rounded-sm border border-border bg-surface-1 p-2 shadow-sm">
          <div className="mb-1 font-mono text-[10px] uppercase tracking-wider text-subtle">
            coverage below goals
          </div>
          {gaps.length === 0 ? (
            <div className="text-[11px] text-status-good">
              no gaps against the current goals
            </div>
          ) : (
            <ul className="space-y-1">
              {gaps.map((g, i) => (
                <li key={i} className="flex items-start gap-1.5 text-[11px] leading-snug">
                  {g.severity === "serious" ? (
                    <OctagonAlert size={11} className="mt-0.5 shrink-0 text-status-serious" aria-hidden />
                  ) : (
                    <AlertTriangle size={11} className="mt-0.5 shrink-0 text-status-warn" aria-hidden />
                  )}
                  <span className={g.severity === "serious" ? "" : "text-muted"}>{g.text}</span>
                </li>
              ))}
            </ul>
          )}
          <div className="mt-2 border-t border-border pt-1.5 text-[10px] leading-snug text-subtle">
            gap = a dimension below its editable goal: entity class under its
            instance target, clock hour or weather band under its hours goal,
            missing night/twilight, or an empty dark-luminance quarter
          </div>
        </div>
      )}
    </div>
  );
}

function Kpi({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="bg-surface-1 px-4 py-3" title={hint}>
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
        {label}
      </div>
      <div className="mt-1 font-mono text-xl tabular-nums tracking-tight">
        {value}
      </div>
    </div>
  );
}

export default function DataMapPage() {
  const [clips, setClips] = useState<ClipSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [goals, setGoals] = useState<CoverageGoals>(defaultGoals);
  const [editing, setEditing] = useState(false);
  const [advanced, setAdvanced] = useState(false);
  const [draft, setDraft] = useState<CoverageGoals | null>(null);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "failed">("idle");

  useEffect(() => {
    // deleted sets do not count toward coverage
    listCuratedClips()
      .then(({ active }) => setClips(active))
      .catch((e: unknown) => setError(String(e)));
    getGoalsBackend()
      .load()
      .then(setGoals)
      .catch(() => {}); // defaults stand if the shared row is unreachable
  }, []);

  const saveGoals = async () => {
    if (!draft) return;
    setSaveState("saving");
    try {
      await getGoalsBackend().save(draft);
      setGoals(draft);
      setEditing(false);
      setDraft(null);
      setSaveState("saved");
      setTimeout(() => setSaveState("idle"), 2500);
    } catch {
      setSaveState("failed");
    }
  };

  if (error) {
    return (
      <div className="p-8 font-mono text-sm text-status-serious">{error}</div>
    );
  }
  if (!clips) {
    return <div className="p-8 font-mono text-sm text-subtle">loading…</div>;
  }

  const withEnrichment = enriched(clips);
  if (withEnrichment.length === 0) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-12 text-sm text-muted">
        No enrichment data in this catalogue yet — run
        <code className="mx-1 rounded-sm bg-surface-3 px-1 font-mono text-xs">
          dashboard/tools/enrich_bundle.py
        </code>
        after baking, then re-ingest.
      </div>
    );
  }

  const arcs = sunArcs(clips);
  const parts = daypartTotals(clips);
  const hours = hourlyDensity(clips);
  const entities = entityCoverage(clips, goals.entities);
  const gaps = [
    ...coverageGaps(clips, entities, { weatherHeuristics: false }),
    ...goalGaps(clips, goals, WEATHER_BAND_DEFS, framesToHours),
  ];
  const totalFrames = clips.reduce((a, c) => a + c.n_frames, 0);
  const days = new Set(withEnrichment.map((c) => c.triplet_ts.slice(0, 10)));
  const lat = withEnrichment[0]!.enrichment.gps_init.lat;
  const lon = withEnrichment[0]!.enrichment.gps_init.lon;

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <div className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
        Sensor-fusion training corpus · {lat.toFixed(4)}°N {lon.toFixed(4)}°E
      </div>
      <div className="mt-1 flex items-center justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">
          Data coverage map
        </h1>
        <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setAdvanced((a) => !a)}
          className={`rounded-sm border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors ${advanced ? "border-seeblau-100 bg-seeblau-100/10 text-seeblau-100" : "border-border text-muted hover:border-border-strong hover:text-foreground"}`}
        >
          {advanced ? "basic mode" : "advanced mode"}
        </button>
        <button
          type="button"
          onClick={() => {
            if (editing) {
              setEditing(false);
              setDraft(null);
            } else {
              setDraft(goals);
              setEditing(true);
            }
          }}
          className={`flex items-center gap-1.5 rounded-sm border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors ${editing ? "border-seeblau-100 bg-seeblau-100/10 text-seeblau-100" : "border-border text-muted hover:border-border-strong hover:text-foreground"}`}
        >
          {editing ? <X size={12} aria-hidden /> : <Pencil size={12} aria-hidden />}
          {editing ? "close editor" : "edit goals"}
        </button>
        </div>
      </div>
      {saveState === "saved" && (
        <div className="mt-2 font-mono text-[11px] text-status-good">
          goals saved{getGoalsBackend().mode === "supabase" ? " — shared with the team" : " locally (demo mode)"}
        </div>
      )}
      {editing && draft && (
        <section className="mt-4 rounded-sm border border-seeblau-100/40 bg-surface-1 p-5">
          <div className="mb-4 flex items-baseline justify-between">
            <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
              Coverage goals — what the corpus should contain
            </h2>
            <span className="font-mono text-[10px] text-subtle">
              drives ring outlines, entity gauges and the gaps list
            </span>
          </div>
          <GoalsEditor goals={draft} onChange={setDraft} />
          <div className="mt-4 flex items-center gap-2 border-t border-border pt-3">
            <button
              type="button"
              onClick={saveGoals}
              disabled={saveState === "saving"}
              className="rounded-sm bg-seeblau-100 px-4 py-1.5 font-mono text-[11px] uppercase tracking-wider text-inverse transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {saveState === "saving" ? "saving…" : "save goals"}
            </button>
            <button
              type="button"
              onClick={() => setDraft(defaultGoals())}
              className="rounded-sm border border-border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider text-muted hover:text-foreground"
            >
              reset to defaults
            </button>
            {saveState === "failed" && (
              <span className="font-mono text-[11px] text-status-serious">
                save failed — check the connection and retry
              </span>
            )}
            <span className="ml-auto font-mono text-[10px] text-subtle">
              {getGoalsBackend().mode === "supabase" ? "stored in Supabase · shared" : "stored in this browser (demo)"}
            </span>
          </div>
        </section>
      )}
      <p className="mt-2 max-w-2xl text-sm text-muted">
        What the dataset has seen — and what it hasn&apos;t. Sun geometry,
        light, weather and detected entities, mapped against the conditions
        the fusion scorer must eventually handle.
      </p>

      <div className="mt-6 grid grid-cols-2 gap-px rounded-sm border border-border bg-border sm:grid-cols-4">
        <Kpi label="capture days" value={String(days.size)} />
        <Kpi label="frames" value={totalFrames.toLocaleString("en-GB")} />
        <Kpi
          label="daylight / night"
          value={`${(((parts.day ?? 0) + (parts.golden ?? 0)) / totalFrames * 100).toFixed(0)}% / ${(((parts.night ?? 0) / totalFrames) * 100).toFixed(0)}%`}
        />
        <GapsKpi gaps={gaps} />
      </div>

      {/* HERO: sky dome */}
      <section className="mt-8 rounded-sm border border-border bg-surface-1 p-5">
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
            Sun-position coverage
          </h2>
          <span className="font-mono text-[10px] text-subtle">
            arc width ∝ frames · dashed = solstice envelope · ● capture end
          </span>
        </div>
        <SkyDome arcs={arcs} latitude={lat} nightFrames={parts.night ?? 0} />
      </section>

      {/* rings + spectrum */}
      <section className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="rounded-sm border border-border bg-surface-1 p-4">
          <h2 className="mb-3 font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
            Time of day
          </h2>
          <div className="flex justify-center">
            <CoverageRing
              title="24-hour clock"
              unit="frames per hour"
              domain={[0, 24]}
              tickEvery={6}
              bands={hours.map((h) => ({
                from: h.hour,
                to: h.hour + 1,
                value: h.frames,
                clips: h.clips,
              }))}
              goals={hours.map((h) => goals.hours[String(h.hour)] ?? 0)}
              framesToHours={framesToHours}
              format={(v) => `${v}h`}
            />
          </div>
        </div>
        <div className="rounded-sm border border-border bg-surface-1 p-4 lg:col-span-2">
          <h2 className="mb-3 font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
            Weather envelope
          </h2>
          <div className="flex flex-wrap justify-around gap-2">
            <CoverageRing
              title="Cloud cover"
              unit="% sky"
              domain={[0, 100]}
              tickEvery={25}
              startDeg={-120}
              sweepDeg={240}
              goals={goalsForBands(goals.weather.cloud_cover_pct, WEATHER_BAND_DEFS.cloud_cover_pct.edges)}
              framesToHours={framesToHours}
              bands={weatherBands(clips, "cloud_cover_pct", [0, 20, 40, 60, 80, 100.1])}
            />
            <CoverageRing
              title="Wind"
              unit="km/h"
              domain={[0, 40]}
              tickEvery={10}
              startDeg={-120}
              sweepDeg={240}
              goals={goalsForBands(goals.weather.wind_speed_kmh, WEATHER_BAND_DEFS.wind_speed_kmh.edges)}
              framesToHours={framesToHours}
              bands={weatherBands(clips, "wind_speed_kmh", [0, 10, 20, 30, 40.1])}
            />
            <CoverageRing
              title="Precipitation"
              unit="mm/h"
              domain={[0, 4]}
              tickEvery={1}
              startDeg={-120}
              sweepDeg={240}
              goals={goalsForBands(goals.weather.precipitation_mm, WEATHER_BAND_DEFS.precipitation_mm.edges)}
              framesToHours={framesToHours}
              bands={weatherBands(clips, "precipitation_mm", [0, 0.1, 1, 2, 4.1])}
            />
            <CoverageRing
              title="Wind direction"
              unit="compass (from)"
              domain={[0, 360]}
              tickEvery={90}
              goals={goalsForBands(goals.weather.wind_dir_deg, WEATHER_BAND_DEFS.wind_dir_deg.edges)}
              framesToHours={framesToHours}
              bands={weatherBands(clips, "wind_dir_deg", [0, 45, 90, 135, 180, 225, 270, 315, 360.1])}
              format={(v) => degToCompass(Math.min(v, 360) % 360)}
            />
          </div>
        </div>
      </section>

      <section className="mt-4 rounded-sm border border-border bg-surface-1 p-4">
        <SpectrumStrip hist={luminanceHist(clips)} bins={luminanceBins(clips)} />
      </section>

      {/* per-capture-day weather (Open-Meteo at the GPS position + hours) */}
      <section className="mt-4 rounded-sm border border-border bg-surface-1 p-4">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
            Capture-day weather
          </h2>
          <span className="font-mono text-[10px] text-subtle">
            open-meteo · localised to {lat.toFixed(3)}°N {lon.toFixed(3)}°E at capture hours
          </span>
        </div>
        <div className="grid grid-cols-1 gap-px overflow-hidden rounded-sm border border-border bg-border sm:grid-cols-2 lg:grid-cols-4">
          {[...days].sort().map((day) => {
            const dayClips = withEnrichment.filter((c) =>
              c.triplet_ts.startsWith(day),
            );
            const w = dayClips.find((c) => c.enrichment.weather)?.enrichment
              .weather;
            const frames = dayClips.reduce((a, c) => a + c.n_frames, 0);
            const [y, mo, d] = day.split("-");
            return (
              <div key={day} className="bg-surface-1 p-3">
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-sm tabular-nums font-medium">
                    {d}-{mo}-{y}
                  </span>
                  <span className="font-mono text-[10px] tabular-nums text-subtle">
                    {dayClips.length} clip{dayClips.length > 1 ? "s" : ""} ·{" "}
                    {frames.toLocaleString("en-GB")}f
                  </span>
                </div>
                {w ? (
                  <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-[11px] tabular-nums text-muted">
                    <span className="flex items-center gap-1">
                      <Cloud size={12} aria-hidden /> {w.cloud_cover_pct}%
                    </span>
                    <span className="flex items-center gap-1">
                      <Wind size={12} aria-hidden /> {w.wind_speed_kmh} km/h
                      {w.wind_dir_deg != null && (
                        <span className="text-subtle">
                          {" "}{degToCompass(w.wind_dir_deg)} {w.wind_dir_deg}°
                        </span>
                      )}
                    </span>
                    <span className="flex items-center gap-1">
                      <Thermometer size={12} aria-hidden /> {w.temperature_c}°C
                    </span>
                    <span className="flex items-center gap-1">
                      <CloudRain size={12} aria-hidden /> {w.precipitation_mm} mm
                    </span>
                  </div>
                ) : (
                  <div className="mt-2 font-mono text-[11px] text-subtle">
                    weather unknown
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* entities */}
      <section className="mt-4 rounded-sm border border-border bg-surface-1 p-4">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
            Detected entities vs dataset targets
          </h2>
          <span className="font-mono text-[10px] text-subtle">
            from annotation + segmentation passes · targets editable via edit goals
          </span>
        </div>
        <TargetArcs items={entities} colour={colourForClass} />
      </section>

      {/* advanced mode: entity x condition intersection */}
      {advanced && (
        <section className="mt-4 rounded-sm border border-seeblau-100/40 bg-surface-1 p-4">
          <div className="mb-3 flex items-baseline justify-between">
            <h2 className="font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
              Entity × condition intersections
            </h2>
            <span className="font-mono text-[10px] text-subtle">
              which classes were captured under which conditions — amber = deployment-risk gap
            </span>
          </div>
          <CrossMatrix
            matrix={entityConditionMatrix(clips, goals.entities)}
            colour={colourForClass}
          />
        </section>
      )}

      {/* gaps */}
      <section className="mt-4 rounded-sm border border-border bg-surface-1 p-4">
        <h2 className="mb-1 font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
          Open gaps — what to capture next
        </h2>
        <p className="mb-3 text-[11px] leading-snug text-subtle">
          A gap is any coverage dimension below its goal (edit goals, top
          right): an entity class under its instance target, a clock hour or
          weather band with fewer captured hours than its goal, no night or
          twilight footage, or no frames in the darkest luminance quarter.
          Serious (red) = nothing captured at all; warning (amber) = partial.
        </p>
        {gaps.length === 0 ? (
          <p className="text-sm text-status-good">No gaps against the current targets.</p>
        ) : (
          <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
            {gaps.map((g, i) => (
              <li key={i} className="flex items-center gap-2 text-sm">
                {g.severity === "serious" ? (
                  <OctagonAlert size={15} className="shrink-0 text-status-serious" aria-hidden />
                ) : (
                  <AlertTriangle size={15} className="shrink-0 text-status-warn" aria-hidden />
                )}
                <span className={g.severity === "serious" ? "font-medium" : "text-muted"}>
                  {g.text}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

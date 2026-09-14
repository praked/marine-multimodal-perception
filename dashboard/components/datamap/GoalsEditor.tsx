"use client";

import {
  WEATHER_BAND_DEFS,
  type CoverageGoals,
  type WeatherGoalKey,
} from "@/lib/goals";
import { degToCompass } from "@/lib/datamap";

/* Edit mode for the coverage map: set the capture targets the visuals and
   the gaps list measure against — instances per entity class, hours of
   footage per clock hour, hours per weather band. Pure controlled
   component; persistence lives in the page (lib/goals backend). */

function NumInput({
  value,
  onChange,
  step = 0.5,
  width = "w-16",
}: {
  value: number;
  onChange: (v: number) => void;
  step?: number;
  width?: string;
}) {
  return (
    <input
      type="number"
      min={0}
      step={step}
      value={value}
      onChange={(e) => {
        const v = Number(e.target.value);
        onChange(Number.isFinite(v) && v >= 0 ? v : 0);
      }}
      className={`${width} rounded-sm border border-border bg-surface-1 px-1.5 py-0.5 text-right font-mono text-[11px] tabular-nums focus:border-seeblau-100 focus:outline-none`}
    />
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.28em] text-muted">
      {children}
    </div>
  );
}

export function GoalsEditor({
  goals,
  onChange,
}: {
  goals: CoverageGoals;
  onChange: (g: CoverageGoals) => void;
}) {
  const setEntity = (cls: string, v: number) =>
    onChange({ ...goals, entities: { ...goals.entities, [cls]: v } });
  const setHour = (h: number, v: number) =>
    onChange({ ...goals, hours: { ...goals.hours, [String(h)]: v } });
  const setWeather = (key: WeatherGoalKey, from: number, v: number) =>
    onChange({
      ...goals,
      weather: {
        ...goals.weather,
        [key]: { ...goals.weather[key], [String(from)]: v },
      },
    });

  return (
    <div className="space-y-5">
      <div>
        <SectionLabel>Entity instances · target count per class</SectionLabel>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1.5 sm:grid-cols-4">
          {Object.entries(goals.entities).map(([cls, v]) => (
            <label key={cls} className="flex items-center justify-between gap-2 text-xs">
              <span className="text-muted">{cls}</span>
              <NumInput value={v} step={50} onChange={(n) => setEntity(cls, n)} />
            </label>
          ))}
        </div>
      </div>

      <div>
        <SectionLabel>Time of day · hours of footage per clock hour</SectionLabel>
        <div className="grid grid-cols-6 gap-1.5 sm:grid-cols-12">
          {Array.from({ length: 24 }, (_, h) => (
            <label key={h} className="flex flex-col items-center gap-0.5">
              <span className="font-mono text-[9px] tabular-nums text-subtle">
                {String(h).padStart(2, "0")}h
              </span>
              <NumInput
                value={goals.hours[String(h)] ?? 0}
                width="w-full"
                onChange={(n) => setHour(h, n)}
              />
            </label>
          ))}
        </div>
      </div>

      {(Object.keys(WEATHER_BAND_DEFS) as WeatherGoalKey[]).map((key) => {
        const def = WEATHER_BAND_DEFS[key];
        const edges = def.edges;
        return (
          <div key={key}>
            <SectionLabel>
              {def.label} · hours per band ({def.unit})
            </SectionLabel>
            <div className="flex flex-wrap gap-x-5 gap-y-1.5">
              {edges.slice(0, -1).map((from, i) => (
                <label key={from} className="flex items-center gap-2 text-xs">
                  <span className="font-mono text-[10px] tabular-nums text-muted">
                    {key === "wind_dir_deg"
                      ? `${degToCompass(from)}–${degToCompass(Math.min(edges[i + 1]!, 360) % 360)}`
                      : `${from}–${Math.round(edges[i + 1]!)}`}
                  </span>
                  <NumInput
                    value={goals.weather[key][String(from)] ?? 0}
                    onChange={(n) => setWeather(key, from, n)}
                  />
                </label>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

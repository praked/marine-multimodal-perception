import { isSupabaseConfigured } from "@/lib/data/provider";
import { createClient } from "@/lib/supabase/client";
import { ENTITY_TARGETS } from "@/lib/datamap";
import { z } from "zod";

/* Editable coverage goals for the data map: what the corpus SHOULD contain
   — instance targets per entity class, hours of footage per clock hour, and
   hours per weather band. Goals are a team-level planning artefact, so the
   primary store is one shared Supabase row (sail_goals, id='default');
   demo mode falls back to localStorage. Same seam as lib/audit/backend. */

export const CAPTURE_FPS = 3;
export const framesToHours = (frames: number) => frames / (CAPTURE_FPS * 3600);

/** Band edges shared by the rings, the goal editor and the gap analysis —
    one definition so a goal always refers to a rendered band. */
export const WEATHER_BAND_DEFS = {
  cloud_cover_pct: {
    label: "Cloud cover",
    unit: "% sky",
    edges: [0, 20, 40, 60, 80, 100.1],
  },
  wind_speed_kmh: {
    label: "Wind",
    unit: "km/h",
    edges: [0, 10, 20, 30, 40.1],
  },
  precipitation_mm: {
    label: "Precipitation",
    unit: "mm/h",
    edges: [0, 0.1, 1, 2, 4.1],
  },
  wind_dir_deg: {
    label: "Wind direction",
    unit: "compass (from)",
    edges: [0, 45, 90, 135, 180, 225, 270, 315, 360.1],
  },
} as const;

export type WeatherGoalKey = keyof typeof WEATHER_BAND_DEFS;

const numRecord = z.record(z.string(), z.number().min(0).finite());

export const CoverageGoalsSchema = z.object({
  version: z.literal(1),
  /** target instances per canonical entity class */
  entities: numRecord,
  /** target hours of footage per clock hour, keyed "0".."23" */
  hours: numRecord,
  /** target hours per weather band, keyed by the band's `from` edge */
  weather: z.object({
    cloud_cover_pct: numRecord,
    wind_speed_kmh: numRecord,
    precipitation_mm: numRecord,
    wind_dir_deg: numRecord,
  }),
});

export type CoverageGoals = z.infer<typeof CoverageGoalsSchema>;

function bandDefaults(key: WeatherGoalKey, hours: number[]): Record<string, number> {
  const edges = WEATHER_BAND_DEFS[key].edges;
  const out: Record<string, number> = {};
  for (let i = 0; i < edges.length - 1; i++) {
    out[String(edges[i])] = hours[i] ?? hours[hours.length - 1] ?? 1;
  }
  return out;
}

/** Mission-shaped starting point: daylight-heavy hour targets, a spread of
    weather bands with rain/strong-wind kept realistic. All editable. */
export function defaultGoals(): CoverageGoals {
  const hours: Record<string, number> = {};
  for (let h = 0; h < 24; h++) {
    hours[String(h)] = h >= 6 && h <= 21 ? 2 : 1;
  }
  return {
    version: 1,
    entities: { ...ENTITY_TARGETS },
    hours,
    weather: {
      cloud_cover_pct: bandDefaults("cloud_cover_pct", [2, 2, 2, 2, 2]),
      wind_speed_kmh: bandDefaults("wind_speed_kmh", [3, 3, 2, 1]),
      precipitation_mm: bandDefaults("precipitation_mm", [6, 2, 1, 0.5]),
      wind_dir_deg: bandDefaults("wind_dir_deg", [1, 1, 1, 1, 1, 1, 1, 1]),
    },
  };
}

/** Merge a stored (possibly partial/older) goals object over the defaults so
    new dimensions always have values. */
export function mergeGoals(stored: unknown): CoverageGoals {
  const base = defaultGoals();
  const parsed = CoverageGoalsSchema.safeParse(stored);
  if (!parsed.success) return base;
  const g = parsed.data;
  return {
    version: 1,
    entities: { ...base.entities, ...g.entities },
    hours: { ...base.hours, ...g.hours },
    weather: {
      cloud_cover_pct: { ...base.weather.cloud_cover_pct, ...g.weather.cloud_cover_pct },
      wind_speed_kmh: { ...base.weather.wind_speed_kmh, ...g.weather.wind_speed_kmh },
      precipitation_mm: { ...base.weather.precipitation_mm, ...g.weather.precipitation_mm },
      wind_dir_deg: { ...base.weather.wind_dir_deg, ...g.weather.wind_dir_deg },
    },
  };
}

export interface GoalsBackend {
  readonly mode: "local" | "supabase";
  load(): Promise<CoverageGoals>;
  save(goals: CoverageGoals): Promise<void>;
}

const LS_KEY = "asvproject.goals.v1";

export function createLocalGoalsBackend(): GoalsBackend {
  return {
    mode: "local",
    async load() {
      if (typeof window === "undefined") return defaultGoals();
      try {
        const raw = window.localStorage.getItem(LS_KEY);
        return mergeGoals(raw ? JSON.parse(raw) : null);
      } catch {
        return defaultGoals();
      }
    },
    async save(goals: CoverageGoals) {
      window.localStorage.setItem(LS_KEY, JSON.stringify(goals));
    },
  };
}

export function createSupabaseGoalsBackend(): GoalsBackend {
  const supabase = createClient();
  return {
    mode: "supabase",
    async load() {
      const { data, error } = await supabase
        .from("sail_goals")
        .select("goals")
        .eq("id", "default")
        .maybeSingle();
      if (error) throw error;
      return mergeGoals(data?.goals ?? null);
    },
    async save(goals: CoverageGoals) {
      const { error } = await supabase
        .from("sail_goals")
        .upsert({ id: "default", goals, updated_at: new Date().toISOString() });
      if (error) throw error;
    },
  };
}

let cached: GoalsBackend | null = null;

export function getGoalsBackend(): GoalsBackend {
  if (!cached) {
    cached = isSupabaseConfigured()
      ? createSupabaseGoalsBackend()
      : createLocalGoalsBackend();
  }
  return cached;
}

/** Goal hours aligned with a ring's bands, from the keyed goal record. */
export function goalsForBands(
  goals: Record<string, number>,
  edges: readonly number[],
): number[] {
  const out: number[] = [];
  for (let i = 0; i < edges.length - 1; i++) {
    out.push(goals[String(edges[i])] ?? 0);
  }
  return out;
}

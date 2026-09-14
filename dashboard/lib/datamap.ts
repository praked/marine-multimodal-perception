import type { ClipSummary, Enrichment } from "@/lib/types";

/* Pure aggregation for the data-completion map. Everything here consumes
   the generic Enrichment shape and produces render-ready structures, so the
   map components stay portable to other projects: swap the adapter, keep
   the visuals. */

export interface SunArc {
  id: string;
  label: string;
  /** [elevation_deg, azimuth_deg] along the clip */
  path: [number, number][];
  frames: number;
  luminance: number | null;
}

export interface HourDensity {
  hour: number;
  frames: number;
  clips: SegmentContribution[];
}

export interface Band {
  from: number; // domain units
  to: number;
  value: number; // density (frames)
  clips: SegmentContribution[];
}

export const DAYPART_ORDER = ["night", "twilight", "golden", "day"] as const;

const COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S",
  "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"] as const;

/** 242 -> "WSW" (16-point compass). */
export function degToCompass(deg: number): string {
  return COMPASS[Math.round(((deg % 360) + 360) % 360 / 22.5) % 16]!;
}

export function enriched(
  clips: ClipSummary[],
): (ClipSummary & { enrichment: Enrichment })[] {
  return clips.filter(
    (c): c is ClipSummary & { enrichment: Enrichment } => !!c.enrichment,
  );
}

export function sunArcs(clips: ClipSummary[]): SunArc[] {
  return enriched(clips).map((c) => ({
    id: c.clip_id,
    label: c.title,
    path: c.enrichment.sun_samples.map(
      ([, elev, az]) => [elev, az] as [number, number],
    ),
    frames: c.n_frames,
    luminance: c.enrichment.luminance.mean,
  }));
}

/** Frames per hour of day (0..23), from sun samples' timestamps weighted by
    each clip's frame count spread over its samples. */
export function hourlyDensity(clips: ClipSummary[]): HourDensity[] {
  const hours = Array.from({ length: 24 }, () => new Map<string, number>());
  const titles = new Map<string, string>();
  for (const c of enriched(clips)) {
    const samples = c.enrichment.sun_samples;
    if (samples.length === 0) continue;
    titles.set(c.clip_id, c.title);
    const perSample = c.n_frames / samples.length;
    for (const [ts] of samples) {
      const h = hours[Number(ts.slice(0, 2)) % 24]!;
      h.set(c.clip_id, (h.get(c.clip_id) ?? 0) + perSample);
    }
  }
  return hours.map((m, hour) => {
    const clips2 = [...m.entries()]
      .map(([clipId, v]) => ({
        clipId,
        title: titles.get(clipId) ?? clipId,
        value: Math.round(v),
      }))
      .filter((c) => c.value > 0)
      .sort((a, b) => b.value - a.value);
    return {
      hour,
      frames: clips2.reduce((a, c) => a + c.value, 0),
      clips: clips2,
    };
  });
}

export function daypartTotals(clips: ClipSummary[]): Record<string, number> {
  const totals: Record<string, number> = {};
  for (const part of DAYPART_ORDER) totals[part] = 0;
  for (const c of enriched(clips)) {
    for (const [part, n] of Object.entries(c.enrichment.dayparts)) {
      totals[part] = (totals[part] ?? 0) + n;
    }
  }
  return totals;
}

/** Merge luminance histograms (equal bin counts assumed). */
export function luminanceHist(clips: ClipSummary[]): number[] {
  let out: number[] = [];
  for (const c of enriched(clips)) {
    const h = c.enrichment.luminance.hist;
    if (out.length === 0) out = new Array(h.length).fill(0);
    h.forEach((v, i) => (out[i] = (out[i] ?? 0) + v));
  }
  return out;
}

/** Coverage bands over a weather dimension: which value-ranges we have
    captured footage under, weighted by frames. */
export function weatherBands(
  clips: ClipSummary[],
  key:
    | "cloud_cover_pct"
    | "wind_speed_kmh"
    | "precipitation_mm"
    | "temperature_c"
    | "wind_dir_deg",
  binEdges: number[],
): Band[] {
  const bands: Band[] = [];
  for (let i = 0; i < binEdges.length - 1; i++) {
    bands.push({ from: binEdges[i]!, to: binEdges[i + 1]!, value: 0, clips: [] });
  }
  for (const c of enriched(clips)) {
    const w = c.enrichment.weather;
    if (!w) continue;
    const v = w[key];
    if (v == null) continue;
    const band =
      bands.find((b) => v >= b.from && v < b.to) ?? bands[bands.length - 1]!;
    band.value += c.n_frames;
    band.clips.push({ clipId: c.clip_id, title: c.title, value: c.n_frames });
  }
  for (const b of bands) b.clips.sort((a, x) => x.value - a.value);
  return bands;
}

/** Per-luminance-bin contributions. Uses each clip's mean luminance for
    attribution (the per-frame split is not in the summary), so the counts
    match the histogram while attribution is clip-granular. */
export function luminanceBins(clips: ClipSummary[]): Band[] {
  const hist = luminanceHist(clips);
  const bands: Band[] = hist.map((v, i) => ({
    from: (i * 256) / hist.length,
    to: ((i + 1) * 256) / hist.length,
    value: v,
    clips: [],
  }));
  for (const c of enriched(clips)) {
    const mean = c.enrichment.luminance.mean;
    if (mean == null || bands.length === 0) continue;
    const idx = Math.min(
      Math.floor((mean / 256) * bands.length),
      bands.length - 1,
    );
    bands[idx]!.clips.push({
      clipId: c.clip_id,
      title: c.title,
      value: c.n_frames,
    });
  }
  return bands;
}

export interface EntityContribution {
  clipId: string;
  title: string;
  instances: number;
  sources: Record<string, number>; // per detection-stream instance counts
}

export interface EntityCoverage {
  cls: string;
  instances: number;
  frames: number;
  target: number;
  contributions: EntityContribution[];
}

export interface SegmentContribution {
  clipId: string;
  title: string;
  value: number;
}

/** Detector taxonomies -> canonical dataset classes (LaRS typed classes and
    the label-tool vocabulary describe the same things). */
export const CLASS_CANON: Record<string, string> = {
  boat_ship: "boat",
  row_boats: "boat",
  swimmer: "person",
  paddle_board: "float",
};

export function entityCoverage(
  clips: ClipSummary[],
  targets: Record<string, number>,
): EntityCoverage[] {
  const acc: Record<
    string,
    { instances: number; frames: number; contributions: EntityContribution[] }
  > = {};
  for (const c of enriched(clips)) {
    // Fold WITHIN the clip first: canonical classes take the MAX across raw
    // classes/streams (different detectors saw the same physical objects);
    // only across clips do counts add.
    const perClip: Record<
      string,
      { instances: number; frames: number; sources: Record<string, number> }
    > = {};
    for (const [raw, e] of Object.entries(c.enrichment.entities)) {
      const cls = CLASS_CANON[raw] ?? raw;
      const p = (perClip[cls] ??= { instances: 0, frames: 0, sources: {} });
      if (e.instances > p.instances) {
        p.instances = e.instances;
        p.frames = e.frames;
      }
      for (const [stream, sv] of Object.entries(e.by_source ?? {})) {
        p.sources[stream] = (p.sources[stream] ?? 0) + sv.instances;
      }
    }
    for (const [cls, p] of Object.entries(perClip)) {
      const a = (acc[cls] ??= { instances: 0, frames: 0, contributions: [] });
      a.instances += p.instances;
      a.frames += p.frames;
      if (p.instances > 0) {
        a.contributions.push({
          clipId: c.clip_id,
          title: c.title,
          instances: p.instances,
          sources: p.sources,
        });
      }
    }
  }
  const classes = new Set([...Object.keys(targets), ...Object.keys(acc)]);
  return [...classes]
    .map((cls) => ({
      cls,
      instances: acc[cls]?.instances ?? 0,
      frames: acc[cls]?.frames ?? 0,
      target: targets[cls] ?? 0,
      contributions: (acc[cls]?.contributions ?? []).sort(
        (a, b) => b.instances - a.instances,
      ),
    }))
    .sort((a, b) => b.instances - a.instances);
}

export interface Gap {
  severity: "serious" | "warn";
  text: string;
}

/** Goal-driven shortfalls: hours-of-day + weather bands vs the editable
    goals. Hour shortfalls are summarised to one line (24 would drown the
    list); weather bands are listed individually — zero coverage against a
    goal is serious, partial is a warning. */
export function goalGaps(
  clips: ClipSummary[],
  goals: {
    hours: Record<string, number>;
    weather: Record<string, Record<string, number>>;
  },
  weatherDefs: Record<
    string,
    { label: string; unit: string; edges: readonly number[] }
  >,
  framesToHours: (frames: number) => number,
): Gap[] {
  const gaps: Gap[] = [];
  const byHour = hourlyDensity(clips);
  const short = byHour.filter(
    (h) => (goals.hours[String(h.hour)] ?? 0) > framesToHours(h.frames),
  );
  if (short.length > 0) {
    const worst = [...short].sort(
      (a, b) =>
        framesToHours(a.frames) - (goals.hours[String(a.hour)] ?? 0) -
        (framesToHours(b.frames) - (goals.hours[String(b.hour)] ?? 0)),
    )[0]!;
    gaps.push({
      severity: short.some((h) => h.frames === 0) ? "serious" : "warn",
      text: `${short.length} of 24 clock-hours below target (worst: ${String(worst.hour).padStart(2, "0")}h at ${framesToHours(worst.frames).toFixed(1)}/${goals.hours[String(worst.hour)]}h)`,
    });
  }
  for (const [key, def] of Object.entries(weatherDefs)) {
    const bands = weatherBands(
      clips,
      key as Parameters<typeof weatherBands>[1],
      [...def.edges],
    );
    const goalRec = goals.weather[key] ?? {};
    bands.forEach((b) => {
      const goal = goalRec[String(b.from)] ?? 0;
      const have = framesToHours(b.value);
      if (goal <= 0 || have >= goal) return;
      const range = `${def.label} ${b.from}–${Math.round(b.to)} ${def.unit}`;
      gaps.push(
        b.value === 0
          ? { severity: "serious", text: `${range}: nothing captured (goal ${goal}h)` }
          : { severity: "warn", text: `${range}: ${have.toFixed(1)}h of ${goal}h` },
      );
    });
  }
  return gaps;
}

export function coverageGaps(
  clips: ClipSummary[],
  entities: EntityCoverage[],
  opts: { weatherHeuristics?: boolean } = {},
): Gap[] {
  const gaps: Gap[] = [];
  const parts = daypartTotals(clips);
  if (!parts.night) {
    gaps.push({ severity: "serious", text: "No night footage at all" });
  }
  if (!parts.twilight) {
    gaps.push({ severity: "warn", text: "No twilight footage" });
  }
  const withWeather = enriched(clips).filter((c) => c.enrichment.weather);
  // Skipped when goal-driven weather gaps are shown instead (goalGaps).
  if (withWeather.length && (opts.weatherHeuristics ?? true)) {
    if (withWeather.every((c) => c.enrichment.weather!.precipitation_mm < 0.1)) {
      gaps.push({ severity: "serious", text: "No rain footage" });
    }
    if (withWeather.every((c) => c.enrichment.weather!.wind_speed_kmh < 20)) {
      gaps.push({ severity: "warn", text: "No strong-wind footage (≥20 km/h)" });
    }
    if (withWeather.every((c) => c.enrichment.weather!.cloud_cover_pct < 80)) {
      gaps.push({ severity: "warn", text: "No overcast footage (≥80% cloud)" });
    }
  }
  const hist = luminanceHist(clips);
  const dark = hist.slice(0, Math.floor(hist.length / 4)).reduce((a, b) => a + b, 0);
  if (hist.length && dark === 0) {
    gaps.push({ severity: "warn", text: "No low-luminance frames (<25% luma)" });
  }
  for (const e of entities) {
    if (e.target > 0 && e.instances === 0) {
      gaps.push({ severity: "serious", text: `No ${e.cls} instances yet` });
    } else if (e.target > 0 && e.instances < e.target / 2) {
      gaps.push({
        severity: "warn",
        text: `${e.cls}: ${e.instances} of ${e.target} target instances`,
      });
    }
  }
  return gaps;
}

/** Sensor-fusion targets before the real scorer training round (see
    docs/plans/fusion_scoring_training.md; swimmers/animals are priority
    thermal classes). Deliberately visible + editable in one place. */
export const ENTITY_TARGETS: Record<string, number> = {
  boat: 5000,
  person: 1500, // includes LaRS "swimmer" via CLASS_CANON
  duck: 1000,
  animal: 500,
  buoy: 500,
  structure: 2000,
  float: 300,
  other: 300,
};

/* ---- Advanced mode: entity x condition intersection matrix -------------- */

export interface MatrixCondition {
  key: string;          // stable id, e.g. "cloud:40"
  label: string;        // "40–60%"
  group: string;        // "Daypart" | "Cloud" | ...
  /** does this clip fall in the condition? */
  test: (c: ClipSummary & { enrichment: Enrichment }) => boolean;
}

export interface MatrixCell {
  instances: number;
  clips: SegmentContribution[];
  /** class detected somewhere in the corpus but never under this condition */
  gap: boolean;
}

export interface EntityConditionMatrix {
  classes: string[];
  conditions: MatrixCondition[];
  cells: Record<string, Record<string, MatrixCell>>; // cls -> cond.key
}

function band(v: number | null | undefined, edges: number[]): number {
  if (v == null) return -1;
  for (let i = 0; i < edges.length - 1; i++) {
    if (v >= edges[i]! && v < edges[i + 1]!) return i;
  }
  return edges.length - 2;
}

export function matrixConditions(): MatrixCondition[] {
  const conds: MatrixCondition[] = [];
  for (const part of DAYPART_ORDER) {
    conds.push({
      key: `part:${part}`, label: part, group: "Daypart",
      test: (c) => (c.enrichment.dayparts[part] ?? 0) > 0,
    });
  }
  const cloud = [0, 20, 40, 60, 80, 100.1];
  cloud.slice(0, -1).forEach((from, i) => conds.push({
    key: `cloud:${from}`, label: `${from}–${Math.round(cloud[i + 1]!)}%`,
    group: "Cloud",
    test: (c) => band(c.enrichment.weather?.cloud_cover_pct, cloud) === i,
  }));
  const wind = [0, 10, 20, 30, 40.1];
  wind.slice(0, -1).forEach((from, i) => conds.push({
    key: `wind:${from}`, label: `${from}–${Math.round(wind[i + 1]!)}`,
    group: "Wind km/h",
    test: (c) => band(c.enrichment.weather?.wind_speed_kmh, wind) === i,
  }));
  const precip = [0, 0.1, 1, 4.1];
  const plabels = ["dry", "drizzle", "rain"];
  precip.slice(0, -1).forEach((from, i) => conds.push({
    key: `precip:${from}`, label: plabels[i]!, group: "Precip",
    test: (c) => band(c.enrichment.weather?.precipitation_mm, precip) === i,
  }));
  const luma = [0, 85, 170, 256];
  const llabels = ["dark", "mid", "bright"];
  luma.slice(0, -1).forEach((from, i) => conds.push({
    key: `luma:${from}`, label: llabels[i]!, group: "Luminance",
    test: (c) => band(c.enrichment.luminance.mean, luma) === i,
  }));
  return conds;
}

/** Instances of each entity class captured under each condition. Reuses the
    within-clip max-across-streams fold (never sums detectors), then sums
    across clips that fall in the condition. A cell is a GAP when the class
    exists somewhere in the corpus but never under that condition — the
    "all our ducks were seen in the rain" deployment risk. */
export function entityConditionMatrix(
  clips: ClipSummary[],
  targets: Record<string, number>,
): EntityConditionMatrix {
  const conds = matrixConditions();
  const cov = entityCoverage(clips, targets);
  const classes = cov.filter((e) => e.instances > 0).map((e) => e.cls);
  const withE = enriched(clips);
  const cells: EntityConditionMatrix["cells"] = {};
  for (const e of cov) {
    if (e.instances === 0) continue;
    cells[e.cls] = {};
    for (const cond of conds) {
      const inCond = withE.filter((c) => cond.test(c));
      const ids = new Set(inCond.map((c) => c.clip_id));
      const contribs = e.contributions
        .filter((c) => ids.has(c.clipId))
        .map((c) => ({ clipId: c.clipId, title: c.title, value: c.instances }));
      const instances = contribs.reduce((a, c) => a + c.value, 0);
      cells[e.cls]![cond.key] = {
        instances,
        clips: contribs.sort((a, b) => b.value - a.value),
        gap: instances === 0 && inCond.length > 0,
      };
    }
  }
  return { classes, conditions: conds, cells };
}

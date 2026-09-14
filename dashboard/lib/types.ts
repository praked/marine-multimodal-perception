import { z } from "zod";

/* ---------------------------------------------------------------------------
   Data contracts. These mirror the repo's on-disk formats exactly:
   - sector protocol v1/v1.1 + learned-scorer fields (docs/reference/sector_protocol.md)
   - label records (labels/*.jsonl, docs/reference/data_formats.md)
   - typed detections (data/det) and instance masks (data/det_seg)
   The demo bundle and the Supabase Storage bundle share these shapes
   (dashboard/tools/bake_demo.py is the writer).
--------------------------------------------------------------------------- */

export const AttitudeSchema = z.object({
  roll_deg: z.number(),
  pitch_deg: z.number(),
  yaw_deg: z.number(),
});
export type Attitude = z.infer<typeof AttitudeSchema>;

export const MotionStateSchema = z.enum([
  "closing",
  "crossing",
  "diverging",
  "static",
  "unknown",
]);
export type MotionState = z.infer<typeof MotionStateSchema>;

/** One tracked radar object (protocol v1.2 `targets`; wire shape =
    `target_to_dict` in scripts/sensor_processing/target_motion.py).
    Position is polar (bearing right+, 0 = bow; range in metres) — the
    boat-frame x/y are derived, not carried. Velocity channels are null
    while they warm up (no Doppler point / rate window unfilled); CPA is
    null when speed is under the noise floor or the CPA is in the past. */
export const TargetSchema = z.object({
  id: z.number(),
  bearing_deg: z.number(),
  range_m: z.number(),
  n_points: z.number(),
  age_frames: z.number(),
  v_radial_mps: z.number().nullable(), // range-rate, negative = approaching
  v_tangential_mps: z.number().nullable(), // r×bearing-rate, positive = right
  vx_mps: z.number().nullable(), // boat-frame lateral velocity
  vy_mps: z.number().nullable(), // boat-frame forward velocity
  speed_mps: z.number().nullable(),
  course_deg: z.number().nullable(),
  closing_mps: z.number().nullable(), // = -v_radial (positive = approaching)
  cpa_m: z.number().nullable(),
  t_cpa_s: z.number().nullable(),
  motion_state: MotionStateSchema,
  ego_corrected: z.boolean(),
});
export type Target = z.infer<typeof TargetSchema>;

export const SectorRecordSchema = z.object({
  protocol: z.number(),
  timestamp: z.string(),
  clip_id: z.string(),
  bin_centers_deg: z.array(z.number()),
  scores: z.array(z.number()),
  min_range_m: z.array(z.number().nullable()),
  sensor_hits: z.array(z.tuple([z.number(), z.number(), z.number()])),
  tracked: z.boolean(),
  per_bin_velocity_mps: z.array(z.number().nullable()).optional(),
  per_bin_ttc_s: z.array(z.number().nullable()).optional(),
  recommended_heading_deg: z.number().nullable().optional(),
  heading_reason: z.string().optional(),
  heading_candidates_deg: z.array(z.number()).optional(),
  heading_cost_curve: z.array(z.number()).optional(),
  smoothed_heading_deg: z.number().nullable().optional(),
  // v1.1 optional
  attitude: AttitudeSchema.nullable().optional(),
  free_space_m: z.array(z.number().nullable()).optional(),
  confirmed: z.array(z.boolean()).optional(),
  // learned scorer (--scorer)
  p_obstacle: z.array(z.number()).optional(),
  threat: z.array(z.number()).optional(),
  // v1.2 optional: per-target motion (--targets), sorted nearest first
  targets: z.array(TargetSchema).optional(),
});
export type SectorRecord = z.infer<typeof SectorRecordSchema>;

export const BoxSchema = z.object({
  cls: z.string(),
  xyxy: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  confidence: z.number().nullable().optional(),
  source: z.string().optional(),
  /** which text prompt fired (open-vocabulary teachers; tuning signal) */
  prompt: z.string().optional(),
  /** teacher instance mask as a flat normalised x,y polygon (DART/SAM3) */
  polygon: z.array(z.number()).optional(),
  /** normalised pseudo-centroid inside the box: where the object actually
      is (mask/point-prompt seed for instance annotation) */
  centroid: z.tuple([z.number(), z.number()]).optional(),
  /** image space of xyxy: undistorted fisheye (default) or the thermal
      frame (pure-thermal annotation, 2026-09-12: night frames where the
      RGB is black are boxed directly on the 160x120 Lepton frame) */
  space: z.enum(["fisheye", "thermal"]).optional(),
});
export type Box = z.infer<typeof BoxSchema>;

export const InstanceSchema = z.object({
  cls: z.string(),
  confidence: z.number().nullable().optional(),
  polygon: z.array(z.number()), // flat normalised x,y pairs
});
export type Instance = z.infer<typeof InstanceSchema>;

export const LabelRecordSchema = z.object({
  frame_id: z.string(),
  scene: z.string().optional(),
  triplet_ts: z.string().optional(),
  frame_ts: z.string().optional(),
  source: z.string(),
  model_version: z.string().optional(),
  audited: z.boolean().optional(),
  fisheye_bboxes: z.array(BoxSchema),
  obstacle_bins_fisheye: z.array(z.number()).optional(),
  width: z.number().optional(),
  height: z.number().optional(),
  safe_heading_deg: z.number().nullable().optional(),
});
export type LabelRecord = z.infer<typeof LabelRecordSchema>;

export const FrameEntrySchema = z.object({
  ts: z.string(), // "HH:MM:SS.f" — the cross-stream RoundedTime key
  fisheye: z.boolean(),
  thermal: z.boolean(),
  /** present-but-degraded thermal (below the quality-guard contrast floors) */
  thermal_q: z.enum(["low"]).optional(),
  seg: z.boolean(),
  /** Source capture chunk (triplet_ts) when the clip is a concatenated
      activity and this frame is NOT from the first chunk. Keeps frame_ids
      round-tripping to the right capture files. */
  chunk: z.string().optional(),
});
export type FrameEntry = z.infer<typeof FrameEntrySchema>;

export const EnrichmentSchema = z.object({
  gps_init: z.object({
    lat: z.number(),
    lon: z.number(),
    source: z.string(),
  }),
  /** [ts "HH:MM:SS.f", elevation_deg, azimuth_deg from N] every ~60 s. */
  sun_samples: z.array(z.tuple([z.string(), z.number(), z.number()])),
  dayparts: z.record(z.string(), z.number()),
  luminance: z.object({
    mean: z.number().nullable(),
    hist: z.array(z.number()),
    sampled_every: z.number(),
  }),
  weather: z
    .object({
      cloud_cover_pct: z.number(),
      precipitation_mm: z.number(),
      wind_speed_kmh: z.number(),
      wind_dir_deg: z.number().nullable().optional(),
      temperature_c: z.number(),
      source: z.string(),
    })
    .nullable(),
  entities: z.record(
    z.string(),
    z.object({
      instances: z.number(),
      frames: z.number(),
      /** Per detection-stream split (typed-det / instance-seg / labels).
          The headline `instances` is the MAX across streams — never a sum,
          since the streams are different detectors over the same frames. */
      by_source: z
        .record(
          z.string(),
          z.object({ instances: z.number(), frames: z.number() }),
        )
        .optional(),
    }),
  ),
});
export type Enrichment = z.infer<typeof EnrichmentSchema>;

export const StreamsSchema = z.object({
  fisheye: z.boolean(),
  thermal: z.boolean(),
  radar: z.boolean(),
  imu: z.boolean(),
  seg: z.boolean(),
  sectors: z.boolean(),
});
export type Streams = z.infer<typeof StreamsSchema>;

export const ClipMetaSchema = z.object({
  clip_id: z.string(),
  scene: z.string(),
  triplet_ts: z.string(),
  title: z.string(),
  description: z.string(),
  image_size: z.tuple([z.number(), z.number()]),
  native_size: z.tuple([z.number(), z.number()]),
  thermal_size: z.tuple([z.number(), z.number()]).nullable(),
  streams: StreamsSchema,
  bin_centers_deg: z.array(z.number()),
  n_frames: z.number(),
  n_labelled: z.number().optional(),
  thumb_ts: z.string().optional(),
  /** Concatenated-activity support: all member chunk timestamps (first =
      triplet_ts) and the last chunk's timestamp for range display. */
  chunks: z.array(z.string()).optional(),
  end_ts: z.string().optional(),
  enrichment: EnrichmentSchema.optional(),
  frames: z.array(FrameEntrySchema),
});
export type ClipMeta = z.infer<typeof ClipMetaSchema>;

export const ClipSummarySchema = ClipMetaSchema.pick({
  clip_id: true,
  scene: true,
  triplet_ts: true,
  title: true,
  description: true,
  streams: true,
  n_frames: true,
  n_labelled: true,
  thumb_ts: true,
  chunks: true,
  end_ts: true,
  enrichment: true,
});
export type ClipSummary = z.infer<typeof ClipSummarySchema>;

export const CatalogueSchema = z.object({
  clips: z.array(ClipSummarySchema),
});

/** Radar points per RoundedTime: rows of [x, y, z, v?] metres / m/s. */
export type RadarFrames = Record<string, number[][]>;

/* ------------------------------- annotation ------------------------------ */

export const AuditVerdictSchema = z.enum(["accept", "reject", "edit"]);
export type AuditVerdict = z.infer<typeof AuditVerdictSchema>;

export const AuditRecordSchema = z.object({
  id: z.string(), // client uuid
  clip_key: z.string(), // "<scene>__<triplet_ts>"
  frame_ts: z.string(),
  frame_id: z.string(),
  verdict: AuditVerdictSchema,
  boxes: z.array(BoxSchema), // resulting boxes (empty for reject)
  source: z.string(), // originating label source
  created_at: z.string(), // ISO timestamp
});
export type AuditRecord = z.infer<typeof AuditRecordSchema>;

/** Key used on disk and in URLs: "<scene>__<triplet_ts>". */
export function clipKey(scene: string, tripletTs: string): string {
  return `${scene}__${tripletTs}`;
}

export function clipKeyFromId(clipId: string): string {
  return clipId.replace("/", "__");
}

/** "HH:MM:SS.f" -> filename-safe "HH-MM-SS.f" (mirrors safe_ts in the repo). */
export function safeTs(ts: string): string {
  return ts.replaceAll(":", "-");
}

/** frame_id = "<scene>/<triplet_ts>/ts=<HH-MM-SS.f>" (repo convention). */
export function frameIdFor(scene: string, tripletTs: string, ts: string): string {
  return `${scene}/${tripletTs}/ts=${safeTs(ts)}`;
}

import { z } from "zod";

/* Set curation records (public.sail_curation). Metadata only: a record
   says what the corpus SHOULD show; no write here touches a file. Keys are
   dashboard clip keys "<scene>__<triplet_ts>" — an activity (catalogue
   row) or a lone capture chunk that never had a bundle (the baker's former
   hard-coded exclusions). Cuts are in frame-id timestamp units
   ("HH:MM:SS.f", inclusive both ends) so labels/audits/sectors keyed by
   frame id keep lining up. */

export const CutSchema = z.object({
  start_ts: z.string(),
  end_ts: z.string(),
  note: z.string().optional(),
});
export type Cut = z.infer<typeof CutSchema>;

export const CurationRecordSchema = z.object({
  clip_key: z.string(),
  deleted_at: z.string().nullable().default(null),
  restored_at: z.string().nullable().default(null),
  purged_at: z.string().nullable().default(null),
  cuts: z.array(CutSchema).default([]),
  note: z.string().nullable().optional(),
  updated_at: z.string(),
  updated_by: z.string().nullable().optional(),
});
export type CurationRecord = z.infer<typeof CurationRecordSchema>;

export type CurationAction = "delete" | "restore" | "set_cuts" | "purge" | "seed";

/** Retention: a deleted set's bundle is kept this long before `prune` may
    remove it. Until then restore is instant. */
export const RETENTION_DAYS = 30;

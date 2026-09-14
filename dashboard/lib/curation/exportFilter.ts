import { inCut } from "./logic";
import type { CurationRecord } from "./types";

/* Which bundle objects a data link ships once cuts are applied: per-frame
   files (frames/thermal/seg) whose timestamp lies in a cut of their set
   are omitted; everything else (meta/sectors/radar/boxes/labels JSON) is
   shipped whole — the Python readers skip cut entries via
   configs/curation.yaml. Pure; tested in tests/curation.test.ts. */

const FRAME_OBJECT = /^[^/]+\/(frames|thermal|seg)\/ts=([0-9]{2}-[0-9]{2}-[0-9]{2}\.[0-9])\.(jpg|png)$/;

export function filterCutObjects<T extends { key: string }>(
  objects: readonly T[],
  records: ReadonlyMap<string, CurationRecord>,
): { kept: T[]; cut: number } {
  let cut = 0;
  const kept = objects.filter((o) => {
    const m = FRAME_OBJECT.exec(o.key);
    if (!m) return true;
    const rec = records.get(o.key.split("/")[0]!);
    if (!rec || rec.cuts.length === 0) return true;
    if (inCut(rec.cuts, m[2]!)) {
      cut++;
      return false;
    }
    return true;
  });
  return { kept, cut };
}

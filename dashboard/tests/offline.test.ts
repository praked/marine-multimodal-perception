import "fake-indexeddb/auto";
import { mergeAudits } from "@/lib/audit/backend";
import { idbReset } from "@/lib/offline/db";
import {
  classifyError,
  enqueueAudit,
  isNetworkError,
  queuedAudits,
  queuedCount,
  syncQueue,
} from "@/lib/offline/queue";
import {
  estimateBytes,
  packableFrames,
  packablePlanFrames,
} from "@/lib/offline/packs";
import type { CurationRecord } from "@/lib/curation";
import type { AuditRecord, ClipMeta } from "@/lib/types";
import { beforeEach, describe, expect, it } from "vitest";

function rec(id: string, clip = "s__t", ts = "12:00:00.0"): AuditRecord {
  return {
    id, clip_key: clip, frame_ts: ts, frame_id: `s/t/ts=${ts.replaceAll(":", "-")}`,
    verdict: "accept", boxes: [], source: "test", created_at: `2026-08-28T10:00:0${id.slice(-1)}Z`,
  };
}

describe("offline audit queue", () => {
  beforeEach(async () => {
    await idbReset();
  });

  it("queues, lists per clip and counts", async () => {
    await enqueueAudit(rec("a1"));
    await enqueueAudit(rec("a2", "other__x"));
    expect(await queuedCount()).toBe(2);
    expect((await queuedAudits("s__t")).map((q) => q.id)).toEqual(["a1"]);
  });

  it("sync replays in order; duplicate-key counts as synced (never a second row)", async () => {
    await enqueueAudit(rec("a1"));
    await enqueueAudit(rec("a2"));
    await enqueueAudit(rec("a3"));
    const inserted: string[] = [];
    const report = await syncQueue(async (r) => {
      if (r.id === "a2") throw { code: "23505", message: 'duplicate key value violates unique constraint "sail_audits_pkey"' };
      inserted.push(r.id);
    });
    expect(inserted).toEqual(["a1", "a3"]);
    expect(report).toMatchObject({ synced: 2, duplicates: 1, rejected: 0, remaining: 0, stoppedOnNetwork: false });
    expect(await queuedCount()).toBe(0);
  });

  it("stops at a network failure and keeps order; rejected records stay with their error", async () => {
    await enqueueAudit(rec("a1"));
    await enqueueAudit(rec("a2"));
    await enqueueAudit(rec("a3"));
    let calls = 0;
    const r1 = await syncQueue(async (r) => {
      calls++;
      if (r.id === "a1") throw { code: "23514", message: "new row violates check constraint" };
      if (r.id === "a2") throw new TypeError("Failed to fetch");
    });
    expect(calls).toBe(2); // a3 never attempted after the network stop
    expect(r1).toMatchObject({ synced: 0, rejected: 1, stoppedOnNetwork: true, remaining: 3 });
    const q = await queuedAudits();
    expect(q.find((x) => x.id === "a1")!.last_error).toMatch(/check constraint/);
    expect(q.find((x) => x.id === "a2")!.attempts).toBe(1);
    // retried sync with the network back: a1 still rejected, a2 + a3 land
    const r2 = await syncQueue(async (r) => {
      if (r.id === "a1") throw { code: "23514", message: "still bad" };
    });
    expect(r2).toMatchObject({ synced: 2, rejected: 1, remaining: 1 });
  });

  it("classifies errors", () => {
    expect(classifyError({ code: "23505", message: "dup" })).toBe("duplicate");
    expect(classifyError(new TypeError("Failed to fetch"))).toBe("retry");
    expect(classifyError({ message: "TypeError: Load failed" })).toBe("retry");
    expect(classifyError({ code: "42501", message: "permission denied" })).toBe("rejected");
    expect(isNetworkError({ status: 0 })).toBe(true);
  });

  it("mergeAudits de-duplicates by id and keeps time order", () => {
    const merged = mergeAudits([rec("a2"), rec("a1")], [rec("a1"), rec("a3")]);
    expect(merged.map((r) => r.id)).toEqual(["a1", "a2", "a3"]);
  });
});

describe("offline pack manifest logic", () => {
  const meta = {
    frames: [
      { ts: "12:00:00.0", fisheye: true, thermal: true, seg: false },
      { ts: "12:00:01.0", fisheye: true, thermal: false, seg: false },
      { ts: "12:00:02.0", fisheye: true, thermal: true, seg: false },
      { ts: "12:00:03.0", fisheye: false, thermal: true, seg: false },
    ],
  } as unknown as ClipMeta;
  const cutRec = {
    clip_key: "s__t", deleted_at: null, restored_at: null, purged_at: null,
    cuts: [{ start_ts: "12:00:01.0", end_ts: "12:00:01.0" }], updated_at: "x",
  } as CurationRecord;

  it("packableFrames drops non-fisheye and cut frames, none for a deleted set", () => {
    expect(packableFrames("s__t", meta, undefined).map((f) => f.ts)).toEqual(["12:00:00.0", "12:00:01.0", "12:00:02.0"]);
    expect(packableFrames("s__t", meta, cutRec).map((f) => f.ts)).toEqual(["12:00:00.0", "12:00:02.0"]);
    expect(packableFrames("s__t", meta, { ...cutRec, deleted_at: "2026-08-28T00:00:00Z" })).toEqual([]);
  });

  it("packablePlanFrames filters by curation and annotates thermal", () => {
    const plan = [
      { clipKey: "s__t", ts: "12:00:00.0" },
      { clipKey: "s__t", ts: "12:00:01.0" }, // cut
      { clipKey: "gone__x", ts: "12:00:00.0" }, // no meta
    ];
    const out = packablePlanFrames(plan, new Map([["s__t", meta]]), new Map([["s__t", cutRec]]));
    expect(out.frames).toEqual([{ clipKey: "s__t", ts: "12:00:00.0", thermal: true }]);
    expect(out.dropped).toBe(2);
  });

  it("estimateBytes budgets fisheye + thermal", () => {
    expect(estimateBytes([{ thermal: true }, { thermal: false }])).toBe(75_000 * 2 + 6_000);
  });
});

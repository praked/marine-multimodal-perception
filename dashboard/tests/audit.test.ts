import {
  auditsToJsonl,
  createLocalBackend,
  foldAudits,
} from "@/lib/audit/backend";
import type { AuditRecord } from "@/lib/types";
import { beforeEach, describe, expect, it } from "vitest";

function rec(
  frameTs: string,
  verdict: AuditRecord["verdict"],
  createdAt: string,
  id = crypto.randomUUID(),
): AuditRecord {
  return {
    id,
    clip_key: "2026-07-08__2026-07-08_16-37-01",
    frame_ts: frameTs,
    frame_id: `2026-07-08/2026-07-08_16-37-01/ts=${frameTs.replaceAll(":", "-")}`,
    verdict,
    boxes:
      verdict === "reject"
        ? []
        : [{ cls: "boat", xyxy: [0.1, 0.2, 0.3, 0.4] }],
    source: "grounding-dino",
    created_at: createdAt,
  };
}

describe("append-only audit log", () => {
  it("folds to the latest verdict per frame", () => {
    const records = [
      rec("16:37:01.1", "edit", "2026-08-06T01:00:00Z"),
      rec("16:37:01.1", "reject", "2026-08-06T02:00:00Z"),
      rec("16:37:03.1", "accept", "2026-08-06T01:30:00Z"),
    ];
    const folded = foldAudits(records);
    expect(folded.size).toBe(2);
    expect(folded.get("16:37:01.1")!.verdict).toBe("reject");
    expect(folded.get("16:37:03.1")!.verdict).toBe("accept");
  });

  it("exports folded JSONL in the repo's training-frames shape", () => {
    const jsonl = auditsToJsonl(
      [
        rec("16:37:01.1", "edit", "2026-08-06T01:00:00Z"),
        rec("16:37:03.1", "reject", "2026-08-06T01:30:00Z"),
      ],
      { width: 864, height: 648 },
    );
    const lines = jsonl.split("\n").map((l) => JSON.parse(l));
    expect(lines).toHaveLength(2);
    expect(lines[0]).toMatchObject({
      frame_id: "2026-07-08/2026-07-08_16-37-01/ts=16-37-01.1",
      scene: "2026-07-08",
      triplet_ts: "2026-07-08_16-37-01",
      source: "dashboard-web",
      audited: true,
      width: 864,
      height: 648,
    });
    expect(lines[0].fisheye_bboxes).toEqual([
      { cls: "boat", xyxy: [0.1, 0.2, 0.3, 0.4] },
    ]);
    expect(lines[1].fisheye_bboxes).toEqual([]);
  });
});

describe("local audit backend (demo mode)", () => {
  beforeEach(() => window.localStorage.clear());

  it("appends and lists per clip, surviving malformed stored data", async () => {
    const backend = createLocalBackend();
    await backend.append(rec("16:37:01.1", "accept", "2026-08-06T01:00:00Z"));
    await backend.append(rec("16:37:03.1", "edit", "2026-08-06T01:01:00Z"));
    const listed = await backend.list("2026-07-08__2026-07-08_16-37-01");
    expect(listed).toHaveLength(2);
    expect(await backend.list("other__clip")).toHaveLength(0);

    window.localStorage.setItem("asvproject.audits.v1", "not json");
    expect(await backend.list("2026-07-08__2026-07-08_16-37-01")).toEqual([]);
  });
});

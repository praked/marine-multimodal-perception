import { renderCurationYaml } from "@/tools/curation/export";
import type { CurationRecord } from "@/lib/curation";
import { describe, expect, it } from "vitest";

function rec(over: Partial<CurationRecord>): CurationRecord {
  return {
    clip_key: "s__t", deleted_at: null, restored_at: null, purged_at: null,
    cuts: [], note: null, updated_at: "2026-08-28T10:00:00Z", updated_by: "test",
    ...over,
  };
}

describe("curation.yaml export", () => {
  it("writes only deleted or trimmed sets, sorted, with chunks from the catalogue", () => {
    const yaml = renderCurationYaml(
      [
        rec({ clip_key: "z__2026-01-01_00-00-00", deleted_at: "2026-08-28T10:00:00Z", note: "indoor" }),
        rec({ clip_key: "a__2026-02-02_00-00-00", cuts: [{ start_ts: "10:00:00.0", end_ts: "10:00:05.0", note: "op" }] }),
        rec({ clip_key: "m__2026-03-03_00-00-00", deleted_at: "2026-08-01T00:00:00Z", restored_at: "2026-08-02T00:00:00Z" }), // restored, no cuts: omitted
        rec({ clip_key: "p__2026-04-04_00-00-00", deleted_at: "2026-06-01T00:00:00Z", purged_at: "2026-07-05T00:00:00Z" }),
      ],
      new Map([["a__2026-02-02_00-00-00", ["2026-02-02_00-00-00", "2026-02-02_00-05-00"]]]),
      "2026-08-28T20:00:00Z",
    );
    expect(yaml).toContain('exported_at: "2026-08-28T20:00:00Z"');
    const keys = [...yaml.matchAll(/^  "([^"]+)":$/gm)].map((m) => m[1]);
    expect(keys).toEqual(["a__2026-02-02_00-00-00", "p__2026-04-04_00-00-00", "z__2026-01-01_00-00-00"]);
    expect(yaml).toContain('chunks: ["2026-02-02_00-00-00", "2026-02-02_00-05-00"]');
    expect(yaml).toContain('chunks: ["2026-01-01_00-00-00"]'); // fallback: the key's own ts
    expect(yaml).toContain('- {start_ts: "10:00:00.0", end_ts: "10:00:05.0", note: "op"}');
    expect(yaml).toContain('purged_at: "2026-07-05T00:00:00Z"');
    expect(yaml).toContain('note: "indoor"');
  });

  it("emits an empty mapping when nothing is curated", () => {
    expect(renderCurationYaml([], new Map(), "2026-08-28T20:00:00Z")).toContain("sets: {}");
  });
});

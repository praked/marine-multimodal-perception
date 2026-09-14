import { isValidAssetPath, quantisedSigningDate, SIGN_TTL_S, SIGN_WINDOW_S } from "@/lib/r2";
import { gateToken } from "@/lib/gate";
import { describe, expect, it } from "vitest";

describe("quantised signing", () => {
  it("floors to the window so URLs are stable within it", () => {
    const w = SIGN_WINDOW_S * 1000;
    const t0 = 1754500000000;
    const base = Math.floor(t0 / w) * w;
    expect(quantisedSigningDate(base + 1).getTime()).toBe(base);
    expect(quantisedSigningDate(base + w - 1).getTime()).toBe(base);
    expect(quantisedSigningDate(base + w).getTime()).toBe(base + w);
  });

  it("TTL safely exceeds the window", () => {
    expect(SIGN_TTL_S).toBeGreaterThanOrEqual(SIGN_WINDOW_S * 2);
  });
});

describe("asset path validation", () => {
  it("accepts bundle-shaped keys", () => {
    expect(isValidAssetPath("2026-07-08__2026-07-08_16-37-01/meta.json")).toBe(true);
    expect(isValidAssetPath("2026-07-08__2026-07-08_16-37-01/frames/ts=16-37-01.1.jpg")).toBe(true);
    expect(isValidAssetPath("2026-07-08__2026-07-08_16-37-01/seg/ts=16-37-01.1.png")).toBe(true);
    expect(isValidAssetPath("clips.json")).toBe(false); // top-level via catalogue table, not R2 route? -> still 2 segments minimum
  });

  it("rejects traversal, hidden files and unknown extensions", () => {
    expect(isValidAssetPath("../secrets.json")).toBe(false);
    expect(isValidAssetPath("a/../../b.json")).toBe(false);
    expect(isValidAssetPath("clip/.env")).toBe(false);
    expect(isValidAssetPath("clip/frames/x.exe")).toBe(false);
    expect(isValidAssetPath("clip//x.jpg")).toBe(false);
    expect(isValidAssetPath("a/b/c/d/e.jpg")).toBe(false);
  });
});

describe("gate token", () => {
  it("is a deterministic sha256 hex of the password", async () => {
    const a = await gateToken("hunter2");
    const b = await gateToken("hunter2");
    const c = await gateToken("other");
    expect(a).toBe(b);
    expect(a).toMatch(/^[0-9a-f]{64}$/);
    expect(a).not.toBe(c);
  });
});

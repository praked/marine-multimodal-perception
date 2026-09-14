import { describe, expect, it } from "vitest";
import {
  MAX_FILE_BYTES,
  checkSession,
  isValidCaptureFile,
  isValidSessionName,
  uploadKey,
} from "@/lib/upload";

describe("isValidSessionName", () => {
  it("accepts capture-day style names", () => {
    expect(isValidSessionName("2026-08-25_deployment")).toBe(true);
    expect(isValidSessionName("2026-08-19_afloat")).toBe(true);
  });
  it("rejects traversal, separators and short/odd names", () => {
    expect(isValidSessionName("../etc")).toBe(false);
    expect(isValidSessionName("a/b")).toBe(false);
    expect(isValidSessionName("a b")).toBe(false);
    expect(isValidSessionName("ab")).toBe(false);
    expect(isValidSessionName("-leading")).toBe(false);
    expect(isValidSessionName("x".repeat(65))).toBe(false);
  });
});

describe("isValidCaptureFile", () => {
  it("accepts the continuous_capture output set", () => {
    for (const n of [
      "fisheye_2026-08-25_09-30-00.mp4",
      "thermal_2026-08-25_09-30-00.mp4",
      "mmwave_2026-08-25_09-30-00.csv",
      "imu_2026-08-25_09-30-00.csv",
      "frames_2026-08-25_09-30-00.csv",
      "gps_2026-08-25_09-30-00.csv",
      "radar_profile.cfg",
      "box.env",
      "capture.log",
    ]) {
      expect(isValidCaptureFile(n), n).toBe(true);
    }
  });
  it("rejects everything else", () => {
    for (const n of [
      "fisheye_2026-08-25_09-30-00.avi",
      "mmwave_2026-08-25.csv",
      "notes.txt",
      "../fisheye_2026-08-25_09-30-00.mp4",
      "a/fisheye_2026-08-25_09-30-00.mp4",
      "fisheye_2026-08-25_09-30-00.mp4.exe",
      ".env",
    ]) {
      expect(isValidCaptureFile(n), n).toBe(false);
    }
  });
});

describe("checkSession", () => {
  const ts = "2026-08-25_09-30-00";
  it("passes a full chunk and reports thermal-absent honestly", () => {
    const full = checkSession([
      `fisheye_${ts}.mp4`, `thermal_${ts}.mp4`, `mmwave_${ts}.csv`,
      `imu_${ts}.csv`, `frames_${ts}.csv`, `gps_${ts}.csv`,
      "radar_profile.cfg",
    ]);
    expect(full.ok).toBe(true);
    expect(full.chunks).toHaveLength(1);
    expect(full.extras).toEqual(["radar_profile.cfg"]);
    expect(full.problems).toEqual([]);

    const noThermal = checkSession([
      `fisheye_${ts}.mp4`, `mmwave_${ts}.csv`,
    ]);
    expect(noThermal.ok).toBe(true); // thermal is honest-absent, not required
    expect(noThermal.problems.some((p) => p.includes("no thermal"))).toBe(true);
  });
  it("fails when a chunk misses a required stream", () => {
    const c = checkSession([`fisheye_${ts}.mp4`, `thermal_${ts}.mp4`]);
    expect(c.ok).toBe(false);
    expect(c.problems.some((p) => p.includes("missing mmwave"))).toBe(true);
  });
  it("fails on an empty or unrecognised pick, listing rejects", () => {
    const c = checkSession(["notes.txt", "IMG_1234.jpg"]);
    expect(c.ok).toBe(false);
    expect(c.rejected).toEqual(["notes.txt", "IMG_1234.jpg"]);
    expect(c.problems[0]).toContain("no capture chunks");
  });
  it("pairs multiple chunks by timestamp, sorted", () => {
    const t2 = "2026-08-25_09-35-00";
    const c = checkSession([
      `fisheye_${t2}.mp4`, `mmwave_${t2}.csv`,
      `fisheye_${ts}.mp4`, `mmwave_${ts}.csv`,
    ]);
    expect(c.ok).toBe(true);
    expect(c.chunks.map((x) => x.ts)).toEqual([ts, t2]);
  });
});

describe("uploadKey", () => {
  it("builds only whitelisted keys under incoming/", () => {
    expect(uploadKey("2026-08-25_deployment", "fisheye_2026-08-25_09-30-00.mp4"))
      .toBe("incoming/2026-08-25_deployment/fisheye_2026-08-25_09-30-00.mp4");
    expect(uploadKey("../clips", "fisheye_2026-08-25_09-30-00.mp4")).toBeNull();
    expect(uploadKey("2026-08-25_deployment", "evil.sh")).toBeNull();
  });
  it("caps at 3 GB", () => {
    expect(MAX_FILE_BYTES).toBe(3 * 1024 ** 3);
  });
});

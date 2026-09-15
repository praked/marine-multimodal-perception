import {
  formatActivityRange,
  formatScene,
  formatTripletTs,
  parseTripletTs,
} from "@/lib/format";
import { describe, expect, it } from "vitest";

describe("dd-mm-yyyy display formatting", () => {
  it("parses canonical triplet timestamps", () => {
    expect(parseTripletTs("2026-08-18_17-57-27")).toEqual({
      date: "18-08-2026",
      time: "17:57:27",
    });
    expect(parseTripletTs("garbage")).toBeNull();
  });

  it("formats single timestamps", () => {
    expect(formatTripletTs("2026-08-18_17-57-27")).toBe("18-08-2026 17:57");
    expect(formatTripletTs("not-a-ts")).toBe("not-a-ts");
  });

  it("formats activity ranges, collapsing same-day ends", () => {
    expect(
      formatActivityRange("2026-08-18_17-57-27", "2026-08-18_19-28-33"),
    ).toBe("18-08-2026 17:57–19:28");
    expect(formatActivityRange("2026-08-18_17-57-27")).toBe(
      "18-08-2026 17:57",
    );
    expect(
      formatActivityRange("2026-08-18_23-57-27", "2026-08-19_00-08-33"),
    ).toBe("18-08-2026 23:57 – 19-08-2026 00:08");
  });

  it("humanises scene slugs", () => {
    expect(formatScene("2026-06-17_institutionone_day1")).toBe("Institutionone day1");
    expect(formatScene("Boats")).toBe("Boats");
    expect(formatScene("2026-07-08")).toBe("2026-07-08");
  });
});

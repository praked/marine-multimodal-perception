import { StreamChips } from "@/components/catalogue/StreamChips";
import { BinTable } from "@/components/viewer/BinTable";
import { HeadingPanel } from "@/components/viewer/HeadingPanel";
import { LayerPanel } from "@/components/viewer/LayerPanel";
import { RadarPanel } from "@/components/viewer/RadarPanel";
import { ScorerPanel } from "@/components/viewer/ScorerPanel";
import { TargetChips } from "@/components/viewer/TargetChips";
import { sectorView } from "@/lib/sectors";
import { useViewerStore } from "@/lib/stores/viewer";
import {
  SectorRecordSchema,
  type SectorRecord,
  type Target,
} from "@/lib/types";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

/* Component render tests use a REAL sector record from the baked bundle so
   the components are exercised with production-shaped data. */

const sectorsPath = path.resolve(
  __dirname,
  "..",
  "public",
  "demo",
  "2026-07-08__2026-07-08_16-37-01",
  "sectors.json",
);
const sectors = JSON.parse(readFileSync(sectorsPath, "utf8")) as Record<
  string,
  unknown
>;
const records = Object.values(sectors).map((r) => SectorRecordSchema.parse(r));
// a busy frame: something confirmed and some scores present
const busy: SectorRecord =
  records.find(
    (r) => r.confirmed?.some(Boolean) && r.scores.some((s) => s > 0.5),
  ) ?? records[Math.floor(records.length / 2)]!;
const busyView = sectorView(busy, 10); // native width -> identity

describe("ScorerPanel", () => {
  it("renders one wedge per non-zero sector and confirmed markers", () => {
    const { container } = render(
      <ScorerPanel view={busyView} scoreSource="scores" showConfirmed />,
    );
    const wedges = container.querySelectorAll("path[fill='var(--seeblau-deep)']");
    const nonZero = busy.scores.filter((s) => s > 0.005).length;
    expect(wedges).toHaveLength(nonZero);
    const markers = container.querySelectorAll(
      "path[fill='var(--seeblau-navy)']",
    );
    expect(markers).toHaveLength(busy.confirmed!.filter(Boolean).length);
  });

  it("switches to p_obstacle values when asked", () => {
    const { container } = render(
      <ScorerPanel view={busyView} scoreSource="p_obstacle" showConfirmed={false} />,
    );
    expect(
      container.querySelectorAll("path[fill='var(--seeblau-navy)']"),
    ).toHaveLength(0);
    const label = container.querySelector("text[font-weight='600']");
    expect(label).not.toBeNull();
    const max = Math.max(...busy.p_obstacle!);
    expect(label!.textContent).toBe(max.toFixed(2));
  });

  it("shows the empty state without a record", () => {
    render(<ScorerPanel view={null} scoreSource="scores" showConfirmed />);
    expect(screen.getByText(/sectors · none/i)).toBeInTheDocument();
  });
});

describe("RadarPanel", () => {
  const group = [
    [0.1, 2.0, 0, 0],
    [1.5, 4.2, 0, -0.3],
    [-0.2, 0.3, 0, 0], // below y_min: filtered when the clutter filter is on
  ];

  it("draws current returns above the y_min filter", () => {
    const { container } = render(
      <RadarPanel trail={[group]} down={false} yMin={0.5} />,
    );
    const points = container.querySelectorAll(
      "circle[fill='var(--seeblau-deep)']",
    );
    expect(points).toHaveLength(2);
  });

  it("draws a fading trail for previous groups", () => {
    const { container } = render(
      <RadarPanel trail={[group, group]} down={false} yMin={0} />,
    );
    expect(
      container.querySelectorAll("circle[fill='var(--viz-trail)']"),
    ).toHaveLength(3);
    expect(
      container.querySelectorAll("circle[fill='var(--seeblau-deep)']"),
    ).toHaveLength(3);
  });

  it("shows the honest DOWN state", () => {
    render(<RadarPanel trail={[]} down yMin={0} />);
    expect(screen.getByText(/radar · down/i)).toBeInTheDocument();
  });
});

/* --------------------- protocol v1.2 tracked targets --------------------- */

const closingTarget: Target = {
  id: 7,
  bearing_deg: -12,
  range_m: 4.2,
  n_points: 9,
  age_frames: 14,
  v_radial_mps: -0.61,
  v_tangential_mps: 0.11,
  vx_mps: 0.05,
  vy_mps: -0.6,
  speed_mps: 0.602,
  course_deg: 175.2,
  closing_mps: 0.61,
  cpa_m: 1.8,
  t_cpa_s: 12.4,
  motion_state: "closing",
  ego_corrected: true,
};

const crossingTarget: Target = {
  id: 12,
  bearing_deg: 25,
  range_m: 6.5,
  n_points: 4,
  age_frames: 6,
  v_radial_mps: null,
  v_tangential_mps: null,
  vx_mps: null,
  vy_mps: null,
  speed_mps: null,
  course_deg: null,
  closing_mps: null,
  cpa_m: null,
  t_cpa_s: null,
  motion_state: "crossing",
  ego_corrected: false,
};

describe("RadarPanel targets layer (protocol v1.2)", () => {
  const group = [[0.1, 2.0, 0, 0]];
  const targets = [closingTarget, crossingTarget];

  it("draws one motion-state-coloured marker per target", () => {
    const { container } = render(
      <RadarPanel trail={[group]} down={false} yMin={0} targets={targets} showTargets />,
    );
    const markers = container.querySelectorAll("circle[data-target-marker]");
    expect(markers).toHaveLength(2);
    expect(markers[0]!.getAttribute("stroke")).toBe("var(--status-serious)");
    expect(markers[1]!.getAttribute("stroke")).toBe("var(--viz-water-edge)");
    // a11y: each target group describes itself
    const g = container.querySelector("g[data-target-id='7']");
    expect(g!.getAttribute("aria-label")).toMatch(/target 7: closing/);
    expect(g!.getAttribute("aria-label")).toMatch(/CPA 1\.8 m/);
  });

  it("draws the velocity arrow and dashed CPA ghost only when known", () => {
    const { container } = render(
      <RadarPanel trail={[group]} down={false} yMin={0} targets={targets} showTargets />,
    );
    // closing target has velocity + CPA; crossing one is still warming up
    expect(container.querySelectorAll("line[data-target-arrow]")).toHaveLength(1);
    const ghost = container.querySelectorAll("circle[data-target-cpa-ghost]");
    expect(ghost).toHaveLength(1);
    expect(ghost[0]!.getAttribute("stroke-dasharray")).toBeTruthy();
    expect(
      container.querySelectorAll("line[data-target-cpa-connector]"),
    ).toHaveLength(1);
  });

  it("hides the layer when toggled off and without targets renders as today", () => {
    const off = render(
      <RadarPanel trail={[group]} down={false} yMin={0} targets={targets} showTargets={false} />,
    );
    expect(
      off.container.querySelectorAll("[data-target-marker]"),
    ).toHaveLength(0);
    const plain = render(<RadarPanel trail={[group]} down={false} yMin={0} showTargets />);
    expect(
      plain.container.querySelectorAll("[data-target-marker]"),
    ).toHaveLength(0);
    expect(
      plain.container.querySelectorAll("circle[fill='var(--seeblau-deep)']"),
    ).toHaveLength(1); // raw returns unaffected
  });
});

describe("TargetChips", () => {
  it("lists state, range, closing speed and CPA per target in mono chips", () => {
    const { container } = render(
      <TargetChips targets={[closingTarget, crossingTarget]} />,
    );
    const items = within(container).getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]!.textContent).toContain("#7");
    expect(items[0]!.textContent).toContain("closing");
    expect(items[0]!.textContent).toContain("4.2 m");
    expect(items[0]!.textContent).toContain("0.61 m/s");
    expect(items[0]!.textContent).toContain("cpa 1.8 m/12 s");
    // warm-up channels stay honest dashes
    expect(items[1]!.textContent).toContain("crossing");
    expect(items[1]!.textContent).toContain("cpa –");
  });

  it("renders nothing for an empty target list", () => {
    const { container } = render(<TargetChips targets={[]} />);
    expect(container.firstChild).toBeNull();
  });
});

describe("Target tracks layer toggle", () => {
  const streams = {
    fisheye: true,
    thermal: true,
    radar: true,
    imu: true,
    seg: true,
    sectors: true,
  };

  it("defaults on, toggles the store, and persists via the viewer store", () => {
    useViewerStore.getState().setLayer("targetTracks", true);
    const { container } = render(<LayerPanel streams={streams} hasScorer />);
    const q = within(container);
    const btn = q.getByRole("button", { name: /target tracks/i });
    expect(btn).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(btn);
    expect(useViewerStore.getState().layers.targetTracks).toBe(false);
    expect(
      q.getByRole("button", { name: /target tracks/i }),
    ).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(q.getByRole("button", { name: /target tracks/i }));
    expect(useViewerStore.getState().layers.targetTracks).toBe(true);
  });

  it("is n/a when the clip has no sectors stream", () => {
    const { container } = render(
      <LayerPanel streams={{ ...streams, sectors: false }} hasScorer />,
    );
    expect(
      within(container).getByRole("button", { name: /target tracks/i }),
    ).toBeDisabled();
  });
});

describe("BinTable", () => {
  it("renders one row per bin with score and range", () => {
    render(<BinTable view={busyView} threshold={0.33} />);
    const rows = screen.getAllByRole("row");
    expect(rows).toHaveLength(1 + busy.bin_centers_deg.length);
  });

  it("handles a missing record", () => {
    render(<BinTable view={null} threshold={0.33} />);
    expect(screen.getByText(/no fusion record/i)).toBeInTheDocument();
  });
});

describe("HeadingPanel", () => {
  it("shows the recommendation or an honest abstain", () => {
    render(<HeadingPanel record={busy} />);
    if (busy.recommended_heading_deg == null) {
      expect(screen.getByText(/abstain/)).toBeInTheDocument();
    } else {
      expect(screen.getByText("recommended")).toBeInTheDocument();
    }
  });
});

describe("StreamChips", () => {
  it("strikes through absent streams instead of hiding them", () => {
    render(
      <StreamChips
        streams={{
          fisheye: true,
          thermal: false,
          radar: false,
          imu: false,
          seg: true,
          sectors: false,
        }}
      />,
    );
    expect(screen.getByTitle("radar stream absent")).toHaveClass(
      "line-through",
    );
    expect(screen.getByTitle("fisheye stream present")).not.toHaveClass(
      "line-through",
    );
  });
});


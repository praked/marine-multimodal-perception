"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

/* Layer + transport state for the clip viewer. Persisted per browser so a
   reload keeps the operator's layer choices (playhead is deliberately not
   persisted). */

export interface LayerState {
  seg: boolean;
  waterEdge: boolean;
  typedBoxes: boolean;
  labelBoxes: boolean;
  instances: boolean;
  /** Hide radar returns with Y < 0.5 m (mmwave.y_min mount-clutter guard). */
  clutterFilter: boolean;
  confirmedMarkers: boolean;
  /** Protocol-v1.2 tracked-target markers + chip row on the radar panel. */
  targetTracks: boolean;
  /** Radar returns projected into the fisheye (extrinsics + mount yaw). */
  radarOverlay: boolean;
}

export type ScoreSource = "scores" | "p_obstacle" | "threat";

interface ViewerState {
  layers: LayerState;
  threshold: number;
  scoreSource: ScoreSource;
  speed: number; // playback multiplier: 1 = real time (~3 fps footage)
  /** Displayed sector width in degrees (5° steps; native bins are
      resampled conservatively — see lib/sectors.ts). */
  sectorWidthDeg: number;
  /** Curation cuts: playback + stepping skip cut frames unless this is on
      (the "show cut frames" toggle). Not persisted: a review session that
      opted in should not silently carry into the next clip. */
  showCutFrames: boolean;
  setLayer(key: keyof LayerState, value: boolean): void;
  setThreshold(value: number): void;
  setScoreSource(value: ScoreSource): void;
  setSpeed(value: number): void;
  setSectorWidthDeg(value: number): void;
  setShowCutFrames(value: boolean): void;
}

export const DEFAULT_LAYERS: LayerState = {
  seg: true,
  waterEdge: true,
  typedBoxes: true,
  labelBoxes: false,
  instances: false,
  clutterFilter: true,
  confirmedMarkers: true,
  targetTracks: true,
  radarOverlay: false,
};

export const useViewerStore = create<ViewerState>()(
  persist(
    (set) => ({
      layers: DEFAULT_LAYERS,
      threshold: 0.33,
      // Learned p(obstacle) is the default since the corpus-trained scorer
      // beat the incumbent (2026-08-22); clips without scorer fields fall
      // back to the fused n/3 score at render time.
      scoreSource: "p_obstacle",
      speed: 1,
      sectorWidthDeg: 15, // = the D.2 native bin width
      showCutFrames: false,
      setLayer: (key, value) =>
        set((s) => ({ layers: { ...s.layers, [key]: value } })),
      setThreshold: (threshold) => set({ threshold }),
      setScoreSource: (scoreSource) => set({ scoreSource }),
      setSpeed: (speed) => set({ speed }),
      setSectorWidthDeg: (v) =>
        set({ sectorWidthDeg: Math.max(5, Math.min(45, Math.round(v / 5) * 5)) }),
      setShowCutFrames: (showCutFrames) => set({ showCutFrames }),
    }),
    {
      name: "asvproject.viewer.v1",
      partialize: (s) => ({
        layers: s.layers,
        threshold: s.threshold,
        scoreSource: s.scoreSource,
        speed: s.speed,
        sectorWidthDeg: s.sectorWidthDeg,
      }),
      // Deep-merge `layers` so a persisted state from before a new layer
      // key existed still picks up that key's default (shallow merge would
      // drop it and the toggle would silently read undefined).
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<ViewerState>;
        return {
          ...current,
          ...p,
          layers: { ...current.layers, ...(p.layers ?? {}) },
        };
      },
    },
  ),
);

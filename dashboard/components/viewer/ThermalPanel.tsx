"use client";

import { ThermometerSnowflake } from "lucide-react";
import { useEffect, useRef } from "react";

interface Props {
  /** Pre-decoded thermal frame from the shared FrameAssetLoader; swaps in
      the same commit as the fisheye so the two panels never drift. */
  img: HTMLImageElement | null;
  down: boolean;
  /** Why it's down — distinguishes "clip has no thermal at all" from "the
      sensor died at this point in the mission" (see the coverage ribbon). */
  downReason?: string;
}

/** Thermal camera panel. The Lepton stream is 160×120; pixels are shown
    honestly (nearest-neighbour upscale), as in the poster figure. Canvas
    drawImage of a decoded image = an atomic swap (an <img src> flip shows
    stale pixels until the new bytes arrive). */
export function ThermalPanel({ img, down, downReason }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext("2d")!.drawImage(img, 0, 0);
  }, [img]);

  if (down) {
    return (
      <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 text-subtle">
        <ThermometerSnowflake size={20} aria-hidden />
        <span className="font-mono text-[11px] uppercase tracking-[0.28em]">
          thermal · down
        </span>
        <span className="text-xs">
          {downReason ?? "no thermal stream on this frame"}
        </span>
      </div>
    );
  }
  return (
    <canvas
      ref={canvasRef}
      className="pixelated h-full w-full object-contain"
      role="img"
      aria-label="Thermal camera frame (160 by 120 pixels)"
    />
  );
}

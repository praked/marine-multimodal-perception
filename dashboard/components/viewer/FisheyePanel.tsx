"use client";

import { colourForClass } from "@/lib/palette";
import type { LayerState } from "@/lib/stores/viewer";
import type { Box, Instance, LabelRecord } from "@/lib/types";
import type { FrameAssets } from "@/lib/viewer/frameAssets";
import { projectRadarToFisheye } from "@/lib/viewer/radarProjection";
import { CameraOff } from "lucide-react";
import { useEffect, useRef } from "react";

interface Props {
  /** Pre-decoded assets from the shared FrameAssetLoader; null while the
      first frame is loading. Both camera panels swap on the same commit. */
  assets: FrameAssets | null;
  boxes: Box[];
  instances: Instance[];
  label: LabelRecord | null;
  layers: LayerState;
  size: [number, number]; // bundle image size
  /** Current frame's radar points ([x,y,z,(v)] rows) for the projection
      overlay; empty/undefined when the stream is down. */
  radarPoints?: number[][];
}

/** Fisheye camera panel: synchronous canvas compositing of pre-decoded
    assets — same draw order as the Tkinter dashboard's _overlay_camera.
    All async work lives in FrameAssetLoader. */
export function FisheyePanel({
  assets,
  boxes,
  instances,
  label,
  layers,
  size,
  radarPoints,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [w, h] = size;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !assets?.fisheye) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(assets.fisheye, 0, 0, w, h);

    const seg = assets.seg;
    if (seg && layers.seg) {
      const off = document.createElement("canvas");
      off.width = w;
      off.height = h;
      off.getContext("2d")!.putImageData(seg.tint, 0, 0);
      ctx.drawImage(off, 0, 0);
    }

    if (seg && layers.waterEdge) {
      ctx.strokeStyle = "#f2a33c"; // poster amber
      ctx.lineWidth = Math.max(2, w / 220);
      ctx.lineJoin = "round";
      for (const segment of seg.segments) {
        ctx.beginPath();
        segment.forEach(([x, y], i) => {
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
    }

    if (layers.instances) {
      for (const inst of instances) {
        const colour = colourForClass(inst.cls);
        ctx.beginPath();
        for (let i = 0; i + 1 < inst.polygon.length; i += 2) {
          const x = inst.polygon[i]! * w;
          const y = inst.polygon[i + 1]! * h;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.globalAlpha = 0.35;
        ctx.fillStyle = colour;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = colour;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    if (layers.typedBoxes) {
      ctx.font = `${Math.max(10, w / 40)}px ui-monospace, Menlo, monospace`;
      for (const box of boxes) {
        const colour = colourForClass(box.cls);
        const [x0, y0, x1, y1] = box.xyxy;
        const bx = x0 * w;
        const by = y0 * h;
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2;
        ctx.strokeRect(bx, by, (x1 - x0) * w, (y1 - y0) * h);
        const text = `${box.cls}${
          box.confidence != null ? ` ${box.confidence.toFixed(2)}` : ""
        }`;
        const tw = ctx.measureText(text).width;
        ctx.fillStyle = "rgba(10,42,58,0.75)";
        ctx.fillRect(bx, Math.max(0, by - 14), tw + 6, 13);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(text, bx + 3, Math.max(10, by - 4));
      }
    }

    if (layers.labelBoxes && label) {
      ctx.setLineDash([6, 4]);
      for (const box of label.fisheye_bboxes) {
        const [x0, y0, x1, y1] = box.xyxy;
        ctx.strokeStyle = colourForClass(box.cls);
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h);
      }
      ctx.setLineDash([]);
    }

    if (layers.radarOverlay && radarPoints && radarPoints.length) {
      // returns projected through the measured extrinsics + mount yaw;
      // the display clutter filter applies here too (same y ≥ 0.5 rule)
      const projected = projectRadarToFisheye(
        radarPoints, layers.clutterFilter ? 0.5 : 0);
      for (const r of projected) {
        const px = r.x * w;
        const py = r.y * h;
        // marker size falls with range (9 m envelope); colour by Doppler:
        // approaching red, receding blue, static/no-Doppler radar purple
        const radius = Math.max(2, 6 - (r.rangeY / 9) * 4);
        const colour = Number.isNaN(r.doppler) || Math.abs(r.doppler) < 0.05
          ? "#7A6BB5"
          : r.doppler < 0 ? "#C4442A" : "#1487B8";
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fillStyle = `${colour}B3`; // ~70% alpha
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = colour;
        ctx.stroke();
      }
    }
  }, [assets, boxes, instances, label, layers, w, h, radarPoints]);

  if (assets && !assets.fisheye) {
    return (
      <div className="flex h-full min-h-40 flex-col items-center justify-center gap-2 text-subtle">
        <CameraOff size={20} aria-hidden />
        <span className="font-mono text-[11px] uppercase tracking-[0.28em]">
          fisheye · down
        </span>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full object-contain"
      role="img"
      aria-label="Fisheye camera with segmentation and detection overlays"
    />
  );
}

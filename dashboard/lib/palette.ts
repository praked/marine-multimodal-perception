/* Data-carrying colours. UI chrome colours live in globals.css as tokens;
   these are the entity colours drawn on imagery + charts.

   The typed-class colours mirror scripts/utils/detections.CLASS_COLOURS and
   the label-tool classes keep their established identities so annotators
   moving from the Tkinter dashboard recognise every class instantly. */

/** eWaSR segmentation palette (RGB), scripts/utils/segmentation.py. */
export const SEG_PALETTE: Record<number, [number, number, number]> = {
  0: [247, 195, 37], // obstacle — amber
  1: [41, 167, 224], // water — seeblau
  2: [90, 75, 164], // sky — purple
};

/** LaRS typed-detector classes (RGB), scripts/utils/detections.CLASS_COLOURS. */
export const TYPED_CLASS_COLOURS: Record<string, string> = {
  boat_ship: "rgb(255,64,64)",
  row_boats: "rgb(255,140,0)",
  paddle_board: "rgb(255,215,0)",
  buoy: "rgb(0,220,120)",
  swimmer: "rgb(0,200,255)",
  animal: "rgb(180,100,255)",
  float: "rgb(255,0,200)",
  other: "rgb(180,180,180)",
};
export const TYPED_DEFAULT_COLOUR = "rgb(255,255,0)";

/** Manual/label-tool classes (converted from the tool's BGR overlays). */
export const LABEL_CLASS_COLOURS: Record<string, string> = {
  boat: "rgb(0,200,0)",
  duck: "rgb(255,200,0)",
  buoy: "rgb(230,0,0)",
  person: "rgb(0,80,255)",
  structure: "rgb(255,165,0)",
  other: "rgb(160,0,160)",
  unset: "rgb(128,128,128)",
};

export const LABEL_CLASSES = [
  "boat",
  "duck",
  "buoy",
  "person",
  "structure",
  "other",
] as const;
export type LabelClass = (typeof LABEL_CLASSES)[number];

/** Class key bindings, identical to the Tkinter dashboard / label tool. */
export const CLASS_KEYS: Record<string, LabelClass> = {
  b: "boat",
  d: "duck",
  B: "buoy",
  p: "person",
  m: "structure",
  o: "other",
};

export function colourForClass(cls: string): string {
  return (
    LABEL_CLASS_COLOURS[cls] ?? TYPED_CLASS_COLOURS[cls] ?? TYPED_DEFAULT_COLOUR
  );
}

/** Sequential accent blue ramp (light -> deep) for magnitude encodings. */
export const SEEBLAU_RAMP = [
  "#cceef9",
  "#a6e1f4",
  "#59c7eb",
  "#00a9e0",
  "#1487b8",
] as const;

/** Colour for a radar point by forward range (near = deep, far = light). */
export function rampForRange(rangeM: number, maxM = 9): string {
  const t = Math.min(Math.max(rangeM / maxM, 0), 1);
  const idx = Math.min(
    SEEBLAU_RAMP.length - 1,
    Math.floor((1 - t) * SEEBLAU_RAMP.length),
  );
  return SEEBLAU_RAMP[idx]!;
}

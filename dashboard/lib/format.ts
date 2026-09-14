/* Human-facing date/time formatting (en-GB, dd-mm-yyyy).
   Canonical identifiers stay ISO `YYYY-MM-DD_HH-MM-SS` everywhere — they
   sort chronologically as strings and round-trip to the capture files;
   these helpers only change what people read. */

/** "2026-08-18_17-57-27" -> {date: "18-08-2026", time: "17:57:27"} */
export function parseTripletTs(
  ts: string,
): { date: string; time: string } | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$/.exec(ts);
  if (!m) return null;
  const [, y, mo, d, h, mi, s] = m;
  return { date: `${d}-${mo}-${y}`, time: `${h}:${mi}:${s}` };
}

/** "2026-08-18_17-57-27" -> "18-08-2026 17:57" */
export function formatTripletTs(ts: string): string {
  const p = parseTripletTs(ts);
  if (!p) return ts;
  return `${p.date} ${p.time.slice(0, 5)}`;
}

/** Time range for an activity: "18-08-2026 17:57–19:28" (same day)
    or full both sides when the range crosses midnight. */
export function formatActivityRange(startTs: string, endTs?: string): string {
  const a = parseTripletTs(startTs);
  if (!a) return startTs;
  if (!endTs || endTs === startTs) return `${a.date} ${a.time.slice(0, 5)}`;
  const b = parseTripletTs(endTs);
  if (!b) return `${a.date} ${a.time.slice(0, 5)}`;
  if (a.date === b.date) {
    return `${a.date} ${a.time.slice(0, 5)}–${b.time.slice(0, 5)}`;
  }
  return `${a.date} ${a.time.slice(0, 5)} – ${b.date} ${b.time.slice(0, 5)}`;
}

/** Scene slug -> readable label: "2026-06-17_institutionone_day1" -> "InstitutionOne day1",
    "Boats" -> "Boats". Leading ISO dates are dropped (shown separately). */
export function formatScene(scene: string): string {
  const stripped = scene.replace(/^\d{4}-\d{2}-\d{2}_?/, "");
  if (!stripped) return scene; // scene was just a date
  return (stripped.charAt(0).toUpperCase() + stripped.slice(1)).replaceAll(
    "_",
    " ",
  );
}

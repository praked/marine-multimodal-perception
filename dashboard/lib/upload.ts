/* Capture-session upload: validation shared by the /upload page (client)
   and the /api/upload presign route (server). Uploads land under the
   `incoming/<session>/` prefix in the private bucket — strictly separated
   from the published clip bundles — and are processed on the workstation
   by dashboard/tools/process_incoming.ts (validate → bake → enrich →
   publish), which reports status back through `sail_ingests`. */

/** Session dirs: capture-day naming, safe charset only. */
const SESSION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{2,63}$/;

/** Files a capture session may contain (continuous_capture output). */
const FILE_RES: RegExp[] = [
  /^(fisheye|thermal)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mp4$/,
  /^(mmwave|imu|frames|gps)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.csv$/,
  /^radar_profile\.cfg$/,
  /^box\.env$/,
  /^capture\.log$/,
];

export const MAX_FILE_BYTES = 3 * 1024 * 1024 * 1024; // 3 GB per file

export function isValidSessionName(session: string): boolean {
  return SESSION_RE.test(session) && !session.includes("..");
}

export function isValidCaptureFile(name: string): boolean {
  return !name.includes("/") && !name.includes("..") &&
    FILE_RES.some((re) => re.test(name));
}

export interface ChunkStatus {
  ts: string;
  fisheye: boolean;
  thermal: boolean;
  mmwave: boolean;
  imu: boolean;
  frames: boolean;
  gps: boolean;
}

export interface SessionCheck {
  chunks: ChunkStatus[];
  extras: string[]; // valid non-chunk files (cfg/env/log)
  rejected: string[]; // filenames that are not capture files
  ok: boolean; // every chunk has at least fisheye + mmwave
  problems: string[];
}

/** Client-side structural validation of a picked session directory:
    pair chunk files by timestamp; fisheye + mmwave are required per chunk
    (the pipeline's `require` set), the rest are reported honestly. */
export function checkSession(fileNames: string[]): SessionCheck {
  const chunks = new Map<string, ChunkStatus>();
  const extras: string[] = [];
  const rejected: string[] = [];
  for (const name of fileNames) {
    if (!isValidCaptureFile(name)) {
      rejected.push(name);
      continue;
    }
    const m = name.match(/^(\w+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\./);
    if (!m) {
      extras.push(name);
      continue;
    }
    const [, kind, ts] = m as unknown as [string, keyof ChunkStatus & string, string];
    const c = chunks.get(ts) ?? {
      ts, fisheye: false, thermal: false, mmwave: false,
      imu: false, frames: false, gps: false,
    };
    if (kind in c) (c as unknown as Record<string, boolean>)[kind] = true;
    chunks.set(ts, c);
  }
  const list = [...chunks.values()].sort((a, b) => a.ts.localeCompare(b.ts));
  const problems: string[] = [];
  for (const c of list) {
    if (!c.fisheye) problems.push(`${c.ts}: missing fisheye mp4`);
    if (!c.mmwave) problems.push(`${c.ts}: missing mmwave csv`);
    if (!c.thermal) problems.push(`${c.ts}: no thermal (allowed, will be honest-absent)`);
  }
  const ok = list.length > 0 &&
    list.every((c) => c.fisheye && c.mmwave);
  if (list.length === 0) problems.unshift("no capture chunks recognised");
  return { chunks: list, extras, rejected, ok, problems };
}

export function uploadKey(session: string, file: string): string | null {
  if (!isValidSessionName(session) || !isValidCaptureFile(file)) return null;
  return `incoming/${session}/${file}`;
}

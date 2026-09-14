import { GetObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

/* Server-side R2 access for the PRIVATE clip bucket. The browser never sees
   credentials: /api/assets/<path> validates the path, presigns a GET and
   302-redirects. Signing timestamps are quantised to the hour so the same
   asset yields the SAME signed URL within a window — that keeps browser
   caching effective across a scrubbing session. */

export const SIGN_WINDOW_S = 3600; // quantisation step
export const SIGN_TTL_S = 7200; // must exceed the window by a safe margin

export function isR2Configured(): boolean {
  return Boolean(
    process.env.R2_ACCOUNT_ID &&
      process.env.R2_ACCESS_KEY_ID &&
      process.env.R2_SECRET_ACCESS_KEY &&
      process.env.R2_BUCKET,
  );
}

let client: S3Client | null = null;

export function r2Client(): S3Client {
  if (!client) {
    client = new S3Client({
      region: "auto",
      endpoint: `https://${process.env.R2_ACCOUNT_ID}.r2.cloudflarestorage.com`,
      credentials: {
        accessKeyId: process.env.R2_ACCESS_KEY_ID!,
        secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
      },
    });
  }
  return client;
}

/** Quantised signing date: floor(now / window). Pure — unit-tested. */
export function quantisedSigningDate(
  nowMs: number,
  windowS = SIGN_WINDOW_S,
): Date {
  const windowMs = windowS * 1000;
  return new Date(Math.floor(nowMs / windowMs) * windowMs);
}

const SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._=-]*$/;
const EXTENSIONS = new Set(["json", "jpg", "png"]);

/** Only bundle-shaped keys may be signed: shallow, safe segments, known
    extensions. Anything else is rejected before touching R2. Pure. */
export function isValidAssetPath(path: string): boolean {
  const segments = path.split("/");
  if (segments.length < 2 || segments.length > 4) return false;
  if (!segments.every((s) => SEGMENT.test(s))) return false;
  const last = segments[segments.length - 1]!;
  const ext = last.split(".").pop() ?? "";
  return EXTENSIONS.has(ext);
}

export async function signAssetUrl(
  path: string,
  opts?: { distinct?: boolean },
): Promise<string> {
  return getSignedUrl(
    r2Client(),
    new GetObjectCommand({
      Bucket: process.env.R2_BUCKET!,
      Key: path,
      // json-mode (CORS consumers) signs a response-cache-control override:
      // the param is part of the signature, so CORS-mode URLs are DISTINCT
      // from redirect-mode URLs — a no-CORS <img> response can never be
      // served from cache to a CORS fetch (the Arc "missing ACAO" bug) —
      // and the assets gain explicit immutable caching.
      ...(opts?.distinct
        ? { ResponseCacheControl: "private, max-age=3000, immutable" }
        : {}),
    }),
    { expiresIn: SIGN_TTL_S, signingDate: quantisedSigningDate(Date.now()) },
  );
}

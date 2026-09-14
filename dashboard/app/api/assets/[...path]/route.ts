import { isR2Configured, isValidAssetPath, signAssetUrl } from "@/lib/r2";
import { NextResponse } from "next/server";

/* Redirects a bundle asset path to a presigned URL on the private R2 bucket.
   Quantised signing (lib/r2) keeps the target URL stable within the hour, so
   browsers cache both the redirect and the image response. */

export async function GET(
  req: Request,
  ctx: { params: Promise<{ path: string[] }> },
) {
  if (!isR2Configured()) {
    return NextResponse.json({ error: "R2 not configured" }, { status: 503 });
  }
  const { path } = await ctx.params;
  const key = path.join("/");
  if (!isValidAssetPath(key)) {
    return NextResponse.json({ error: "invalid asset path" }, { status: 400 });
  }
  // json mode: the client fetches the signed URL same-origin (cookies
  // intact) and then loads the asset DIRECTLY from R2 — a clean first-party
  // CORS request. The 302 path is kept for plain <img> loads, but browsers
  // serialise the post-redirect Origin as "null" on cross-origin redirects,
  // which the bucket's specific-origin CORS policy rejects — so anything
  // needing CORS (canvas pixels, fetch) must use ?json=1. json-mode URLs
  // are signed distinct (see signAssetUrl) to avoid cross-mode cache hits.
  if (new URL(req.url).searchParams.has("json")) {
    return NextResponse.json(
      { url: await signAssetUrl(key, { distinct: true }) },
      { headers: { "Cache-Control": "private, max-age=3000" } },
    );
  }
  const url = await signAssetUrl(key);
  return NextResponse.redirect(url, {
    status: 302,
    headers: { "Cache-Control": "private, max-age=3000" },
  });
}

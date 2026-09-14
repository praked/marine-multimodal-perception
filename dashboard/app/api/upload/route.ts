import { MAX_FILE_BYTES, uploadKey } from "@/lib/upload";
import { isR2Configured, r2Client } from "@/lib/r2";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { NextRequest, NextResponse } from "next/server";

/* Presign a PUT for one capture-session file into the private bucket's
   `incoming/<session>/` prefix. Path + extension are whitelisted
   (lib/upload), size is capped, and the whole route sits behind the
   password-gate middleware like every other page and API. The signed URL
   pins the exact key and expires in 15 minutes. */

export async function POST(req: NextRequest) {
  if (!isR2Configured()) {
    return NextResponse.json({ error: "R2 not configured" }, { status: 503 });
  }
  let body: { session?: string; file?: string; size?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  const { session, file, size } = body;
  if (typeof session !== "string" || typeof file !== "string" ||
      typeof size !== "number" || !Number.isFinite(size)) {
    return NextResponse.json({ error: "session, file, size required" }, { status: 400 });
  }
  if (size <= 0 || size > MAX_FILE_BYTES) {
    return NextResponse.json(
      { error: `file size out of range (max ${MAX_FILE_BYTES} bytes)` },
      { status: 400 });
  }
  const key = uploadKey(session, file);
  if (!key) {
    return NextResponse.json(
      { error: "not a valid capture session/file name" }, { status: 400 });
  }
  const url = await getSignedUrl(
    r2Client(),
    new PutObjectCommand({
      Bucket: process.env.R2_BUCKET!,
      Key: key,
      ContentLength: size,
    }),
    { expiresIn: 900 },
  );
  return NextResponse.json({ url, key });
}

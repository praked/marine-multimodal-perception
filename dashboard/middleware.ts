import { GATE_COOKIE, gateToken } from "@/lib/gate";
import { NextResponse, type NextRequest } from "next/server";

export async function middleware(req: NextRequest) {
  const password = process.env.DASHBOARD_PASSWORD;
  if (!password) return NextResponse.next(); // gate off (CI, demo, local)

  const { pathname } = req.nextUrl;
  if (pathname === "/login") return NextResponse.next();
  // PWA plumbing holds no data: the worker script, manifest and icons must
  // be fetchable before/without the cookie (browsers fetch the worker with
  // credentials, but a manifest/icon request from the OS installer may not).
  if (
    pathname === "/sw.js" ||
    pathname === "/manifest.webmanifest" ||
    pathname.startsWith("/icons/")
  ) {
    return NextResponse.next();
  }

  const cookie = req.cookies.get(GATE_COOKIE)?.value;
  if (cookie && cookie === (await gateToken(password))) {
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "unauthorised" }, { status: 401 });
  }
  const login = req.nextUrl.clone();
  login.pathname = "/login";
  login.search = "";
  return NextResponse.redirect(login);
}

export const config = {
  // Everything except Next internals + favicon. Note /demo/* (the committed
  // demo bundle) is deliberately gated too: hidden means hidden.
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};

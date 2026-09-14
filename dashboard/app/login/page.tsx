import { GATE_COOKIE, gateToken } from "@/lib/gate";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { Waves } from "lucide-react";

async function login(formData: FormData) {
  "use server";
  const password = process.env.DASHBOARD_PASSWORD;
  const given = formData.get("password");
  if (!password || typeof given !== "string" || given !== password) {
    redirect("/login?e=1");
  }
  (await cookies()).set(GATE_COOKIE, await gateToken(password), {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  redirect("/clips");
}

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ e?: string }>;
}) {
  const { e } = await searchParams;
  return (
    <div className="flex h-full items-center justify-center bg-surface-2">
      <form
        action={login}
        className="w-80 rounded-sm border border-border bg-surface-1 p-6"
      >
        <div className="mb-4 flex items-center gap-2">
          <span className="flex h-8 w-8 items-center justify-center rounded-sm bg-seeblau-100 text-inverse">
            <Waves size={17} aria-hidden />
          </span>
          <div>
            <div className="text-sm font-semibold tracking-tight">
              ASVProject
            </div>
            <div className="font-mono text-[9px] uppercase tracking-[0.28em] text-subtle">
              obstacle detection
            </div>
          </div>
        </div>
        <p className="mb-3 text-xs text-muted">
          This dashboard holds unpublished research data. Enter the team
          password to continue.
        </p>
        <label
          htmlFor="password"
          className="font-mono text-[10px] uppercase tracking-wider text-subtle"
        >
          password
        </label>
        <input
          id="password"
          name="password"
          type="password"
          autoFocus
          required
          className="mt-1 w-full rounded-sm border border-border bg-surface-2 px-2 py-1.5 text-sm outline-none focus:border-accent"
        />
        {e && (
          <p className="mt-2 text-xs text-status-serious">
            Wrong password — try again.
          </p>
        )}
        <button
          type="submit"
          className="mt-4 w-full rounded-sm bg-seeblau-100 py-1.5 text-sm font-medium text-inverse hover:bg-seeblau-deep"
        >
          Enter
        </button>
      </form>
    </div>
  );
}

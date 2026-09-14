"use client";

import { SyncIndicator } from "@/components/shell/SyncIndicator";
import { isSupabaseConfigured } from "@/lib/data";
import { Moon, Sun } from "lucide-react";
import { useEffect, useState, useSyncExternalStore } from "react";

const THEME_EVENT = "asvproject-theme";

function subscribeTheme(cb: () => void) {
  window.addEventListener(THEME_EVENT, cb);
  return () => window.removeEventListener(THEME_EVENT, cb);
}

function readTheme(): string {
  return document.documentElement.dataset.theme ?? "light";
}

function ThemeToggle() {
  const theme = useSyncExternalStore(subscribeTheme, readTheme, () => "light");
  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("asvproject.theme", next);
    } catch {}
    window.dispatchEvent(new Event(THEME_EVENT));
  }
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
      className="flex h-8 w-8 items-center justify-center rounded-sm text-muted hover:bg-surface-3 hover:text-foreground"
    >
      {theme === "dark" ? (
        <Sun size={16} aria-hidden />
      ) : (
        <Moon size={16} aria-hidden />
      )}
    </button>
  );
}

function Clock() {
  const [now, setNow] = useState<string>("");
  useEffect(() => {
    const tick = () =>
      setNow(new Date().toLocaleTimeString("en-GB", { hour12: false }));
    const first = setTimeout(tick, 0);
    const id = setInterval(tick, 1000);
    return () => {
      clearTimeout(first);
      clearInterval(id);
    };
  }, []);
  return (
    <span className="font-mono text-xs tabular-nums text-subtle">{now}</span>
  );
}

export function TopStrip() {
  const supabase = isSupabaseConfigured();
  return (
    <header className="flex h-12 shrink-0 items-center gap-4 border-b border-border bg-surface-1 px-4">
      <div className="flex items-baseline gap-2">
        <span className="text-sm font-semibold tracking-tight">ASVProject</span>
        <span className="font-mono text-[10px] uppercase tracking-[0.28em] text-subtle">
          obstacle detection
        </span>
      </div>
      <div className="h-5 w-px bg-border" aria-hidden />
      <span
        className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.28em] text-subtle"
        title={
          supabase
            ? "Reading from Supabase"
            : "No Supabase configured — reading the bundled demo data"
        }
      >
        <span
          aria-hidden
          className={
            supabase
              ? "live-pip h-1.5 w-1.5 rounded-full bg-status-good"
              : "h-1.5 w-1.5 rounded-full bg-seeblau-65"
          }
        />
        {supabase ? "supabase" : "demo bundle"}
      </span>
      <div className="ml-auto flex items-center gap-3">
        <SyncIndicator />
        <Clock />
        <ThemeToggle />
      </div>
    </header>
  );
}

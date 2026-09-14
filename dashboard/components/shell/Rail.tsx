"use client";

import { cn } from "@/lib/cn";
import { Crop, Film, HardDriveDownload, Orbit, SquarePen, SunMoon, Waves } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

interface NavItem {
  href: string;
  label: string;
  icon: typeof Film;
  match: (pathname: string) => boolean;
}

const NAV: NavItem[] = [
  {
    href: "/clips",
    label: "Clips",
    icon: Film,
    match: (p) => p === "/" || p.startsWith("/clips"),
  },
  {
    href: "/annotate",
    label: "Annotate",
    icon: SquarePen,
    match: (p) => p.startsWith("/annotate"),
  },
  {
    href: "/crops",
    label: "Boat crops",
    icon: Crop,
    match: (p) => p.startsWith("/crops"),
  },
  {
    href: "/datamap",
    label: "Data map",
    icon: SunMoon,
    match: (p) => p.startsWith("/datamap"),
  },
  {
    href: "/replay",
    label: "Mission replay",
    icon: Orbit,
    match: (p) => p.startsWith("/replay"),
  },
  {
    href: "/offline",
    label: "Offline packs",
    icon: HardDriveDownload,
    match: (p) => p.startsWith("/offline"),
  },
];

export function Rail() {
  const pathname = usePathname();
  return (
    <nav
      aria-label="Primary"
      className="flex w-14 shrink-0 flex-col items-center gap-1 border-r border-border bg-surface-1 py-3"
    >
      <Link
        href="/clips"
        aria-label="ASVProject home"
        className="mb-3 flex h-9 w-9 items-center justify-center rounded-sm bg-seeblau-100 text-inverse"
      >
        <Waves size={20} strokeWidth={2} aria-hidden />
      </Link>
      {NAV.map((item) => {
        const active = item.match(pathname);
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-label={item.label}
            aria-current={active ? "page" : undefined}
            className={cn(
              "relative flex h-10 w-10 items-center justify-center rounded-sm text-muted hover:bg-surface-3 hover:text-foreground",
              active && "bg-surface-3 text-accent-strong",
            )}
          >
            {active && (
              <span
                aria-hidden
                className="absolute left-[-9px] top-1.5 h-7 w-0.5 rounded-full bg-accent"
              />
            )}
            <Icon size={18} strokeWidth={active ? 2 : 1.5} aria-hidden />
            <span className="sr-only">{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

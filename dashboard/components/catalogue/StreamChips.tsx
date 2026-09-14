import { cn } from "@/lib/cn";
import type { Streams } from "@/lib/types";

const STREAMS: { key: keyof Streams; label: string }[] = [
  { key: "fisheye", label: "fisheye" },
  { key: "thermal", label: "thermal" },
  { key: "radar", label: "radar" },
  { key: "imu", label: "imu" },
  { key: "seg", label: "seg" },
  { key: "sectors", label: "sectors" },
];

/** Present/absent stream chips. Absence is information (e.g. radar-dead
    outings), so missing streams render struck-through, not hidden. */
export function StreamChips({ streams }: { streams: Streams }) {
  return (
    <div className="flex flex-wrap gap-1">
      {STREAMS.map(({ key, label }) => {
        const on = streams[key];
        return (
          <span
            key={key}
            className={cn(
              "rounded-sm border px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider",
              on
                ? "border-accent bg-surface-2 text-accent-strong"
                : "border-border text-subtle line-through",
            )}
            title={on ? `${label} stream present` : `${label} stream absent`}
          >
            {label}
          </span>
        );
      })}
    </div>
  );
}

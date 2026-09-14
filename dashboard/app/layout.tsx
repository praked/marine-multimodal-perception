import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { PwaRegister } from "@/components/shell/PwaRegister";
import { Rail } from "@/components/shell/Rail";
import { TopStrip } from "@/components/shell/TopStrip";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

export const metadata: Metadata = {
  title: "ASVProject · Obstacle Detection",
  description:
    "Data visualisation and annotation dashboard for the ASVProject obstacle-detection payload (Institution One).",
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, title: "ASVProject", statusBarStyle: "default" },
};

export const viewport = { themeColor: "#00a9e0" };

const themeInit = `
try {
  var q = new URLSearchParams(location.search).get("theme");
  var t = q === "light" || q === "dark" ? q : localStorage.getItem("asvproject.theme");
  if (!t) t = matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  document.documentElement.dataset.theme = t;
} catch (e) {}
`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en-GB" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body className="flex h-screen overflow-hidden">
        <PwaRegister />
        <Rail />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopStrip />
          <main className="min-h-0 flex-1 overflow-auto">{children}</main>
        </div>
      </body>
    </html>
  );
}

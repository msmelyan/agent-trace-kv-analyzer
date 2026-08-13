import type { Metadata } from "next";
import { headers } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  return {
    metadataBase: new URL(origin),
    title: "AgentX Trace Atlas — Coding-Agent Workload Analysis",
    description: "A quantitative reconstruction of 393 Claude Code sessions and 98,827 model requests from the SemiAnalysis AgentX corpus.",
    openGraph: {
      title: "AgentX Trace Atlas",
      description: "What 98,827 coding-agent requests reveal about context, KV reuse, timing, and subagent concurrency.",
      type: "website",
      images: [{ url: `${origin}/agentx-social.png`, width: 1200, height: 630, alt: "AgentX Trace Atlas" }],
    },
    twitter: { card: "summary_large_image", title: "AgentX Trace Atlas", description: "A realistic coding-agent workload, measured.", images: [`${origin}/agentx-social.png`] },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={`${geistSans.variable} ${geistMono.variable}`}>{children}</body></html>;
}

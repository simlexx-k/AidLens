import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = { title: "AidLens — Development Evidence Intelligence", description: "Search and compare evidence from development program evaluations." };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><header className="site-header"><Link href="/" className="brand" aria-label="AidLens home"><span className="brand-mark">A</span><span>AidLens</span></Link><nav><Link href="/search">Evidence search</Link><Link href="/corpus">Corpus</Link><a href="http://localhost:8000/docs">API</a></nav></header>{children}<footer><strong>AidLens</strong><span>Development Evidence Intelligence</span></footer></body></html>;
}

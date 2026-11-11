import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Blue Bikes — Bluebikes tracker",
  description: "Markov model of Bluebikes station bike counts on the Boston map.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

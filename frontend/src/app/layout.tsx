import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "X-Fusion | Clinical AI Workstation",
  description: "AI-assisted chest X-ray screening and analysis.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light">
      <body className={`${inter.className} min-h-screen bg-background font-sans antialiased text-foreground selection:bg-primary/20`}>
        {children}
      </body>
    </html>
  );
}


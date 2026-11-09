import type { Metadata } from "next";
import "./globals.css";


export const metadata: Metadata = {
  title: "X-Fusion | Tri-modal research prototype",
  description: "Research chest X-ray analysis with observed clinical notes and patient attributes.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="light">
      <body className="min-h-screen bg-background font-sans antialiased text-foreground selection:bg-primary/20">
        {children}
      </body>
    </html>
  );
}


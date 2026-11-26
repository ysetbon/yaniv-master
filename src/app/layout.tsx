import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { Fraunces, Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

const fraunces = Fraunces({
  variable: "--font-display",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Yaniv",
  description: "Play Yaniv against a neural network whose weights were written by hand.",
};

// viewport-fit=cover lets the table run under a phone's notch and home bar; the page pads itself with the
// safe-area insets (globals.css), so nothing playable sits under them.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#120e0b",
  colorScheme: "dark",
};

// A phone with "Desktop site" switched on (Chrome, Samsung Internet, Safari's "Request Desktop Website")
// ignores the viewport tag above and lays the page out about 980px wide, then shrinks it onto the screen:
// tiny text and small cards. This runs before the page is drawn: on a touch screen whose short side is
// under 600px, if the window is much wider than the screen, it zooms the page back to the phone's real
// size and hands the true screen height to the layout as --app-h (globals.css, .app-frame). It runs
// again when the phone turns. With Desktop site off it changes nothing.
const FIT_PHONE = `(function () {
  var root = document.documentElement;
  function fit() {
    var sw = screen.width, sh = screen.height, w = window.innerWidth, h = window.innerHeight;
    var touch = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
    var across = w > h ? Math.max(sw, sh) : Math.min(sw, sh);
    var zoom = touch && Math.min(sw, sh) < 600 && across > 0 ? w / across : 1;
    if (zoom > 1.25) {
      root.style.zoom = String(zoom);
      root.style.setProperty("--app-h", h / zoom + "px");
    } else {
      root.style.removeProperty("zoom");
      root.style.removeProperty("--app-h");
    }
  }
  fit();
  window.addEventListener("resize", fit);
})();`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    // The head script sets a style on <html> before React loads; that is expected, not a mismatch.
    <html lang="en" className={`${outfit.variable} ${fraunces.variable} h-full antialiased`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: FIT_PHONE }} />
      </head>
      <body className="min-h-full bg-[#120e0b] text-[#f6edd9]">{children}</body>
    </html>
  );
}

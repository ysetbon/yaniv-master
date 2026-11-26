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
// tiny text and small cards. This runs before the page is drawn and zooms the page back to the phone's
// real size, handing the true screen height to the layout as --app-h (globals.css, .app-frame).
//
// Two ways to tell, since browsers differ in what they report in desktop mode:
// 1. the screen: a touch screen whose short side is under 600px, in a window much wider than the screen;
// 2. the browser's own shrink: the page is shown at under 80% scale.
// Both need a window at least 700px wide, the width desktop mode lays pages out at (about 980px).
//    The shrink is read once per window width, so pinching to zoom later does not change it.
// It runs again when the phone turns. With Desktop site off neither test passes and nothing changes.
// Open the page with ?debug to see what the phone reports and what was decided.
const FIT_PHONE = `(function () {
  var root = document.documentElement;
  var debug = location.search.indexOf("debug") >= 0;
  var shrinkFor = 0, shrinkZoom = 1;
  var info = {};
  function touchScreen() {
    var mm = window.matchMedia;
    return (navigator.maxTouchPoints || 0) > 0 ||
      !!(mm && (mm("(pointer: coarse)").matches || mm("(any-pointer: coarse)").matches));
  }
  function fit() {
    var sw = screen.width, sh = screen.height, w = window.innerWidth, h = window.innerHeight;
    var vv = window.visualViewport, scale = vv ? vv.scale : 1;
    var touch = touchScreen();
    var across = w > h ? Math.max(sw, sh) : Math.min(sw, sh);
    var byScreen = touch && w >= 700 && Math.min(sw, sh) < 600 && across > 0 ? w / across : 1;
    if (shrinkFor !== w) {
      shrinkFor = w;
      shrinkZoom = touch && w >= 700 && scale > 0 && scale < 0.8 ? 1 / scale : 1;
    }
    var zoom = byScreen > 1.25 ? byScreen : shrinkZoom > 1.25 ? shrinkZoom : 1;
    if (zoom > 1.25) {
      root.style.zoom = String(zoom);
      root.style.setProperty("--app-h", h / zoom + "px");
    } else {
      root.style.removeProperty("zoom");
      root.style.removeProperty("--app-h");
    }
    info = { build: "${process.env.NEXT_PUBLIC_BUILD ?? "dev"}", screen: sw + "x" + sh, window: w + "x" + h,
      scale: Math.round(scale * 1000) / 1000, dpr: window.devicePixelRatio, touch: touch,
      touchPoints: navigator.maxTouchPoints || 0, byScreen: Math.round(byScreen * 100) / 100,
      byShrink: Math.round(shrinkZoom * 100) / 100, zoom: Math.round(zoom * 100) / 100 };
    if (debug) show();
  }
  function show() {
    if (!document.body) return;
    var box = document.getElementById("fit-debug");
    if (!box) {
      box = document.createElement("pre");
      box.id = "fit-debug";
      box.style.cssText = "position:fixed;left:4px;bottom:4px;z-index:99999;margin:0;padding:6px 8px;" +
        "background:#fff;color:#000;font:12px/1.35 monospace;border-radius:6px;opacity:.92;pointer-events:none";
      document.body.appendChild(box);
    }
    box.textContent = Object.keys(info).map(function (k) { return k + ": " + info[k]; }).join("\\n");
  }
  fit();
  window.addEventListener("resize", fit);
  document.addEventListener("DOMContentLoaded", fit);
  window.addEventListener("load", function () { shrinkFor = 0; fit(); });
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

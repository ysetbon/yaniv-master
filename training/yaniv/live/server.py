"""A local web server for the live dashboard. Standard library only."""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
import webbrowser
from dataclasses import asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..backend import Device
from ..train import MODEL_PATH, REPORT_PATH, ROOT, TrainConfig, run_training
from .hub import Hub, HubObserver
from .showcase import Showcase

PAGE = Path(__file__).with_name("index.html")


def make_handler(hub: Hub, showcase: Showcase):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # keep the console for training output
            pass

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, json.dumps(hub.snapshot()).encode(), "application/json")
            elif self.path == "/api/events":
                self._events()
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/control":
                self._send(404, b"not found", "text/plain")
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                update = json.loads(self.rfile.read(length) or b"{}")
                control = showcase.set_control(update)
            except (ValueError, TypeError) as exc:
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")
                return
            self._send(200, json.dumps(control).encode(), "application/json")

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.subscribe()
            try:
                self.wfile.write(b"retry: 1500\n\n")
                self.wfile.flush()
                while True:
                    try:
                        message = q.get(timeout=15)
                        self.wfile.write(f"data: {message}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)

    return Handler


def run_live(cfg: TrainConfig, device: Device, port: int, open_browser: bool, export: bool) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / stamp
    config = {**asdict(cfg), "run_dir": str(run_dir), "export": export}
    hub = Hub(config, device.label(), device.note, device.is_gpu)
    showcase = Showcase(hub)

    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(hub, showcase))
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    def train():
        hub.update("status", "status", "training")
        try:
            run_training(
                cfg,
                HubObserver(hub),
                device,
                model_path=MODEL_PATH if export else None,
                report_path=REPORT_PATH if export else None,
                run_dir=run_dir,
            )
            hub.update("finished", "finished", time.time())
            hub.update("status", "status", "done")
            hub.update("phase", "phase", {"name": "done", "label": "Training finished. The table keeps playing."})
        except Exception as exc:
            hub.log(traceback.format_exc())
            hub.update("error", "error", f"{type(exc).__name__}: {exc}")
            hub.update("status", "status", "error")

    threading.Thread(target=train, daemon=True, name="training").start()
    showcase.start()
    print(f"\nLive dashboard: {url}\nTraining on {device.label()}. Press Ctrl+C to stop.\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()

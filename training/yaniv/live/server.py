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
from ..train import MODEL_PATH, REPORT_PATH, ROOT, TrainConfig, export_best, run_training
from .hub import Hub, HubObserver
from .showcase import Showcase

PAGE = Path(__file__).with_name("index.html")
RUNS = ROOT / "runs"


def past_runs(current: Path) -> list[dict]:
    """Finished runs under runs/, newest first, for comparing across runs."""
    out = []
    if not RUNS.exists():
        return out
    for folder in sorted(RUNS.iterdir(), reverse=True):
        report_file = folder / "report.json"
        if folder.name.startswith(".") or folder == current or not report_file.exists():
            continue
        try:
            report = json.loads(report_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rounds = report.get("round_record_vs_benchmark", {})
        out.append({
            "name": folder.name,
            "device": report.get("device", ""),
            "seconds": report.get("seconds"),
            "generations": report.get("self_play_iterations"),
            "best_generation": report.get("best_generation"),
            "match_win_rate": report.get("match_win_rate_vs_benchmark"),
            "matches": report.get("match_record_vs_benchmark"),
            "round_win_rate": report.get("round_win_rate_vs_benchmark"),
            "hero_avg": rounds.get("hero_avg"),
            "villain_avg": rounds.get("villain_avg"),
            "recipe": report.get("recipe") if report.get("recipe") in ("genetic", "search")
            else "replay" if (report.get("config") or {}).get("replay_capacity") else "original",
        })
    return out[:20]


def make_handler(hub: Hub, showcase: Showcase, run_dir: Path):
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

        def _json(self, code: int, payload) -> None:
            self._send(code, json.dumps(payload).encode(), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(200, hub.snapshot())
            elif self.path == "/api/runs":
                self._json(200, past_runs(run_dir))
            elif self.path == "/api/events":
                self._events()
            else:
                self._send(404, b"not found", "text/plain")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data

        def do_POST(self):
            try:
                body = self._body()
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            if self.path == "/api/control":
                try:
                    self._json(200, showcase.set_control(body))
                except (ValueError, TypeError) as exc:
                    self._json(400, {"error": str(exc)})
            elif self.path == "/api/stop":
                self._stop()
            elif self.path == "/api/export":
                self._export()
            else:
                self._send(404, b"not found", "text/plain")

        def _stop(self):
            if hub.state["status"] != "training":
                self._json(409, {"error": "Training is not running."})
                return
            hub.stop_requested = True
            hub.update("stopping", "stopping", True)
            hub.toast("info", "Training will stop after the current generation, then test and save the best one.")
            self._json(200, {"ok": True})

        def _export(self):
            report = hub.state.get("report")
            if not report:
                self._json(409, {"error": "The best generation is known once training finishes."})
                return
            best_index = report["best_generation"]
            net = hub.net(best_index)
            if net is None:
                self._json(500, {"error": f"Generation {best_index} is not in memory."})
                return
            meta = {
                "match_win_rate_vs_benchmark": report["match_win_rate_vs_benchmark"],
                "self_play_games": report["self_play_games"],
                "expert_decisions": report["expert_decisions"],
                "generation": best_index,
            }
            try:
                export_best(net, report, meta, MODEL_PATH, REPORT_PATH)
            except OSError as exc:
                self._json(500, {"error": f"Could not write the model: {exc}"})
                return
            info = {"generation": best_index, "model": str(MODEL_PATH), "at": time.time()}
            hub.update("exported", "exported", info)
            hub.toast("good", f"Gen {best_index} is now the opponent in the game. Restart `npm run dev` or reload the game page.")
            self._json(200, info)

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


class DashboardServer(ThreadingHTTPServer):
    """The dashboard's server. A browser tab that closes, reloads or sleeps drops its
    connection mid-request; that is normal, so it is not printed as an error."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        import sys

        if isinstance(sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def run_live(cfg: TrainConfig, device: Device, port: int, open_browser: bool, export: bool, tables: int = 6) -> None:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS / stamp
    config = {**asdict(cfg), "run_dir": str(run_dir), "export": export}
    hub = Hub(config, device.label(), device.note, device.is_gpu)
    showcase = Showcase(hub, tables, ("pop", "counter") if cfg.recipe == "genetic" else ("auto", "auto"))

    try:
        server = DashboardServer(("127.0.0.1", port), make_handler(hub, showcase, run_dir))
    except OSError:
        raise SystemExit(f"Port {port} is in use. Close the other dashboard or pass --port {port + 1}.")
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
            if export:
                hub.update("exported", "exported", {"generation": hub.state["report"]["best_generation"],
                                                    "model": str(MODEL_PATH), "at": time.time()})
            hub.update("finished", "finished", time.time())
            hub.update("status", "status", "done")
            hub.update("phase", "phase", {"name": "done", "label": "Training finished. The table keeps playing."})
        except Exception as exc:
            hub.log(traceback.format_exc())
            hub.update("error", "error", f"{type(exc).__name__}: {exc}")
            hub.update("status", "status", "error")
            hub.toast("bad", f"Training stopped: {type(exc).__name__}: {exc}")

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

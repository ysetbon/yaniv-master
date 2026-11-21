"""Shared run state and a fan-out of events to every open dashboard tab."""

from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque

from ..network import MLP
from ..train import Observer


class Hub:
    def __init__(self, config: dict, device: str, device_note: str, is_gpu: bool):
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self.nets: dict[int, MLP] = {}
        self.started = time.time()
        self.state: dict = {
            "config": config,
            "device": {"label": device, "note": device_note, "gpu": is_gpu},
            "status": "starting",
            "phase": {"name": "starting", "label": "Starting up"},
            "progress": {"name": "", "done": 0, "total": 0},
            "history": [],
            "generations": [],
            "ladder": [],
            "logs": [],
            "board": None,
            "showcase": None,
            "report": None,
            "paths": {},
            "error": None,
            "started": self.started,
            "finished": None,
        }
        self._logs: deque = deque(maxlen=300)

    # ---- pub/sub -------------------------------------------------------
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, kind: str, payload) -> None:
        message = json.dumps({"type": kind, "data": payload})
        with self._lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(message)
            except queue.Full:
                pass  # a stalled tab drops events; it resyncs from /api/state on reconnect

    def snapshot(self) -> dict:
        with self._lock:
            state = dict(self.state)
            state["logs"] = list(self._logs)
            state["now"] = time.time()
            return json.loads(json.dumps(state))

    # ---- state updates -------------------------------------------------
    def update(self, kind: str, key: str, value) -> None:
        with self._lock:
            self.state[key] = value
        self.publish(kind, value)

    def append(self, kind: str, key: str, value) -> None:
        with self._lock:
            self.state[key].append(value)
        self.publish(kind, value)

    def log(self, message: str) -> None:
        line = {"t": round(time.time() - self.started, 1), "text": message}
        with self._lock:
            self._logs.append(line)
        self.publish("log", line)

    def add_net(self, index: int, net: MLP) -> None:
        with self._lock:
            self.nets[index] = net

    def net(self, index: int) -> MLP | None:
        with self._lock:
            return self.nets.get(index)

    def latest_generation(self) -> int | None:
        with self._lock:
            return max(self.nets) if self.nets else None


class HubObserver(Observer):
    """Feeds training events into the hub and still prints to the console."""

    def __init__(self, hub: Hub):
        self.hub = hub

    def log(self, message: str) -> None:
        print(message, flush=True)
        self.hub.log(message)

    def phase(self, name: str, label: str) -> None:
        self.hub.update("phase", "phase", {"name": name, "label": label})
        self.hub.update("progress", "progress", {"name": name, "done": 0, "total": 0})

    def progress(self, name: str, done: int, total: int) -> None:
        self.hub.update("progress", "progress", {"name": name, "done": done, "total": total})

    def history(self, entry: dict) -> None:
        self.hub.append("history", "history", entry)

    def generation(self, gen: dict, net: MLP) -> None:
        self.hub.add_net(gen["index"], net)
        self.hub.append("generation", "generations", gen)

    def ladder(self, row: int, col: int, record: dict) -> None:
        self.hub.append("ladder", "ladder", {"row": row, "col": col, **record})

    def done(self, report: dict, paths: dict) -> None:
        self.hub.update("paths", "paths", paths)
        self.hub.update("report", "report", report)

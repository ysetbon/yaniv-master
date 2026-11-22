"""Plays visible matches between generations at a watchable speed."""

from __future__ import annotations

import random
import threading
import time

import numpy as np

from ..engine import (
    MATCH_LIMIT,
    MAX_PLIES,
    Action,
    View,
    apply_action,
    cards_from_mask,
    card_name,
    legal_actions,
    make_view,
    mask_points,
    new_round,
    resolve_showdown,
)
from ..features import action_matrix, state_features
from ..heuristic import choose_greedy, choose_heuristic, choose_random
from .hub import Hub

BOTS = {
    "counter": ("Card counter", choose_heuristic),
    "greedy": ("Greedy dumper", choose_greedy),
    "random": ("Random", choose_random),
}


def describe(view: View, action: Action) -> str:
    if action.kind == "yaniv":
        return f"calls Yaniv on {mask_points(view.my_hand)}"
    thrown = " ".join(card_name(c) for c in cards_from_mask(action.discard_mask))
    if action.draw == "none":
        return f"throws {thrown} and goes out"
    source = "draws from the deck" if action.draw == "stock" else f"takes the {card_name(view.discard[-1])}"
    top = f" ({card_name(action.top)} on top)" if action.discard_mask & (action.discard_mask - 1) else ""
    return f"throws {thrown}{top}, {source}"


class Player:
    def __init__(self, pid: str, label: str, net=None, bot=None):
        self.id = pid
        self.label = label
        self.net = net
        self.bot = bot

    def choose(self, view: View, actions: list[Action], rng) -> tuple[Action, list[dict] | None]:
        if self.bot is not None:
            return self.bot(view, actions, rng), None
        packed_state = np.repeat(state_features(view)[None, :], len(actions), axis=0)
        packed = np.concatenate([packed_state, action_matrix(view, actions)], axis=1)
        scores = self.net.forward(packed)[:, 0].astype(np.float64)
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        order = np.argsort(-scores)
        best = int(order[0])
        top = [
            {"text": describe(view, actions[int(i)]), "score": round(float(scores[i]), 2),
             "p": round(float(probs[i]), 4), "chosen": int(i) == best}
            for i in order[:4]
        ]
        return actions[best], top


class Showcase(threading.Thread):
    def __init__(self, hub: Hub):
        super().__init__(daemon=True, name="showcase")
        self.hub = hub
        self.rng = random.Random()
        self.control = {"delay": 0.9, "paused": False, "a": "auto", "b": "auto"}
        self._changed = threading.Event()
        self._step = threading.Event()
        self.tally: dict[str, dict] = {}
        self.match_no = 0

    # ---- controls from the browser --------------------------------------
    def set_control(self, update: dict) -> dict:
        for key in ("a", "b"):
            if key in update and isinstance(update[key], str):
                if update[key] != self.control[key]:
                    self.control[key] = update[key]
                    self._changed.set()
        if "delay" in update:
            self.control["delay"] = min(5.0, max(0.05, float(update["delay"])))
        if "paused" in update:
            self.control["paused"] = bool(update["paused"])
        if update.get("step"):
            self._step.set()
        self._publish_meta(None)
        return dict(self.control)

    # ---- choosing who plays ---------------------------------------------
    def _player(self, pid: str) -> Player | None:
        if pid in BOTS:
            label, bot = BOTS[pid]
            return Player(pid, label, bot=bot)
        if pid == "best":
            index = self.hub.best_generation()
            if index is not None and self.hub.net(index) is not None:
                return Player(f"gen:{index}", f"Gen {index} (best)", net=self.hub.net(index))
            return None
        if pid.startswith("gen:"):
            index = int(pid[4:])
            net = self.hub.net(index)
            if net is not None:
                return Player(pid, f"Gen {index}", net=net)
        return None

    def _resolve(self) -> tuple[Player, Player, bool]:
        """Seat A and seat B. The flag says the auto pairing is in use."""
        latest = self.hub.latest_generation()
        if latest is None:
            auto = ("counter", "greedy")
        elif latest == 0:
            auto = ("gen:0", "counter")
        else:
            auto = (f"gen:{latest}", f"gen:{latest - 1}")
        a = self.control["a"] if self.control["a"] != "auto" else auto[0]
        b = self.control["b"] if self.control["b"] != "auto" else auto[1]
        pa = self._player(a) or self._player(auto[0])
        pb = self._player(b) or self._player(auto[1])
        dynamic = any(v in ("auto", "best") for v in (self.control["a"], self.control["b"]))
        return pa, pb, dynamic

    def _auto_target(self) -> tuple[str, str]:
        a, b, _ = self._resolve()
        return a.id, b.id

    # ---- publishing ------------------------------------------------------
    def _key(self, a: Player, b: Player) -> str:
        return f"{a.id}|{b.id}"

    def _publish_meta(self, players) -> None:
        payload = {"control": dict(self.control), "tally": {k: {**v, "rounds": list(v["rounds"]), "matches": list(v["matches"])} for k, v in list(self.tally.items())}}
        if players is not None:
            payload["players"] = [{"id": p.id, "label": p.label} for p in players]
        elif self.hub.state.get("showcase"):
            payload["players"] = self.hub.state["showcase"].get("players")
        self.hub.update("showcase", "showcase", payload)

    def _board(self, state, players, round_no, last=None, thinking=None, note=None, pending=None) -> None:
        board = {
            "match": self.match_no,
            "round": round_no,
            "players": [{"id": p.id, "label": p.label} for p in players],
            "hands": [cards_from_mask(state.hands[0]), cards_from_mask(state.hands[1])],
            "points": [mask_points(state.hands[0]), mask_points(state.hands[1])],
            "known": [cards_from_mask(state.known[0]), cards_from_mask(state.known[1])],
            "pile": list(state.discard[-8:]),
            "stock": len(state.stock),
            "scores": list(state.scores),
            "ply": state.ply,
            "turn": state.turn,
            "last": last,
            "thinking": thinking,
            "result": None if state.result is None else state.result.to_json(),
            "note": note,
            "pending": pending,
            "limit": MATCH_LIMIT,
        }
        self.hub.update("board", "board", board)

    # ---- the game loop ---------------------------------------------------
    def _wait(self, seconds: float) -> bool:
        """Sleep while honouring pause. False means the pairing changed."""
        end = time.time() + seconds
        while True:
            if self._changed.is_set():
                return False
            if self.control["paused"]:
                if self._step.is_set():
                    self._step.clear()
                    return True
                end = time.time() + seconds
                time.sleep(0.05)
                continue
            left = end - time.time()
            if left <= 0:
                return True
            time.sleep(min(0.1, left))

    def run(self) -> None:
        while True:
            try:
                self._play_match()
            except Exception as exc:  # keep the table alive whatever happens
                self.hub.log(f"Showcase error: {exc}")
                time.sleep(2)

    def _play_match(self) -> None:
        self._changed.clear()
        a, b, auto = self._resolve()
        players = [a, b]
        key = self._key(a, b)
        tally = self.tally.setdefault(key, {"a": a.label, "b": b.label, "rounds": [0, 0, 0], "matches": [0, 0, 0]})
        self.match_no += 1
        self._publish_meta(players)
        scores = [0, 0]
        leader = self.rng.randrange(2)
        round_no = 0
        while max(scores) < MATCH_LIMIT and round_no < 40:
            round_no += 1
            state = new_round(scores, leader, self.rng)
            self._board(state, players, round_no, note=f"Round {round_no}. {players[leader].label} leads.")
            if not self._wait(self.control["delay"] * 1.5):
                return
            while not state.terminal:
                if state.ply >= MAX_PLIES:
                    resolve_showdown(state)
                    break
                seat = state.turn
                view = make_view(state, seat)
                actions = legal_actions(view)
                if not actions:
                    resolve_showdown(state)
                    break
                action, thinking = players[seat].choose(view, actions, self.rng)
                before = state.hands[seat]
                text = describe(view, action)
                think = None if thinking is None else {"seat": seat, "moves": thinking}
                pending = {"seat": seat, "text": text, "kind": action.kind,
                           "thrown": cards_from_mask(action.discard_mask), "draw": action.draw}
                self._board(state, players, round_no, thinking=think, pending=pending)
                if not self._wait(self.control["delay"] * 0.6):
                    return
                apply_action(state, action)
                drawn = cards_from_mask(state.hands[seat] & ~before)
                last = {
                    "seat": seat,
                    "text": text,
                    "kind": action.kind,
                    "thrown": cards_from_mask(action.discard_mask),
                    "top": action.top,
                    "draw": action.draw,
                    "drawn": drawn[0] if drawn and action.draw != "none" else None,
                }
                self._board(state, players, round_no, last=last, thinking=think)
                if not self._wait(self.control["delay"]):
                    return
            if not state.terminal:
                resolve_showdown(state)
            added = state.result.added
            outcome = 0 if added[0] < added[1] else 1 if added[1] < added[0] else 2
            tally["rounds"][outcome] += 1
            scores = list(state.scores)
            self._board(state, players, round_no, note=self._round_note(state, players))
            self._publish_meta(players)
            if not self._wait(self.control["delay"] * 3):
                return
            if auto and self._auto_target() != (a.id, b.id):
                self.hub.log(f"Showcase switching to the newest generation after round {round_no}")
                return
            leader = 1 - leader
        if scores[0] != scores[1]:
            tally["matches"][0 if scores[0] < scores[1] else 1] += 1
        else:
            tally["matches"][2] += 1
        self._publish_meta(players)
        winner = players[0].label if scores[0] < scores[1] else players[1].label
        self.hub.log(f"Showcase match {self.match_no}: {a.label} {scores[0]} – {b.label} {scores[1]}, {winner} wins")
        self._wait(self.control["delay"] * 4)

    @staticmethod
    def _round_note(state, players) -> str:
        r = state.result
        names = [p.label for p in players]
        if r.reason == "go_out":
            return f"{names[r.winner]} went out for zero."
        if r.reason == "showdown":
            return f"Showdown after {MAX_PLIES} throws: {r.points[0]} to {r.points[1]}."
        caller = names[r.caller]
        if r.assaf:
            return f"{caller} called Yaniv on {r.points[r.caller]} and was caught. Assaf, +30."
        return f"{caller} called Yaniv on {r.points[r.caller]} and won the round."

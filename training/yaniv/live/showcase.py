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


class Showcase:
    """Several tables playing at once, sharing one pairing, speed and pause.

    These exhibition games are separate from the training games, which run
    far too fast to watch. Each table plays its own match with its own deck.
    """

    def __init__(self, hub: Hub, tables: int = 6):
        self.hub = hub
        self.count = max(1, int(tables))
        self.control = {"delay": 0.9, "paused": False, "a": "auto", "b": "auto"}
        self.version = 0  # bumps when the pairing changes, so every table restarts
        self.step_token = 0  # bumps on "next move", so each paused table moves once
        self.tally: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.match_no = 0

    def start(self) -> None:
        for index in range(self.count):
            Table(self, index).start()

    # ---- controls from the browser --------------------------------------
    def set_control(self, update: dict) -> dict:
        with self.lock:
            for key in ("a", "b"):
                if key in update and isinstance(update[key], str) and update[key] != self.control[key]:
                    self.control[key] = update[key]
                    self.version += 1
            if "delay" in update:
                self.control["delay"] = min(5.0, max(0.05, float(update["delay"])))
            if "paused" in update:
                self.control["paused"] = bool(update["paused"])
            if update.get("step"):
                self.step_token += 1
        self.publish_meta(None)
        return dict(self.control)

    # ---- choosing who plays ---------------------------------------------
    def player(self, pid: str) -> Player | None:
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

    def resolve(self) -> tuple[Player, Player, bool]:
        """Seat A and seat B. The flag says the pairing follows new generations."""
        latest = self.hub.latest_generation()
        if latest is None:
            auto = ("counter", "greedy")
        elif latest == 0:
            auto = ("gen:0", "counter")
        else:
            auto = (f"gen:{latest}", f"gen:{latest - 1}")
        a = self.control["a"] if self.control["a"] != "auto" else auto[0]
        b = self.control["b"] if self.control["b"] != "auto" else auto[1]
        pa = self.player(a) or self.player(auto[0])
        pb = self.player(b) or self.player(auto[1])
        dynamic = any(v in ("auto", "best") for v in (self.control["a"], self.control["b"]))
        return pa, pb, dynamic

    def target(self) -> tuple[str, str]:
        a, b, _ = self.resolve()
        return a.id, b.id

    # ---- shared bookkeeping -----------------------------------------------
    def next_match(self) -> int:
        with self.lock:
            self.match_no += 1
            return self.match_no

    def record(self, a: Player, b: Player, rounds: int | None = None, matches: int | None = None) -> None:
        with self.lock:
            tally = self.tally.setdefault(f"{a.id}|{b.id}", {"a": a.label, "b": b.label, "rounds": [0, 0, 0], "matches": [0, 0, 0]})
            if rounds is not None:
                tally["rounds"][rounds] += 1
            if matches is not None:
                tally["matches"][matches] += 1

    def publish_meta(self, players) -> None:
        with self.lock:
            payload = {
                "control": dict(self.control),
                "tables": self.count,
                "tally": {k: {**v, "rounds": list(v["rounds"]), "matches": list(v["matches"])} for k, v in self.tally.items()},
            }
        if players is not None:
            payload["players"] = [{"id": p.id, "label": p.label} for p in players]
        elif self.hub.state.get("showcase"):
            payload["players"] = self.hub.state["showcase"].get("players")
        self.hub.update("showcase", "showcase", payload)

    @staticmethod
    def round_note(state, players) -> str:
        r = state.result
        names = [p.label for p in players]
        if r.reason == "showdown":
            return f"Showdown after {MAX_PLIES} throws: {r.points[0]} to {r.points[1]}."
        caller = names[r.caller]
        if r.assaf:
            return f"{caller} called Yaniv on {r.points[r.caller]} and was caught. Assaf, +30."
        return f"{caller} called Yaniv on {r.points[r.caller]} and won the round."


class Table(threading.Thread):
    def __init__(self, show: Showcase, index: int):
        super().__init__(daemon=True, name=f"table-{index}")
        self.show = show
        self.hub = show.hub
        self.index = index
        self.rng = random.Random()
        self.version = -1
        self.pairing: tuple[str, str] | None = None  # set while following new generations
        self.seen_step = 0
        self.match_no = 0

    def _board(self, state, players, round_no, last=None, thinking=None, note=None, pending=None) -> None:
        self.hub.set_board(self.index, {
            "table": self.index,
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
        })

    def _wait(self, seconds: float) -> bool:
        """Sleep while honouring pause and step. False means the pairing changed."""
        show = self.show
        end = time.time() + seconds
        while True:
            if show.version != self.version:
                return False
            if self.pairing is not None and show.target() != self.pairing:
                return False  # a new generation arrived: every table switches at once
            if show.control["paused"]:
                if show.step_token != self.seen_step:
                    self.seen_step = show.step_token
                    return True
                end = time.time() + seconds
                time.sleep(0.05)
                continue
            self.seen_step = show.step_token
            left = end - time.time()
            if left <= 0:
                return True
            time.sleep(min(0.1, left))

    def run(self) -> None:
        # Stagger the tables so their moves do not land in lockstep.
        time.sleep(self.index * 0.37)
        while True:
            try:
                self._play_match()
            except Exception as exc:  # keep the table alive whatever happens
                self.hub.log(f"Table {self.index + 1} error: {exc}")
                time.sleep(2)

    def _play_match(self) -> None:
        show = self.show
        self.version = show.version
        a, b, dynamic = show.resolve()
        players = [a, b]
        self.pairing = (a.id, b.id) if dynamic else None
        self.match_no = show.next_match()
        show.record(a, b)
        if self.index == 0:
            show.publish_meta(players)
        scores = [0, 0]
        leader = self.rng.randrange(2)
        round_no = 0
        while max(scores) < MATCH_LIMIT and round_no < 40:
            round_no += 1
            state = new_round(scores, leader, self.rng)
            self._board(state, players, round_no, note=f"Round {round_no}. {players[leader].label} leads.")
            if not self._wait(show.control["delay"] * 1.5):
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
                if think is not None and players[seat].id != "gen:0":
                    copy = show.player("gen:0")
                    if copy is not None:
                        ref, _ = copy.choose(view, actions, self.rng)
                        think["reference"] = {"label": "Gen 0", "text": describe(view, ref), "same": ref == action}
                pending = {"seat": seat, "text": text, "kind": action.kind,
                           "thrown": cards_from_mask(action.discard_mask), "draw": action.draw}
                self._board(state, players, round_no, thinking=think, pending=pending)
                if not self._wait(show.control["delay"] * 0.6):
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
                    "drawn": drawn[0] if drawn else None,
                }
                self._board(state, players, round_no, last=last, thinking=think)
                if not self._wait(show.control["delay"]):
                    return
            if not state.terminal:
                resolve_showdown(state)
            added = state.result.added
            show.record(a, b, rounds=0 if added[0] < added[1] else 1 if added[1] < added[0] else 2)
            scores = list(state.scores)
            self._board(state, players, round_no, note=show.round_note(state, players))
            show.publish_meta(players)
            if not self._wait(show.control["delay"] * 3):
                return
            leader = 1 - leader
        show.record(a, b, matches=0 if scores[0] < scores[1] else 1 if scores[1] < scores[0] else 2)
        show.publish_meta(players)
        winner = players[0].label if scores[0] < scores[1] else players[1].label
        self.hub.log(f"Table {self.index + 1}, match {self.match_no}: {a.label} {scores[0]} – {b.label} {scores[1]}, {winner} wins")
        self._wait(show.control["delay"] * 4)

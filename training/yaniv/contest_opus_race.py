"""Team opus, topic `race`: non-call throw/draw decisions once my hand holds 4 cards or fewer (the race to 7).

Play-out probes (alt functions for contest_opus_playout.playout; they read only the hero's view, never ctx's
truth) and, if a class pays, add_race_units(kit). Each record's tag is a JSON string of the position's public
facts, grouped afterwards. Log: results/contest-opus-logs/race.md. Nothing is trained, fitted or searched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .engine import POINTS_OF, cards_from_mask, mask_points, rank_of
from .features import action_key

ROOT = Path(__file__).resolve().parents[2]
_C: dict = {}


def _chooser(name: str):
    if name not in _C:
        from .contest_opus_playout import player
        _C[name] = player(name)
    return _C[name]


def _net(name: str):
    key = "net:" + name
    if key not in _C:
        from .contest_opus_playout import NAMED
        from .network import load_net
        _C[key] = load_net(Path(NAMED.get(name, name)))
    return _C[key]


def _desc(view, act) -> dict:
    """A move's public description."""
    if act.kind == "yaniv":
        return {"k": "Y"}
    cs = cards_from_mask(act.discard_mask)
    kept = view.my_hand & ~act.discard_mask
    kp = mask_points(kept)
    kr = [rank_of(c) for c in cards_from_mask(kept)]
    o = view.discard[-1]
    d = {"k": ("S" if len(cs) == 1 else f"set{len(cs)}"), "tp": POINTS_OF[cs[0]], "thr": sum(POINTS_OF[c] for c in cs),
         "dr": "O" if act.draw == "open" else "D", "kept": kp, "kn": len(kr),
         "kpair": len(set(kr)) < len(kr), "face": POINTS_OF[act.top]}
    if act.draw == "open":
        d["res"] = kp + POINTS_OF[o]
        d["pairs"] = rank_of(o) in kr
    return d


def _facts(view) -> dict:
    from .features import assaf_chance, hidden_lean, opp_min_points
    hand = cards_from_mask(view.my_hand)
    known = cards_from_mask(view.opp_known)
    kpts = sum(POINTS_OF[c] for c in known)
    need = view.opp_size - len(known)
    pts = mask_points(view.my_hand)
    ranks = [rank_of(c) for c in hand]
    return {"n": len(hand), "pts": pts, "on": view.opp_size, "okn": len(known), "okp": kpts,
            "omin": opp_min_points(view), "est": round(kpts + max(0, need) * hidden_lean(view), 1),
            "open": POINTS_OF[view.discard[-1]], "ply": view.ply, "stock": view.stock_size,
            "mykn": bin(view.my_known).count("1"), "pair": len(set(ranks)) < len(ranks),
            "assaf": round(assaf_chance(view), 4) if pts <= 7 else None}


def _record(view, a, b, src):
    if b is None or action_key(b) == action_key(a):
        return None
    f = _facts(view)
    f["A"], f["B"], f["src"] = _desc(view, a), _desc(view, b), src
    f["same_throw"] = a.kind != "yaniv" and b.kind != "yaniv" and a.discard_mask == b.discard_mask
    return b, json.dumps(f)


def _race(view, a) -> bool:
    return a.kind != "yaniv" and view.my_hand.bit_count() <= 4


# ---------------------------------------------------------------- broad maps: another player's move as B
def _alt_player(name):
    def alt(view, actions, a, ctx):
        if not _race(view, a):
            return None
        b = _chooser(name)(view, actions, None)
        if b.kind == "yaniv":
            return None
        return _record(view, a, b, name)
    return alt


alt_trained = _alt_player("trained")
alt_hm4 = _alt_player("hm4")
alt_counter = _alt_player("counter")
alt_hm5 = _alt_player("hm5")


# ---------------------------------------------------------------- own rules
def _hero_best(view, actions, ok, hero="opus"):
    """The hero net's best move among those with ok(action) True (None if none)."""
    from .features import score_actions
    acts = [x for x in actions if x.kind != "yaniv" and ok(x)]
    if not acts:
        return None
    sc = score_actions(_net(hero), view, acts)
    return acts[int(sc.argmax())]


def alt_flipdraw(view, actions, a, ctx):
    """Same throw, the other draw (deck <-> open)."""
    if not _race(view, a):
        return None
    want = "stock" if a.draw == "open" else "open"
    b = next((x for x in actions if x.kind != "yaniv" and x.discard_mask == a.discard_mask and x.draw == want), None)
    return _record(view, a, b, "flip")


def alt_setsingle(view, actions, a, ctx):
    """Set <-> single: if A throws a set, the hero's best single throw; if A throws a single while I hold a
    set (pair+), the hero's best move throwing a set."""
    if not _race(view, a):
        return None
    n = a.discard_mask.bit_count()
    if n > 1:
        b = _hero_best(view, actions, lambda x: x.discard_mask.bit_count() == 1)
    else:
        b = _hero_best(view, actions, lambda x: x.discard_mask.bit_count() > 1)
    return _record(view, a, b, "setsingle")


def alt_otherthrow(view, actions, a, ctx):
    """A different throw of the same size class: the hero's best move whose thrown cards differ in rank from A's
    (single -> another single), any draw."""
    if not _race(view, a):
        return None
    ra = rank_of(a.discard_mask.bit_length() - 1)
    n = a.discard_mask.bit_count()
    b = _hero_best(view, actions, lambda x: x.discard_mask.bit_count() == n
                   and rank_of(x.discard_mask.bit_length() - 1) != ra)
    return _record(view, a, b, "other")


def run(alt: str, opp: str, rounds: int, seed: int, out: str, hero: str = "opus") -> None:
    from .contest_opus_playout import playout
    recs, n, margin = playout(hero, opp, f"training.yaniv.contest_opus_race:{alt}", rounds, seed, processes=2)
    Path(out).write_text(json.dumps({"rounds": n, "margin": margin, "hero": hero, "opp": opp, "alt": alt,
                                     "seed": seed, "recs": recs}))
    print(f"{out}: {len(recs)} records, {n} rounds, margin {margin:+.3f}", flush=True)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    run(a[0], a[1], int(a[2]), int(a[3]), a[4], *(a[5:6]))

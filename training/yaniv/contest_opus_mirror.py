"""Team opus, topic `mirror`: rules that gain against an hm5-like opponent (the rival net is likely built from
handmade5.py too) and do not lose against the trained net.

Part 1: play-out probes (alt functions for contest_opus_playout.playout). Each returns (B, tag) with the tag a JSON
string of public facts of the position and of the two moves, grouped afterwards (scratch script). The truth in ctx
is never read by a rule.
Part 2: add_mirror_units(kit), if a class pays (see results/contest-opus-logs/mirror.md).

Nothing is trained, fitted or searched.
"""

from __future__ import annotations

import json
from pathlib import Path

from .engine import POINTS_OF, cards_from_mask, mask_points, rank_of
from .features import action_key

ROOT = Path(__file__).resolve().parents[2]
_P: dict = {}


def _player(name):
    if name not in _P:
        from .contest_opus_playout import player
        _P[name] = player(name)
    return _P[name]


def _after(view, act) -> float:
    """Expected points held after the move (open card exact, deck draw = unseen average)."""
    from .features import pool_mask
    if act.kind == "yaniv":
        return float(mask_points(view.my_hand))
    kept = mask_points(view.my_hand & ~act.discard_mask)
    if act.draw == "open":
        return kept + POINTS_OF[view.discard[-1]]
    pool = [POINTS_OF[c] for c in cards_from_mask(pool_mask(view))]
    return kept + (sum(pool) / len(pool) if pool else 5.0)


def _kind(view, act) -> str:
    if act.kind == "yaniv":
        return "Y"
    cs = cards_from_mask(act.discard_mask)
    pts = max(POINTS_OF[c] for c in cs)
    o = view.discard[-1]
    kept = [rank_of(c) for c in cards_from_mask(view.my_hand & ~act.discard_mask)]
    d = "D" if act.draw != "open" else ("Op" if rank_of(o) in kept else "O")
    return f"{'S' if len(cs) == 1 else 'set' + str(len(cs))}{pts}/{d}"


def facts(view, a, b) -> dict:
    from .features import assaf_chance, hidden_lean, opp_min_points
    hand = cards_from_mask(view.my_hand)
    known = cards_from_mask(view.opp_known)
    kp = sum(POINTS_OF[c] for c in known)
    need = view.opp_size - len(known)
    my = mask_points(view.my_hand)
    kr = sorted({rank_of(c) for c in known})
    # does B / A throw a rank the opponent is known to hold, or leave on top a rank that pairs a known card
    return {"n": len(hand), "on": view.opp_size, "pts": my, "kn": len(known), "kp": kp,
            "kmax": max([POINTS_OF[c] for c in known], default=0),
            "est": round(kp + max(0, need) * hidden_lean(view), 1), "omin": opp_min_points(view),
            "assaf": round(assaf_chance(view), 4) if my <= 7 else -1.0, "ply": view.ply,
            "stock": view.stock_size, "open": POINTS_OF[view.discard[-1]], "top": max(POINTS_OF[c] for c in hand),
            "a": _kind(view, a), "b": _kind(view, b), "ea": round(_after(view, a), 1), "eb": round(_after(view, b), 1),
            "athr": mask_points(a.discard_mask) if a.kind != "yaniv" else 0,
            "bthr": mask_points(b.discard_mask) if b.kind != "yaniv" else 0,
            "afeed": int(a.kind != "yaniv" and any(rank_of(c) in kr for c in cards_from_mask(a.discard_mask))),
            "bfeed": int(b.kind != "yaniv" and any(rank_of(c) in kr for c in cards_from_mask(b.discard_mask)))}


def _alt_player(name, view, actions, a):
    b = _player(name)(view, actions, None)
    if action_key(b) == action_key(a):
        return None
    f = facts(view, a, b)
    f["src"] = name
    return b, json.dumps(f)


def alt_T(view, actions, a, ctx):
    return _alt_player("trained", view, actions, a)


def alt_H4(view, actions, a, ctx):
    return _alt_player("hm4", view, actions, a)


def alt_C(view, actions, a, ctx):
    return _alt_player("counter", view, actions, a)


def alt_low(view, actions, a, ctx):
    """CATCH probe: wherever the opponent holds <= 4 cards, the non-call move with the lowest expected hand after
    the draw (ties: fewest cards kept, then open card) if it is at least 1 point under hm5's move."""
    if a.kind == "yaniv" or view.opp_size > 4:
        return None
    acts = [x for x in actions if x.kind != "yaniv"]
    b = min(acts, key=lambda x: (_after(view, x), (view.my_hand & ~x.discard_mask).bit_count(), x.draw != "open"))
    if _after(view, a) - _after(view, b) < 1.0:
        return None
    f = facts(view, a, b)
    f["src"] = "low"
    return b, json.dumps(f)


def alt_low5(view, actions, a, ctx):
    """CATCH probe with a 5-card opponent that is known to hold a card worth 7+ (it took it: it likely holds the
    pair and throws it next turn, so its hand drops fast): the lowest expected hand, if 1+ point under hm5's."""
    if a.kind == "yaniv" or view.opp_size != 5:
        return None
    if max([POINTS_OF[c] for c in cards_from_mask(view.opp_known)], default=0) < 7:
        return None
    acts = [x for x in actions if x.kind != "yaniv"]
    b = min(acts, key=lambda x: (_after(view, x), (view.my_hand & ~x.discard_mask).bit_count(), x.draw != "open"))
    if _after(view, a) - _after(view, b) < 1.0:
        return None
    f = facts(view, a, b)
    f["src"] = "low5"
    return b, json.dumps(f)


def run(alt: str, opp: str, rounds: int, seed: int, out: str, hero: str = "hm5") -> None:
    from .contest_opus_playout import playout
    recs, n, margin = playout(hero, opp, f"training.yaniv.contest_opus_mirror:{alt}", rounds, seed, processes=2)
    Path(out).write_text(json.dumps({"rounds": n, "margin": margin, "hero": hero, "opp": opp, "alt": alt,
                                     "seed": seed, "recs": recs}))
    print(f"{out}: {len(recs)} records, {n} rounds, margin {margin:+.3f}", flush=True)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    run(a[0], a[1], int(a[2]), int(a[3]), a[4], *(a[5:6]))

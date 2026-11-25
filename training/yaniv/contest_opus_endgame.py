"""Team opus, topic `endgame`: the Yaniv-call decision and play after both hands hold 4 cards or fewer.

Two parts:
  1. play-out probes (alt functions for contest_opus_playout.playout) that price hm5's call decisions and its
     disagreements with the trained net. They read only the hero's view (the truth in ctx is never used).
     Each record's tag is a JSON string of the position's public facts, grouped afterwards.
  2. add_endgame_units(kit): the rules that came out of the probes (see results/contest-opus-logs/endgame.md).

Nothing is trained, fitted or searched.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_NETS: dict = {}


def _net(name: str):
    if name not in _NETS:
        from .contest_opus_playout import NAMED
        from .network import load_net
        _NETS[name] = load_net(Path(NAMED.get(name, name)))
    return _NETS[name]


def _facts(view, actions) -> dict:
    """Public facts of the position (the same numbers the net's inputs carry, unscaled)."""
    from .engine import POINTS_OF, cards_from_mask, mask_points, popcount
    from .features import assaf_chance, hidden_lean, opp_min_points
    hand = cards_from_mask(view.my_hand)
    known = cards_from_mask(view.opp_known)
    known_pts = sum(POINTS_OF[c] for c in known)
    need = view.opp_size - len(known)
    my_pts = mask_points(view.my_hand)
    legal = my_pts <= 7
    top = max(POINTS_OF[c] for c in hand)
    return {"pts": my_pts, "n": len(hand), "on": view.opp_size, "kp": known_pts, "kn": len(known),
            "omin": opp_min_points(view), "est": round(known_pts + max(0, need) * hidden_lean(view), 2),
            "assaf": round(assaf_chance(view), 5) if legal else 0.0, "ply": view.ply, "stock": view.stock_size,
            "open": POINTS_OF[view.discard[-1]], "top": top, "legal": legal,
            "ms": view.my_score, "os": view.opp_score}


def _kind(act, view) -> str:
    from .engine import POINTS_OF, cards_from_mask
    if act.kind == "yaniv":
        return "Y"
    cs = cards_from_mask(act.discard_mask)
    return f"{'S' if len(cs) == 1 else 'set'}{max(POINTS_OF[c] for c in cs)}/{'O' if act.draw == 'open' else 'D'}"


def _best_noncall(view, actions):
    from .features import score_actions
    acts = [x for x in actions if x.kind != "yaniv"]
    sc = score_actions(_net("hm5"), view, acts)
    return acts[int(sc.argmax())]


def _swap(view, actions):
    """Throw my highest single, take the open card if it is lower than that card, else draw from the deck."""
    from .engine import POINTS_OF
    best = None
    for x in actions:
        if x.kind == "yaniv" or x.discard_mask.bit_count() != 1:
            continue
        c = x.discard_mask.bit_length() - 1
        want = "open" if POINTS_OF[view.discard[-1]] < POINTS_OF[c] else "stock"
        if x.draw != want:
            continue
        if best is None or POINTS_OF[c] > POINTS_OF[best.discard_mask.bit_length() - 1]:
            best = x
    return best


def alt_call(view, actions, a, ctx):
    """Step 1: every call -> hm5's best non-call; every legal non-call -> the call."""
    if not any(x.kind == "yaniv" for x in actions):
        return None
    f = _facts(view, actions)
    if a.kind == "yaniv":
        b = _best_noncall(view, actions)
        f["dir"] = "call->play"
    else:
        b = next(x for x in actions if x.kind == "yaniv")
        f["dir"] = "play->call"
    f["a"], f["b"] = _kind(a, view), _kind(b, view)
    return b, json.dumps(f)


def alt_swap(view, actions, a, ctx):
    """Step 1b: every call -> the swap (throw highest single, take the open card if lower)."""
    if a.kind != "yaniv":
        return None
    b = _swap(view, actions)
    if b is None:
        return None
    f = _facts(view, actions)
    f["dir"] = "call->swap"
    f["a"], f["b"] = "Y", _kind(b, view)
    return b, json.dumps(f)


def alt_trained(view, actions, a, ctx):
    """Step 2: after both hands hold <= 4, the trained net's move."""
    from .engine import popcount
    if popcount(view.my_hand) > 4 or view.opp_size > 4:
        return None
    from .train import net_chooser
    if "Tch" not in _NETS:
        _NETS["Tch"] = net_chooser(_net("trained"), 0.0)
    b = _NETS["Tch"](view, actions, None)
    f = _facts(view, actions)
    f["dir"] = "T"
    f["a"], f["b"] = _kind(a, view), _kind(b, view)
    return b, json.dumps(f)


def run(alt: str, opp: str, rounds: int, seed: int, out: str, hero: str = "hm5") -> None:
    from .contest_opus_playout import playout
    recs, n, margin = playout(hero, opp, f"training.yaniv.contest_opus_endgame:{alt}", rounds, seed, processes=2)
    Path(out).write_text(json.dumps({"rounds": n, "margin": margin, "hero": hero, "opp": opp, "alt": alt,
                                     "seed": seed, "recs": recs}))
    print(f"{out}: {len(recs)} records, {n} rounds, margin {margin:+.3f}")


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    run(a[0], a[1], int(a[2]), int(a[3]), a[4], *(a[5:6]))

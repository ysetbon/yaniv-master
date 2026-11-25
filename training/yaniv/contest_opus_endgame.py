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


def swap_rule(view, actions, f=None, min_assaf=0.005, min_save=2):
    """Rule S (public facts only): Yaniv is legal, the opponent holds 3 or 4 cards, assaf >= min_assaf, and the open
    card is at least min_save points under my highest card -> the swap (throw that card, take the open card)."""
    f = f or _facts(view, actions)
    if not f["legal"] or f["on"] not in (3, 4) or f["assaf"] < min_assaf or f["top"] - f["open"] < min_save:
        return None
    return _swap(view, actions)


def alt_rule(view, actions, a, ctx):
    """Step 3 check: wherever rule S fires and hm5 plays something else, the swap."""
    if not any(x.kind == "yaniv" for x in actions):
        return None
    f = _facts(view, actions)
    b = swap_rule(view, actions, f, 0.0, 1)       # the wide net; the record says which thresholds it meets
    if b is None:
        return None
    f["dir"] = "rule"
    f["a"], f["b"] = _kind(a, view), _kind(b, view)
    return b, json.dumps(f)


def alt_leave(view, actions, a, ctx):
    """Hint check: after both hands hold <= 4, where hm5 takes an open card worth 3-6, the same throw with a deck draw."""
    from .engine import POINTS_OF, cards_from_mask, popcount, rank_of
    if a.kind == "yaniv" or a.draw != "open" or popcount(view.my_hand) > 4 or view.opp_size > 4:
        return None
    o = view.discard[-1]
    if not 3 <= POINTS_OF[o] <= 6:
        return None
    b = next((x for x in actions if x.kind != "yaniv" and x.discard_mask == a.discard_mask and x.draw == "stock"), None)
    if b is None:
        return None
    f = _facts(view, actions)
    kept = [rank_of(c) for c in cards_from_mask(view.my_hand & ~a.discard_mask)]
    f["paired"] = rank_of(o) in kept
    f["kept"] = sum(POINTS_OF[c] for c in cards_from_mask(view.my_hand & ~a.discard_mask))
    f["dir"] = "leave"
    f["a"], f["b"] = _kind(a, view), _kind(b, view)
    return b, json.dumps(f)


def alt_hm5(view, actions, a, ctx):
    """Variant check: hero = a variant net; alternative = hm5's move where it differs (gain < 0: the variant's
    move was better)."""
    from .features import action_key
    from .train import net_chooser
    if "Hch" not in _NETS:
        _NETS["Hch"] = net_chooser(_net("hm5"), 0.0)
    b = _NETS["Hch"](view, actions, None)
    if action_key(b) == action_key(a):
        return None
    f = _facts(view, actions)
    f["dir"] = "hm5"
    f["a"], f["b"] = _kind(a, view), _kind(b, view)
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


# ---------------------------------------------------------------- the block
SWAP_BONUS = 2500.0   # > the Yaniv row's +1000 plus any throw term: the swap row wins outright when the rule fires
SWAP_ASSAF = 0.005    # rule S thresholds (results/contest-opus-logs/endgame.md)
SWAP_SAVE = 2


def add_endgame_units(kit, assaf_min: float = SWAP_ASSAF, save_min: int = SWAP_SAVE, opp=(3, 4)) -> None:
    """Rule S, the swap: when Yaniv is legal, the opponent holds 3-4 cards, the uniform catch chance `assaf` is at
    least assaf_min and the open card is at least save_min points under my highest card, do not call: throw that
    card and take the open card. Fires only on that one throw row (a single of my top rank, taking the open card);
    the Yaniv row and every other row are untouched, so the rule wins by a bonus on the swap row alone.
    13 units."""
    from .handmade_kit import RANK_NAMES, RANK_POINTS
    # layer 1: "this row throws one card of rank r, takes the open card, and no card in my hand is higher" (r = 2..7;
    # a top card of 8+ cannot be held with Yaniv legal, A/joker cannot save 2 points)
    tops = {}
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        if not 2 <= p <= 7:
            continue
        higher = {f"hand_{q}": -4.0 for q, pq in zip(RANK_NAMES, RANK_POINTS) if pq > p}
        tops[kit.unit(1, f"eg_top_{r}", {f"throw_{r}": 1.0, "is_single": 1.0, "take_open": 1.0, **higher},
                      bias=-2.0, note=f"swap row: throws my highest card (a {r}) alone and takes the open card")] = 1.0
    # saving: points thrown - the open card's points >= save_min (integers, ramp from save_min - 1)
    save = {"pts_thrown": 40.0, "incoming": -10.0}
    kit.unit(1, "eg_save_a", save, bias=-(save_min - 1.0), note="relu(thrown - open - (save_min - 1))")
    kit.unit(1, "eg_save_b", save, bias=-float(save_min), note="relu(thrown - open - save_min)")
    # catch chance: assaf >= assaf_min (ramp 0.00001 wide just under it)
    k = 100000.0
    kit.unit(1, "eg_assaf_a", {"assaf": k}, bias=1.0 - k * assaf_min, note="assaf ramp")
    kit.unit(1, "eg_assaf_b", {"assaf": k}, bias=-k * assaf_min, note="assaf ramp shifted: a - b = [assaf >= min]")
    gate = {f"opp_has_{n}": 1.0 for n in opp}
    kit.unit(2, "eg_swap", {**tops, "eg_save_a": 1.0, "eg_save_b": -1.0, "eg_assaf_a": 1.0, "eg_assaf_b": -1.0,
                            "yaniv_legal": 1.0, **gate}, bias=-4.0,
             note="rule S: the swap row while a call is legal but catchable and the open card is 2+ lower")
    kit.out({"eg_swap": SWAP_BONUS})


def build_variant(path: str, **kw) -> None:
    from .contest_opus import build
    from .network import save_net
    net, kit = build(extras=[lambda k: add_endgame_units(k, **kw)])
    save_net(net, Path(path), meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                                    "source": "training/yaniv/contest_opus_endgame.py", "endgame": kw})
    print(path, net.dims, sum(net.dims[1:-1]))

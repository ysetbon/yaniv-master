"""Team opus, topic `mine`: 5-card-phase play and the draw decision (deck or open card).

Play-out rules (alt functions for contest_opus_playout) and, if one pays, a block of hand-written units
add_mine_units(kit) for contest_opus.build(extras=[...]). Nothing here is trained, fitted or searched; ctx's truth
(state, opp_hand) is never read by a rule. Log: results/contest-opus-logs/mine.md.
"""

from __future__ import annotations

from pathlib import Path

from .engine import POINTS_OF, cards_from_mask, mask_points, rank_of
from .features import action_key

ROOT = Path(__file__).resolve().parents[2]
_CACHE: dict = {}


def _trained():
    if "trained" not in _CACHE:
        from .contest_opus_playout import player
        _CACHE["trained"] = player("trained")
    return _CACHE["trained"]


def phase(view) -> str:
    """'5' while either hand holds 5 cards; 'mid' while not both <= 4... ; 'end' once both hands hold 4 or fewer."""
    mine = view.my_hand.bit_count()
    if mine >= 5 or view.opp_size >= 5:
        return "p5"
    return "end"


def _desc(view, act) -> str:
    """Short description of a move: throw kind and draw."""
    if act.kind == "yaniv":
        return "Y"
    cards = cards_from_mask(act.discard_mask)
    n = len(cards)
    p = POINTS_OF[cards[0]]
    kind = f"set{n}" if n > 1 else "single"
    return f"{kind}({p})/{act.draw[0]}"


def kind_of_diff(view, a, b) -> str:
    """Group the difference between A (hero) and B (alternative)."""
    if a.kind == "yaniv" or b.kind == "yaniv":
        return "yaniv:" + ("A" if a.kind == "yaniv" else "B")
    if a.discard_mask == b.discard_mask or action_key(a)[:3] == action_key(b)[:3]:
        op = POINTS_OF[view.discard[-1]]
        hand_ranks = [rank_of(c) for c in cards_from_mask(view.my_hand & ~a.discard_mask)]
        pairs = rank_of(view.discard[-1]) in hand_ranks
        ob = "lo" if op <= 3 else "mid" if op <= 6 else "hi"
        return f"draw:{a.draw[0]}->{b.draw[0]} open={ob}{'+pair' if pairs else ''}"
    ca, cb = cards_from_mask(a.discard_mask), cards_from_mask(b.discard_mask)
    ka = "set" if len(ca) > 1 else "single"
    kb = "set" if len(cb) > 1 else "single"
    pa, pb = mask_points(a.discard_mask), mask_points(b.discard_mask)
    rel = "heavier" if pb > pa else "lighter" if pb < pa else "same"
    return f"throw:{ka}->{kb} B {rel}"


def alt_trained(view, actions, a, ctx):
    """The trained net's move, tagged by phase and kind of difference."""
    b = _trained()(view, actions, None)
    if action_key(b) == action_key(a):
        return None
    return b, f"{phase(view)} {kind_of_diff(view, a, b)}"


def alt_counter(view, actions, a, ctx):
    from .heuristic import choose_heuristic
    b = choose_heuristic(view, actions, None)
    if action_key(b) == action_key(a):
        return None
    return b, f"{phase(view)} {kind_of_diff(view, a, b)}"


def open_rel(view, act) -> str:
    """How the open card relates to the cards a move keeps: 'm<k>' = it matches k kept cards, 'm0' none."""
    o = rank_of(view.discard[-1])
    kept = [rank_of(c) for c in cards_from_mask(view.my_hand & ~act.discard_mask)]
    return f"m{kept.count(o)}"


def alt_trained2(view, actions, a, ctx):
    """The trained net's move in the 5-card phase, tagged finer: throw kinds, draws, open-card match."""
    if phase(view) != "p5":
        return None
    b = _trained()(view, actions, None)
    if action_key(b) == action_key(a) or b.kind == "yaniv" or a.kind == "yaniv":
        return None
    op = POINTS_OF[view.discard[-1]]
    na, nb = a.discard_mask.bit_count(), b.discard_mask.bit_count()
    return b, (f"A{'set' if na > 1 else 'sgl'}/{a.draw[0]} B{'set' if nb > 1 else 'sgl'}/{b.draw[0]}"
               f" Bopen{open_rel(view, b)} op{'lo' if op <= 3 else 'mid' if op <= 6 else 'hi'}")


def alt_flipdraw(view, actions, a, ctx):
    """Same throw as the hero, the other draw (deck <-> open card). Tagged by open card, match, card counts."""
    if a.kind != "play" or view.stock_size == 0:
        return None
    from .engine import Action
    b = Action("play", a.discard_mask, a.top, "open" if a.draw == "stock" else "stock")
    op = POINTS_OF[view.discard[-1]]
    ob = "A/J" if op <= 1 else str(op) if op <= 6 else "7-9" if op <= 9 else "10"
    return b, (f"{a.draw[0]}->{b.draw[0]} op{ob} {open_rel(view, a)} me{view.my_hand.bit_count()}"
               f" opp{view.opp_size}")


def _setsingle(view, a, b):
    """Detail tag when A throws a set and B a lighter single (the class both rivals prefer)."""
    ca, cb = cards_from_mask(a.discard_mask), cards_from_mask(b.discard_mask)
    if a.kind != "play" or b.kind != "play" or len(ca) < 2 or len(cb) != 1:
        return None
    if mask_points(b.discard_mask) >= mask_points(a.discard_mask):
        return None
    o = view.discard[-1]
    same = "open=setrank" if rank_of(o) == rank_of(ca[0]) else f"open{open_rel(view, b)}"
    return (f"set{len(ca)}x{POINTS_OF[ca[0]]} sgl{POINTS_OF[cb[0]]} A/{a.draw[0]} B/{b.draw[0]} {same}"
            f" op{POINTS_OF[o]}")


def alt_ss_trained(view, actions, a, ctx):
    if phase(view) != "p5":
        return None
    b = _trained()(view, actions, None)
    t = _setsingle(view, a, b)
    return None if t is None else (b, t)


def alt_ss_counter(view, actions, a, ctx):
    if phase(view) != "p5":
        return None
    from .heuristic import choose_heuristic
    b = choose_heuristic(view, actions, None)
    t = _setsingle(view, a, b)
    return None if t is None else (b, t)

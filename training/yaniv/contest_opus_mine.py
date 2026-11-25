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


def _triple_b(view, actions, a):
    """Rule TRIPLE: A throws my set of rank R while the open card is also an R. B: keep the set, throw my
    heaviest card of another rank (single), and take the open R (I then hold 3+ R's to throw next turn)."""
    if a.kind != "play":
        return None
    ca = cards_from_mask(a.discard_mask)
    r = rank_of(view.discard[-1])
    if len(ca) < 2 or rank_of(ca[0]) != r:
        return None
    if [rank_of(c) for c in cards_from_mask(view.my_hand)].count(r) != len(ca):
        return None                                    # A must throw all my R's
    others = [c for c in cards_from_mask(view.my_hand) if rank_of(c) != r]
    if not others:
        return None
    hi = max(others, key=lambda c: (POINTS_OF[c], c))
    for b in actions:
        if b.kind == "play" and b.discard_mask == 1 << hi and b.draw == "open":
            return b, hi
    return None


def alt_triple(view, actions, a, ctx):
    got = _triple_b(view, actions, a)
    if got is None:
        return None
    b, hi = got
    rp = POINTS_OF[view.discard[-1]]
    rb = "R<=3" if rp <= 3 else "R4-6" if rp <= 6 else "R7-9" if rp <= 9 else "R10"
    sb = "s<=3" if POINTS_OF[hi] <= 3 else "s4-6" if POINTS_OF[hi] <= 6 else "s7+"
    return b, f"{phase(view)} {rb} {sb} A/{a.draw[0]} me{view.my_hand.bit_count()} opp{view.opp_size}"


def alt_trained_draw(view, actions, a, ctx):
    """Trained net's move only where it keeps the hero's throw and changes the draw; tag: held total after."""
    if a.kind != "play":
        return None
    b = _trained()(view, actions, None)
    if b.kind != "play" or action_key(b)[:3] != action_key(a)[:3] or b.draw == a.draw:
        return None
    kept = mask_points(view.my_hand & ~a.discard_mask)
    op = POINTS_OF[view.discard[-1]]
    held = kept + op
    hb = "<=7" if held <= 7 else "8-10" if held <= 10 else "11-15" if held <= 15 else "16+"
    return b, f"{a.draw[0]}->{b.draw[0]} op{op} held{hb} me{view.my_hand.bit_count()} opp{view.opp_size}"


# ------------------------------------------------------------------------------------------------ the net block
TRIPLE_BONUS = 5.0   # one card charge (CARD = 5): the third R leaves with the pair next turn, a throw saved
TRIPLE_RANKS = ["4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]   # R worth 4+ (R <= 3 measured no better)


def add_mine_units(kit, bonus: float = TRIPLE_BONUS) -> None:
    """Rule TRIPLE (results/contest-opus-logs/mine.md): I hold two or more R's (R worth 4+) and the open card is
    an R. hm5 throws the R's and draws the deck; better: keep them, throw a card of another rank worth 4+
    points, and take the open R, so three R's leave together next turn. Measured +2.7 (counter) and +3.0
    (trained) points each time, 0.05 times a round. 10 layer-1 units, one per rank R."""
    light = {f"throw_{x}": -4.0 for x in ("A", "2", "3", "JOKER")}   # the card thrown now must be worth 4+
    out = {}
    for r in TRIPLE_RANKS:
        # take_open + open_R + (R's in my hand: 2 -> 2) - 4 x [R thrown] - 4 x [light rank thrown] - 3:
        # 1 when all hold (2 if I hold three R's), else 0. Light ranks: A, 2, 3, joker (a joker is 0 points).
        u = kit.unit(1, f"mine_triple_{r}", {"take_open": 1.0, f"open_{r}": 1.0, f"hand_{r}": 4.0,
                                             f"throw_{r}": -4.0, **light}, bias=-3.0,
                     note=f"takes an open {r} onto my two+ {r}s (kept), throwing a 4+ card of another rank")
        out[u] = bonus
    kit.out(out)


def alt_variant(view, actions, a, ctx):
    """The move of the variant net in $MINE_VARIANT where it differs from the hero's; tagged TRIPLE or other."""
    import os
    key = "variant:" + os.environ["MINE_VARIANT"]
    if key not in _CACHE:
        from .contest_opus_playout import player
        _CACHE[key] = player(os.environ["MINE_VARIANT"])
    b = _CACHE[key](view, actions, None)
    if action_key(b) == action_key(a):
        return None
    trip = _triple_b(view, actions, a) is not None and b.kind == "play" and b.draw == "open" \
        and open_rel(view, b) in ("m2", "m3")
    return b, f"{'TRIPLE' if trip else 'other'} {phase(view)} opp{view.opp_size}"


def _hm5_best_where(view, actions, ok):
    """hm5's best move among those with ok(action) True (None if none)."""
    import numpy as np
    from .features import score_actions
    if "hm5net" not in _CACHE:
        from .contest_opus_playout import NAMED
        from .network import load_net
        _CACHE["hm5net"] = load_net(NAMED["hm5"])
    sc = score_actions(_CACHE["hm5net"], view, actions)
    idx = [i for i, x in enumerate(actions) if ok(x)]
    if not idx:
        return None
    return actions[max(idx, key=lambda i: sc[i])]


def alt_deny(view, actions, a, ctx):
    """DENY: A leaves face up a card of a rank the opponent is known to hold (it would take it and throw a set).
    B: hm5's best move that does not."""
    if a.kind != "play":
        return None
    known = {rank_of(c) for c in cards_from_mask(view.opp_known)}
    if rank_of(a.top) not in known:
        return None
    b = _hm5_best_where(view, actions, lambda x: x.kind == "play" and rank_of(x.top) not in known)
    if b is None:
        return None
    return b, f"deny top{POINTS_OF[a.top]} {phase(view)} opp{view.opp_size}"


def alt_keeplight(view, actions, a, ctx):
    """KEEPLIGHT: A throws an ace, 2 or joker while I hold 4+ cards. B: hm5's best move that keeps them."""
    if a.kind != "play" or view.my_hand.bit_count() < 4:
        return None
    if POINTS_OF[a.top] > 2:
        return None
    b = _hm5_best_where(view, actions, lambda x: x.kind == "play" and POINTS_OF[x.top] > 2)
    if b is None:
        return None
    return b, f"keeplight {'set' if a.discard_mask.bit_count() > 1 else 'sgl'}{POINTS_OF[a.top]} {phase(view)} opp{view.opp_size}"

"""Team opus, topic `twin`: keep the high single whose twin the opponent is likely to throw?

The owner's idea: early in a round both players throw their large cards. If I hold two or more high singles,
keep the one whose twin (same rank) is most likely still to come, take that twin from the open pile when the
opponent throws it, and throw the pair next turn - instead of always throwing my highest single (or hm5's move).

This file holds the play-out rules (alt functions for contest_opus_playout). RESULT: hm5 already plays the whole
plan (LIVE charge picks the single, PAIR_OPEN takes the twin, then it throws the pair) and no deviation tested here
paid against both the counter and the trained net, so there is NO add_twin_units block. Results: results/contest-opus-logs/twin.md. Nothing is trained, fitted or searched.

Situation ("early, >= 2 high singles"): I hold 5 cards, the opponent holds >= 4, the open card pairs none
of my cards, and I hold at least two ranks
with exactly ONE copy worth >= HI points (HI = 8, or 7 for the `_7` variants). Rules only read the view (my hand,
the discard pile, the opponent's known cards, its card count); ctx["state"] / ctx["opp_hand"] are never read.
"""

from __future__ import annotations

from .engine import POINTS_OF, cards_from_mask, rank_of
from .features import pool_mask


# ------------------------------------------------------------------ helpers (view only)
def _rank_counts(mask: int) -> dict:
    out: dict = {}
    for c in cards_from_mask(mask):
        out[rank_of(c)] = out.get(rank_of(c), 0) + 1
    return out


def _pts(r: int) -> int:
    return 0 if r == 13 else POINTS_OF[r]


def _unseen(view) -> dict:
    return _rank_counts(pool_mask(view))


def high_singles(view, hi: int) -> list:
    """Ranks I hold exactly once, worth >= hi points: [(rank, card)]."""
    cnt = _rank_counts(view.my_hand)
    return [(rank_of(c), c) for c in cards_from_mask(view.my_hand) if cnt[rank_of(c)] == 1 and _pts(rank_of(c)) >= hi]


def early(view) -> bool:
    """5 cards in my hand, >= 4 in the opponent's, and the open card pairs NO card of mine (that is the pickup
    case, measured separately: the open copy is not 'unseen', so a count rule would throw the very rank whose twin
    lies face up - a first version did exactly that and lost ~5 points each time)."""
    if bin(view.my_hand).count("1") != 5 or view.opp_size < 4:
        return False
    r = rank_of(view.discard[-1])
    return not any(rank_of(c) == r for c in cards_from_mask(view.my_hand))


def _single_of(a):
    if a.kind != "play" or a.discard_mask.bit_count() != 1:
        return None
    return rank_of(a.top)


def throw_card(view, actions, ctx, card):
    """hm5's best move among those that throw exactly this card (it still picks the draw)."""
    sub = [x for x in actions if x.kind == "play" and x.discard_mask == (1 << card)]
    return ctx["hero"](view, sub, None) if sub else None


def _tag(view, a_rank, b_rank, uns, extra=""):
    pa = "x" if a_rank is None else uns.get(a_rank, 0)
    return (f"{extra}|uB{uns.get(b_rank, 0)}|uA{pa}|ply{min(view.ply, 9)}|o{view.opp_size}"
            f"|A{'set' if a_rank is None else _pts(a_rank)}|B{_pts(b_rank)}")


def _choice(view, actions, a, ctx, hi, pick):
    if not early(view):
        return None
    hs = high_singles(view, hi)
    if len(hs) < 2:
        return None
    uns = _unseen(view)
    r, c = pick(view, hs, uns)
    if r is None:
        return None
    a_rank = _single_of(a)
    if a_rank == r:
        return None
    b = throw_card(view, actions, ctx, c)
    if b is None:
        return None
    return b, _tag(view, a_rank, r, uns)


# ------------------------------------------------------------------ (b) throw the high single with FEWEST unseen
def _pick_fewest(view, hs, uns):
    # fewest unseen copies; ties -> the higher points (then hm5 would decide, but we must pick one: higher)
    r, c = min(hs, key=lambda rc: (uns.get(rc[0], 0), -_pts(rc[0]), rc[0]))
    return r, c


def alt_fewest8(view, actions, a, ctx):
    return _choice(view, actions, a, ctx, 8, _pick_fewest)


def alt_fewest7(view, actions, a, ctx):
    return _choice(view, actions, a, ctx, 7, _pick_fewest)


# ------------------------------------------------------------------ (b') the opposite: throw the MOST unseen
def _pick_most(view, hs, uns):
    return max(hs, key=lambda rc: (uns.get(rc[0], 0), _pts(rc[0]), rc[0]))


def alt_most8(view, actions, a, ctx):
    return _choice(view, actions, a, ctx, 8, _pick_most)


# ------------------------------------------------------------------ (c) keep the rank the opponent has shown
def _pick_shown(view, hs, uns):
    """Keep a rank the opponent is known to hold (took it from the pile), or failing that a rank whose copy is
    face up right now under... no: whose copy was thrown by the opponent last turn is the open card (pickup case).
    So: throw the highest high single whose rank the opponent is NOT known to hold, if some kept one is known."""
    known = _rank_counts(view.opp_known)
    keep = [rc for rc in hs if known.get(rc[0], 0) > 0]
    if not keep:
        return None, None
    rest = [rc for rc in hs if known.get(rc[0], 0) == 0]
    if not rest:
        return None, None
    return max(rest, key=lambda rc: (_pts(rc[0]), rc[0]))


def alt_shown8(view, actions, a, ctx):
    return _choice(view, actions, a, ctx, 8, _pick_shown)


def alt_shown7(view, actions, a, ctx):
    return _choice(view, actions, a, ctx, 7, _pick_shown)


# ------------------------------------------------------------------ (d) always the highest single
def _pick_highest(view, hs, uns):
    return max(hs, key=lambda rc: (_pts(rc[0]), -uns.get(rc[0], 0), rc[0]))


def _choice_highest(view, actions, a, ctx, hi):
    """(d): differs only when hm5 does not throw a single worth the top value."""
    if not early(view):
        return None
    hs = high_singles(view, hi)
    if len(hs) < 2:
        return None
    a_rank = _single_of(a)
    top = max(_pts(r) for r, _ in hs)
    if a_rank is not None and _pts(a_rank) == top:
        return None
    uns = _unseen(view)
    r, c = _pick_highest(view, hs, uns)
    b = throw_card(view, actions, ctx, c)
    return None if b is None else (b, _tag(view, a_rank, r, uns))


def alt_highest8(view, actions, a, ctx):
    return _choice_highest(view, actions, a, ctx, 8)


def alt_highest7(view, actions, a, ctx):
    return _choice_highest(view, actions, a, ctx, 7)


# ------------------------------------------------------------------ (e) keep the HIGHEST, throw a lower high single
def _keep_high(view, actions, a, ctx, hi):
    """The opponent throws its highest cards first, so the twin of my 10 is likelier to show up soon than the twin
    of my 8. When hm5 throws a top-value high single, B throws the highest high single worth LESS (keeping the
    top), only when its unseen copies are >= the top's (so LIVE does not argue the other way)."""
    if not early(view):
        return None
    hs = high_singles(view, hi)
    if len(hs) < 2:
        return None
    a_rank = _single_of(a)
    top = max(_pts(r) for r, _ in hs)
    if a_rank is None or _pts(a_rank) != top:
        return None
    uns = _unseen(view)
    lower = [rc for rc in hs if _pts(rc[0]) < top and uns.get(rc[0], 0) <= uns.get(a_rank, 0)]
    if not lower:
        return None
    r, c = max(lower, key=lambda rc: (_pts(rc[0]), rc[0]))
    b = throw_card(view, actions, ctx, c)
    return None if b is None else (b, _tag(view, a_rank, r, uns))


def alt_keephigh8(view, actions, a, ctx):
    return _keep_high(view, actions, a, ctx, 8)


def alt_keephigh7(view, actions, a, ctx):
    return _keep_high(view, actions, a, ctx, 7)


# ------------------------------------------------------------------ what does hm5 do here? (no alternative)
def alt_census8(view, actions, a, ctx):
    """Never proposes a move; used only via census() below."""
    return None


def census(view, a, hi):
    """Classify hm5's move in the situation: which high single (by unseen-copy rank order) or other."""
    if not early(view):
        return None
    hs = high_singles(view, hi)
    if len(hs) < 2:
        return None
    uns = _unseen(view)
    a_rank = _single_of(a)
    if a_rank is None:
        return "set/yaniv"
    if a_rank not in [r for r, _ in hs]:
        return "low single"
    fewest = min(uns.get(r, 0) for r, _ in hs)
    most = max(uns.get(r, 0) for r, _ in hs)
    top = max(_pts(r) for r, _ in hs)
    k = []
    k.append("fewest" if uns.get(a_rank, 0) == fewest else "notfewest")
    if fewest == most:
        k[-1] = "equal-unseen"
    k.append("top" if _pts(a_rank) == top else "lower")
    return "/".join(k)


# ------------------------------------------------------------------ the pickup half
def _pickup(view, actions, a, ctx, hi, need_early):
    """The open card pairs a single I hold worth >= hi. B = the best hm5 move that takes the open card and keeps
    that rank (does not throw it); compared with hm5's move when hm5 does not take it."""
    if need_early and not (bin(view.my_hand).count("1") == 5 and view.opp_size >= 4):
        return None
    r = rank_of(view.discard[-1])
    if _pts(r) < hi:
        return None
    cnt = _rank_counts(view.my_hand)
    if cnt.get(r, 0) != 1:
        return None
    if a.kind == "play" and a.draw == "open" and rank_of(a.top) != r:
        return None          # hm5 already takes it (and keeps its twin)
    sub = [x for x in actions if x.kind == "play" and x.draw == "open"
           and not any(rank_of(c) == r for c in cards_from_mask(x.discard_mask))]
    if not sub:
        return None
    b = ctx["hero"](view, sub, None)
    what = "yaniv" if a.kind == "yaniv" else ("throws-twin" if any(rank_of(c) == r for c in
                                                                   cards_from_mask(a.discard_mask)) else a.draw)
    uns = _unseen(view)
    return b, f"A:{what}|r{_pts(r)}|u{uns.get(r, 0)}|o{view.opp_size}|n{bin(view.my_hand).count('1')}"


def alt_pickup8(view, actions, a, ctx):
    return _pickup(view, actions, a, ctx, 8, False)


def alt_pickup7(view, actions, a, ctx):
    return _pickup(view, actions, a, ctx, 7, False)


# ------------------------------------------------------------------ own idea: the draw after an early throw
def _same_throw_other_draw(actions, a, draw):
    for x in actions:
        if x.kind == "play" and x.discard_mask == a.discard_mask and x.top == a.top and x.draw == draw:
            return x
    return None


def alt_open_mid(view, actions, a, ctx):
    """5 v >=4 cards; hm5 draws from the deck while the open card (pairing nothing of mine) is worth 3-6: take it."""
    if bin(view.my_hand).count("1") != 5 or view.opp_size < 4 or a.kind != "play" or a.draw != "stock":
        return None
    oc = view.discard[-1]
    r = rank_of(oc)
    if not 3 <= _pts(r) <= 6:
        return None
    kept = view.my_hand & ~a.discard_mask
    if any(rank_of(c) == r for c in cards_from_mask(kept)):
        return None
    b = _same_throw_other_draw(actions, a, "open")
    kmax = max((_pts(rank_of(c)) for c in cards_from_mask(kept)), default=0)
    return None if b is None else (b, f"open{_pts(r)}|o{view.opp_size}|kmax{min(kmax, 10)}|ply{min(view.ply, 9)}")


def alt_deck_low(view, actions, a, ctx):
    """5 v >=4 cards; hm5 takes an open card worth 0-3 that pairs nothing kept: draw from the deck instead."""
    if bin(view.my_hand).count("1") != 5 or view.opp_size < 4 or a.kind != "play" or a.draw != "open":
        return None
    r = rank_of(view.discard[-1])
    if _pts(r) > 3:
        return None
    kept = view.my_hand & ~a.discard_mask
    if any(rank_of(c) == r for c in cards_from_mask(kept)):
        return None
    b = _same_throw_other_draw(actions, a, "stock")
    return None if b is None else (b, f"open{_pts(r)}|o{view.opp_size}|ply{min(view.ply, 9)}")

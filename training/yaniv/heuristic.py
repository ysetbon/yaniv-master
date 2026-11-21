"""Card-counting Yaniv bench.

The benchmark only sees its own hand and the public table. It throws the
line that leaves the lowest expected hand, and it calls Yaniv only when the
opponent cannot legally be lower, or when a counted sample says an Assaf is
unlikely. The neural net is trained to play at this level and past it.
"""

from __future__ import annotations

import itertools
import random
from math import comb

from .engine import (
    POINTS_OF,
    YANIV_LIMIT,
    Action,
    View,
    best_keep_points,
    cards_from_mask,
    mask_points,
    popcount,
)
from .features import opp_min_points, pool_mask


def hand_badness(mask: int, opp_size: int) -> float:
    """Lower is better.

    `keep` is the hand you can reach on your next throw. `points` is what you
    are caught with if the opponent calls before then. A player still holding
    five cards cannot call, so the current total barely matters.
    """
    points = mask_points(mask)
    keep = best_keep_points(mask)
    if opp_size >= 5:
        weight = 0.08
    elif opp_size == 4:
        weight = 0.22
    elif opp_size == 3:
        weight = 0.40
    else:
        weight = 0.62
    return weight * points + (1.0 - weight) * keep


def _top_penalty(card: int) -> float:
    points = POINTS_OF[card]
    if points <= 1:
        return 1.8
    if points <= 3:
        return 1.0
    if points <= 5:
        return 0.35
    return 0.0


def _seed(view: View) -> int:
    folded = 2166136261
    for card in view.discard:
        folded ^= card + 0x9E3779B9
        folded = (folded * 16777619) & 0xFFFFFFFF
    for value in (
        view.my_hand,
        view.opp_known,
        view.opp_size,
        view.my_score,
        view.opp_score,
        view.ply,
        view.stock_size,
    ):
        folded ^= value & 0xFFFFFFFF
        folded = (folded * 16777619) & 0xFFFFFFFF
    return folded


def assaf_risk(view: View, samples: int = 36) -> float:
    """Fraction of possible opponent hands that would catch a Yaniv call.

    A skilled opponent keeps low cards, so the real risk is higher than this
    uniform count. Call thresholds stay strict to leave that margin.
    """
    mine = mask_points(view.my_hand)
    if mine < opp_min_points(view):
        return 0.0
    known = cards_from_mask(view.opp_known)
    known_pts = sum(POINTS_OF[c] for c in known)
    need = view.opp_size - len(known)
    pool = cards_from_mask(pool_mask(view))
    if need <= 0:
        return 1.0 if known_pts <= mine else 0.0
    if need > len(pool):
        return 1.0
    if comb(len(pool), need) <= 450:
        bad = 0
        total = 0
        for combo in itertools.combinations(pool, need):
            total += 1
            if known_pts + sum(POINTS_OF[c] for c in combo) <= mine:
                bad += 1
        return bad / total if total else 1.0
    rng = random.Random(_seed(view))
    bad = 0
    for _ in range(samples):
        draw = rng.sample(pool, need)
        if known_pts + sum(POINTS_OF[c] for c in draw) <= mine:
            bad += 1
    return bad / samples


def _call_threshold(points: int) -> float:
    if points <= 1:
        return 0.40
    if points == 2:
        return 0.28
    if points == 3:
        return 0.18
    if points == 4:
        return 0.12
    return 0.08


def should_call(view: View) -> bool:
    mine = mask_points(view.my_hand)
    if mine > YANIV_LIMIT or popcount(view.my_hand) == 0:
        return False
    if mine < opp_min_points(view):
        return True
    return assaf_risk(view) <= _call_threshold(mine)


def _score_action(view: View, action: Action, pool: list[int]) -> tuple[float, float]:
    remaining = view.my_hand & ~action.discard_mask
    penalty = _top_penalty(action.top)
    if action.draw == "open":
        held = remaining | (1 << view.discard[-1])
        return hand_badness(held, view.opp_size) + penalty, float(mask_points(held))
    if not pool:
        return hand_badness(remaining, view.opp_size) + penalty + 4.0, float(mask_points(remaining))
    bad = 0.0
    points = 0.0
    for card in pool:
        held = remaining | (1 << card)
        bad += hand_badness(held, view.opp_size)
        points += mask_points(held)
    n = float(len(pool))
    return bad / n + penalty, points / n


def choose_heuristic(view: View, actions: list[Action], rng=None) -> Action:
    del rng
    for action in actions:
        if action.kind == "play" and action.draw == "none":
            return action
    call = next((action for action in actions if action.kind == "yaniv"), None)
    if call is not None and mask_points(view.my_hand) < opp_min_points(view):
        return call
    pool = cards_from_mask(pool_mask(view))
    best: Action | None = None
    best_key: tuple[float, float, int, int] | None = None
    for action in actions:
        if action.kind != "play":
            continue
        badness, expected = _score_action(view, action, pool)
        key = (badness, expected, -POINTS_OF[action.top], -action.top)
        if best_key is None or key < best_key:
            best_key = key
            best = action
    if call is not None and should_call(view):
        return call
    if best is None:
        return actions[0]
    return best


def choose_greedy(view: View, actions: list[Action], rng=None) -> Action:
    """Throw the fattest combination. Take the open card only when it is cheap."""
    del rng
    for action in actions:
        if action.kind == "play" and action.draw == "none":
            return action
    mine = mask_points(view.my_hand)
    if mine <= YANIV_LIMIT and (mine <= 3 or view.opp_size >= 5):
        call = next(action for action in actions if action.kind == "yaniv")
        return call
    open_card = view.discard[-1]
    best: Action | None = None
    best_key: tuple[int, int, int] | None = None
    for action in actions:
        if action.kind != "play":
            continue
        remaining = view.my_hand & ~action.discard_mask
        open_helps = False
        if action.draw == "open":
            if POINTS_OF[open_card] <= 4:
                open_helps = True
            elif any(card % 13 == open_card % 13 for card in cards_from_mask(remaining)):
                open_helps = True
        preference = 1 if open_helps else 0 if action.draw == "stock" else -1
        key = (mask_points(action.discard_mask), preference, POINTS_OF[action.top])
        if best_key is None or key > best_key:
            best_key = key
            best = action
    return best if best is not None else actions[0]


def choose_random(view: View, actions: list[Action], rng) -> Action:
    del view
    calls = [action for action in actions if action.kind == "yaniv"]
    plays = [action for action in actions if action.kind == "play"]
    if calls and rng.random() < 0.3:
        return calls[0]
    return rng.choice(plays or actions)

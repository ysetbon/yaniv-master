"""State and action features for the Yaniv net.

The layout is the contract with the TypeScript inference code. Every vector
is float32 and the lengths are asserted.
"""

from __future__ import annotations

import numpy as np

from .engine import (
    MAX_PLIES,
    POINTS_OF,
    YANIV_LIMIT,
    Action,
    View,
    best_keep_points,
    cards_from_mask,
    is_valid_combo,
    mask_points,
    popcount,
)

STATE_DIM = 256
ACTION_DIM = 121
INPUT_DIM = STATE_DIM + ACTION_DIM


def _discard_mask(view: View) -> int:
    mask = 0
    for card in view.discard:
        mask |= 1 << card
    return mask


def pool_mask(view: View) -> int:
    discard = _discard_mask(view)
    all_cards = (1 << 52) - 1
    return all_cards & ~view.my_hand & ~discard & ~view.opp_known


def opp_min_points(view: View) -> int:
    known_cards = cards_from_mask(view.opp_known)
    known_pts = sum(POINTS_OF[c] for c in known_cards)
    need = view.opp_size - len(known_cards)
    if need <= 0:
        return known_pts
    pool_pts = sorted(POINTS_OF[c] for c in cards_from_mask(pool_mask(view)))
    if need > len(pool_pts):
        return known_pts
    return known_pts + sum(pool_pts[:need])


def _rank_counts(mask: int) -> list[float]:
    counts = [0.0] * 13
    for card in cards_from_mask(mask):
        counts[card % 13] += 1.0
    return [c / 4.0 for c in counts]


def _suit_counts(mask: int) -> list[float]:
    counts = [0.0] * 4
    for card in cards_from_mask(mask):
        counts[card // 13] += 1.0
    return [c / 5.0 for c in counts]


def _bits(mask: int, size: int = 52) -> list[float]:
    return [1.0 if (mask >> i) & 1 else 0.0 for i in range(size)]


def state_features(view: View) -> np.ndarray:
    discard = _discard_mask(view)
    pool = pool_mask(view)
    pool_cards = cards_from_mask(pool)
    pool_n = len(pool_cards)
    if pool_n:
        pool_pts = [POINTS_OF[c] for c in pool_cards]
        pool_avg = sum(pool_pts) / pool_n
        pool_low = sum(1 for p in pool_pts if p <= 3) / pool_n
    else:
        pool_avg = 0.0
        pool_low = 0.0
    my_pts = mask_points(view.my_hand)
    my_size = popcount(view.my_hand)
    keep = best_keep_points(view.my_hand)
    shed = my_pts - keep
    known_cards = cards_from_mask(view.opp_known)
    known_pts = sum(POINTS_OF[c] for c in known_cards)
    minimum = opp_min_points(view)
    open_card = view.discard[-1]
    yaniv_legal = 1.0 if my_size > 0 and my_pts <= YANIV_LIMIT else 0.0
    guaranteed = 1.0 if yaniv_legal and my_pts < minimum else 0.0
    vec = (
        _bits(view.my_hand)
        + _bits(discard)
        + _bits(1 << open_card)
        + _bits(view.opp_known)
        + _rank_counts(view.my_hand)
        + _rank_counts(pool)
        + _suit_counts(view.my_hand)
        + [
            my_pts / 40.0,
            my_size / 5.0,
            view.opp_size / 5.0,
            view.stock_size / 40.0,
            view.my_score / 120.0,
            view.opp_score / 120.0,
            pool_n / 40.0,
            pool_avg / 10.0,
            pool_low,
            yaniv_legal,
            view.ply / MAX_PLIES,
            keep / 40.0,
            shed / 40.0,
            known_pts / 40.0,
            len(known_cards) / 5.0,
            minimum / 40.0,
            guaranteed,
            POINTS_OF[open_card] / 10.0,
        ]
    )
    if len(vec) != STATE_DIM:
        raise RuntimeError(f"state dim {len(vec)} != {STATE_DIM}")
    return np.asarray(vec, dtype=np.float32)


def _extends(remaining: int, card: int) -> bool:
    suit = card // 13
    rank = card % 13
    for held in cards_from_mask(remaining):
        if held // 13 == suit and abs(held % 13 - rank) == 1:
            return True
    return False


def _kept_pair(mask: int) -> bool:
    seen = [0] * 13
    for card in cards_from_mask(mask):
        seen[card % 13] += 1
        if seen[card % 13] >= 2:
            return True
    return False


def _combo_kind(cards: list[int]) -> tuple[float, float, float]:
    if len(cards) == 1:
        return 0.0, 0.0, 1.0
    if all(c % 13 == cards[0] % 13 for c in cards):
        return 1.0, 0.0, 0.0
    return 0.0, 1.0, 0.0


def action_features(view: View, action: Action) -> np.ndarray:
    if action.kind == "yaniv":
        vec = _bits(0) + _bits(0) + [0.0] * (ACTION_DIM - 104)
        vec[106] = 1.0
        if len(vec) != ACTION_DIM:
            raise RuntimeError(f"action dim {len(vec)}")
        return np.asarray(vec, dtype=np.float32)

    chosen = cards_from_mask(action.discard_mask)
    remaining = view.my_hand & ~action.discard_mask
    remain_pts = mask_points(remaining)
    remain_n = popcount(remaining)
    discard_pts = sum(POINTS_OF[c] for c in chosen)
    is_set, is_run, is_single = _combo_kind(chosen)
    draw_stock = 1.0 if action.draw == "stock" else 0.0
    draw_open = 1.0 if action.draw == "open" else 0.0
    open_card = view.discard[-1]
    pairs = 0.0
    extends = 0.0
    incoming = 0.0
    result_callable = 0.0
    if draw_open:
        incoming = float(POINTS_OF[open_card])
        if any(c % 13 == open_card % 13 for c in cards_from_mask(remaining)):
            pairs = 1.0
        if _extends(remaining, open_card):
            extends = 1.0
        if remain_pts + POINTS_OF[open_card] <= YANIV_LIMIT:
            result_callable = 1.0
    elif draw_stock:
        pool = cards_from_mask(pool_mask(view))
        if pool:
            incoming = sum(POINTS_OF[c] for c in pool) / len(pool)
            worst = max(POINTS_OF[c] for c in pool)
            if remain_pts + worst <= YANIV_LIMIT:
                result_callable = 1.0
    tasty = 1.0 if POINTS_OF[action.top] <= 4 else 0.0
    kept = 1.0 if _kept_pair(remaining) else 0.0
    vec = (
        _bits(action.discard_mask)
        + _bits(1 << action.top)
        + [
            draw_stock,
            draw_open,
            0.0,
            0.0,  # was "goes out"; every throw draws now, but the slot keeps the layout
            discard_pts / 40.0,
            remain_pts / 40.0,
            remain_n / 5.0,
            POINTS_OF[action.top] / 10.0,
            incoming / 10.0,
            pairs,
            extends,
            is_set,
            is_run,
            is_single,
            kept,
            result_callable,
            tasty,
        ]
    )
    if len(vec) != ACTION_DIM:
        raise RuntimeError(f"action dim {len(vec)} != {ACTION_DIM}")
    # The thrown cards must be a real combination. Guards a bad label.
    if not is_valid_combo(chosen):
        raise RuntimeError("action features requested for an illegal throw")
    return np.asarray(vec, dtype=np.float32)


def action_matrix(view: View, actions: list[Action]) -> np.ndarray:
    if not actions:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.stack([action_features(view, action) for action in actions])

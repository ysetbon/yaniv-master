"""State and action features for the Yaniv net.

The layout is the contract with the TypeScript inference code. Every vector
is float32 and the lengths are asserted.
"""

from __future__ import annotations

import numpy as np

from .engine import (
    DECK_SIZE,
    MAX_PLIES,
    POINTS_OF,
    RANKS,
    SCORE_CUTS,
    YANIV_LIMIT,
    Action,
    View,
    best_keep_points,
    cards_from_mask,
    is_valid_combo,
    mask_points,
    popcount,
    rank_of,
)

# state: 4 card masks, 2 rank counts, suit counts, 19 scalars
STATE_DIM = 4 * DECK_SIZE + 2 * RANKS + 4 + 19
# action: thrown mask, face-up card, 14 scalars
ACTION_DIM = 2 * DECK_SIZE + 14
YANIV_SLOT = 2 * DECK_SIZE + 2
INPUT_DIM = STATE_DIM + ACTION_DIM


def _discard_mask(view: View) -> int:
    mask = 0
    for card in view.discard:
        mask |= 1 << card
    return mask


def pool_mask(view: View) -> int:
    discard = _discard_mask(view)
    all_cards = (1 << DECK_SIZE) - 1
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
    counts = [0.0] * RANKS
    for card in cards_from_mask(mask):
        counts[rank_of(card)] += 1.0
    return [c / 4.0 for c in counts]


def _suit_counts(mask: int) -> list[float]:
    counts = [0.0] * 4
    for card in cards_from_mask(mask):
        if card < 52:
            counts[card // 13] += 1.0
    return [c / 5.0 for c in counts]


def _bits(mask: int, size: int = DECK_SIZE) -> list[float]:
    return [1.0 if (mask >> i) & 1 else 0.0 for i in range(size)]


def state_features(view: View) -> np.ndarray:
    """The state half of the net's input (layout above), filled into one array.

    The search calls this for every move of every playout, so it fills a
    NumPy array directly and works out the unseen cards once.
    """
    discard = _discard_mask(view)
    pool = ((1 << DECK_SIZE) - 1) & ~view.my_hand & ~discard & ~view.opp_known
    pool_cards = cards_from_mask(pool)
    pool_n = len(pool_cards)
    pool_pts = [POINTS_OF[c] for c in pool_cards]
    if pool_n:
        pool_avg = sum(pool_pts) / pool_n
        pool_low = sum(1 for p in pool_pts if p <= 3) / pool_n
    else:
        pool_avg = 0.0
        pool_low = 0.0
    hand_cards = cards_from_mask(view.my_hand)
    my_pts = mask_points(view.my_hand)
    my_size = len(hand_cards)
    keep = best_keep_points(view.my_hand)
    shed = my_pts - keep
    known_cards = cards_from_mask(view.opp_known)
    known_pts = sum(POINTS_OF[c] for c in known_cards)
    need = view.opp_size - len(known_cards)  # as opp_min_points, without a second pass over the pool
    minimum = known_pts + sum(sorted(pool_pts)[:need]) if 0 < need <= pool_n else known_pts
    open_card = view.discard[-1]
    yaniv_legal = 1.0 if my_size > 0 and my_pts <= YANIV_LIMIT else 0.0
    guaranteed = 1.0 if yaniv_legal and my_pts < minimum else 0.0
    # Losing the round with this hand would land exactly on 50 or 100 and cut the score.
    cut_if_lost = 1.0 if view.my_score + my_pts in SCORE_CUTS else 0.0

    vec = np.zeros(STATE_DIM, dtype=np.float32)
    d = DECK_SIZE
    vec[hand_cards] = 1.0
    vec[[d + c for c in view.discard]] = 1.0
    vec[2 * d + open_card] = 1.0
    vec[[3 * d + c for c in known_cards]] = 1.0
    hand_ranks = [0] * RANKS
    for c in hand_cards:
        hand_ranks[rank_of(c)] += 1
    pool_ranks = [0] * RANKS
    for c in pool_cards:
        pool_ranks[rank_of(c)] += 1
    suits = [0] * 4
    for c in hand_cards:
        if c < 52:
            suits[c // 13] += 1
    at = 4 * d
    vec[at:at + RANKS] = [n / 4.0 for n in hand_ranks]
    vec[at + RANKS:at + 2 * RANKS] = [n / 4.0 for n in pool_ranks]
    vec[at + 2 * RANKS:at + 2 * RANKS + 4] = [n / 5.0 for n in suits]
    vec[at + 2 * RANKS + 4:] = (
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
        cut_if_lost,
    )
    return vec


def _kept_pair(mask: int) -> bool:
    seen = [0] * RANKS
    for card in cards_from_mask(mask):
        seen[rank_of(card)] += 1
        if seen[rank_of(card)] >= 2:
            return True
    return False


def action_features(view: View, action: Action) -> np.ndarray:
    if action.kind == "yaniv":
        vec = _bits(0) + _bits(0) + [0.0] * (ACTION_DIM - 2 * DECK_SIZE)
        vec[YANIV_SLOT] = 1.0
        if len(vec) != ACTION_DIM:
            raise RuntimeError(f"action dim {len(vec)}")
        return np.asarray(vec, dtype=np.float32)

    chosen = cards_from_mask(action.discard_mask)
    remaining = view.my_hand & ~action.discard_mask
    remain_pts = mask_points(remaining)
    remain_n = popcount(remaining)
    discard_pts = sum(POINTS_OF[c] for c in chosen)
    is_single = 1.0 if len(chosen) == 1 else 0.0
    is_set = 1.0 - is_single
    draw_stock = 1.0 if action.draw == "stock" else 0.0
    draw_open = 1.0 if action.draw == "open" else 0.0
    open_card = view.discard[-1]
    pairs = 0.0
    incoming = 0.0
    result_callable = 0.0
    if draw_open:
        incoming = float(POINTS_OF[open_card])
        if any(rank_of(c) == rank_of(open_card) for c in cards_from_mask(remaining)):
            pairs = 1.0
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
            0.0,  # YANIV_SLOT: 1 only for a Yaniv call
            discard_pts / 40.0,
            remain_pts / 40.0,
            remain_n / 5.0,
            POINTS_OF[action.top] / 10.0,
            incoming / 10.0,
            pairs,
            is_set,
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
    """`action_features` for every action, stacked, built with the per-view work done once.

    Building these rows was about 60% of the time the search spends, so the
    unseen cards, the open card and each thrown set's numbers are worked out
    once per view or per set instead of once per action. The rows equal
    `action_features` exactly (test_features checks it).
    """
    out = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    if not actions:
        return out
    open_card = view.discard[-1]
    open_pts = POINTS_OF[open_card]
    open_rank = rank_of(open_card)
    pool_stats = None
    per_set: dict[int, tuple] = {}
    base = 2 * DECK_SIZE
    for row, action in enumerate(actions):
        if action.kind == "yaniv":
            out[row, YANIV_SLOT] = 1.0
            continue
        mask = action.discard_mask
        info = per_set.get(mask)
        if info is None:
            chosen = cards_from_mask(mask)
            if not is_valid_combo(chosen):
                raise RuntimeError("action features requested for an illegal throw")
            remaining = view.my_hand & ~mask
            left = cards_from_mask(remaining)
            remain_pts = mask_points(remaining)
            ranks = [rank_of(c) for c in left]
            info = (chosen, remain_pts, len(left), sum(POINTS_OF[c] for c in chosen),
                    1.0 if len(chosen) == 1 else 0.0, open_rank in ranks, len(set(ranks)) < len(ranks))
            per_set[mask] = info
        chosen, remain_pts, remain_n, discard_pts, is_single, pairs_open, kept = info
        pairs = incoming = result_callable = 0.0
        draw_stock = 1.0 if action.draw == "stock" else 0.0
        draw_open = 1.0 if action.draw == "open" else 0.0
        if draw_open:
            incoming = float(open_pts)
            pairs = 1.0 if pairs_open else 0.0
            result_callable = 1.0 if remain_pts + open_pts <= YANIV_LIMIT else 0.0
        elif draw_stock:
            if pool_stats is None:
                pool = cards_from_mask(pool_mask(view))
                pool_stats = (sum(POINTS_OF[c] for c in pool) / len(pool), max(POINTS_OF[c] for c in pool)) if pool \
                    else (None, None)
            if pool_stats[0] is not None:
                incoming = pool_stats[0]
                result_callable = 1.0 if remain_pts + pool_stats[1] <= YANIV_LIMIT else 0.0
        out[row, chosen] = 1.0
        out[row, DECK_SIZE + action.top] = 1.0
        out[row, base:] = (
            draw_stock, draw_open, 0.0, discard_pts / 40.0, remain_pts / 40.0, remain_n / 5.0,
            POINTS_OF[action.top] / 10.0, incoming / 10.0, pairs, 1.0 - is_single, is_single,
            1.0 if kept else 0.0, result_callable, 1.0 if POINTS_OF[action.top] <= 4 else 0.0,
        )
    return out

"""State and action features for the Yaniv net: suit-free, by rank.

No rule looks at suits (there are no runs and jokers have none), so every
card is described by its rank alone: 13 ranks plus the jokers. Moves that
differ only by suit, such as which 7 of three is thrown or which card of a
pair stays face up, are one move for the net (`net_actions`).

The layout is the contract with the TypeScript inference code
(src/lib/yaniv/features.ts); fixtures check that both give the same numbers.
Nets trained on the older 389-number layout still run through
features_v1.py (`score_actions` picks the layout from the net's input size).

State (96):
  0-13   ranks in my hand            (count / 4)
  14-27  ranks thrown and out of play (count / 4)
  28-41  rank of the open card        (one on, the rest off)
  42-55  ranks the opponent is known to hold (count / 4)
  56-69  ranks still unseen           (count / 4)
  70-88  19 numbers: my points, my cards, the opponent's cards, the stock,
         both scores, unseen count, unseen average, unseen low share, Yaniv
         legal, turn, best points after a throw, points shed by it, the
         opponent's known points and cards, its lowest possible total, a sure
         Yaniv, the open card's points, losing now lands on 50 or 100
  89-93  the opponent's card count, 1 to 5 (one on)
  94     the opponent's estimated hand total, hidden cards leaning low / 40
  95     the chance a Yaniv call now is caught (exact, 0 if Yaniv is not legal)

Move (29):
  0-13   rank thrown (one on; all off for a Yaniv call)
  14     cards thrown / 4
  15-28  14 numbers: draw from the deck, take the open card, Yaniv, points
         thrown, points kept, cards kept, the face-up card's points, points
         coming in, pairs with the open card, a set, a single, keeps a pair,
         can call Yaniv next, leaves a low card face up
"""

from __future__ import annotations

import math

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
    rank_of,
)

STATE_DIM = 5 * RANKS + 19 + 5 + 2
ACTION_DIM = RANKS + 1 + 14
INPUT_DIM = STATE_DIM + ACTION_DIM
YANIV_SLOT = RANKS + 1 + 2

# The opponent keeps low cards, so its hidden cards average about 3 points by
# mid-round, not the 6 of a uniform draw. Weight exp(-beta * points), with beta
# by the throws made so far (ply // 2); fitted on card-counter games. The
# search samples hidden cards with the same weights.
HIDDEN_BETA = (0.04, 0.09, 0.14, 0.18, 0.21, 0.24, 0.26)
_LEAN_WEIGHT = [[math.exp(-beta * points) for points in range(11)] for beta in HIDDEN_BETA]


def _discard_mask(view: View) -> int:
    mask = 0
    for card in view.discard:
        mask |= 1 << card
    return mask


def pool_mask(view: View) -> int:
    """Cards this player has not seen: in the stock or hidden in the opponent's hand."""
    return ((1 << DECK_SIZE) - 1) & ~view.my_hand & ~_discard_mask(view) & ~view.opp_known


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


def hidden_lean(view: View, pool_pts: list[int] | None = None) -> float:
    """Expected points of one hidden opponent card, leaning low (HIDDEN_BETA).

    `pool_pts` (the unseen cards' points, lowest card id first) saves working
    them out again when the caller has them.
    """
    if pool_pts is None:
        pool_pts = [POINTS_OF[c] for c in cards_from_mask(pool_mask(view))]
    if not pool_pts:
        return 0.0
    weight = _LEAN_WEIGHT[min(view.ply // 2, len(HIDDEN_BETA) - 1)]
    return sum(weight[p] * p for p in pool_pts) / sum(weight[p] for p in pool_pts)


def assaf_chance(view: View) -> float:
    """Share of the opponent's possible hidden hands that would catch a Yaniv call now.

    Exact over every combination of unseen cards, counted by total points,
    so the TypeScript port gives the same number.
    """
    mine = mask_points(view.my_hand)
    known = cards_from_mask(view.opp_known)
    room = mine - sum(POINTS_OF[c] for c in known)  # the hidden cards catch the call at this total or less
    need = view.opp_size - len(known)
    pool = [POINTS_OF[c] for c in cards_from_mask(pool_mask(view))]
    if need <= 0:
        return 1.0 if room >= 0 else 0.0
    if need > len(pool):
        return 1.0
    if room < 0:
        return 0.0
    # ways[k][s]: ways to pick k unseen cards totalling s points (s <= room).
    ways = [[0] * (room + 1) for _ in range(need + 1)]
    ways[0][0] = 1
    for points in pool:
        for k in range(need, 0, -1):
            row, below = ways[k], ways[k - 1]
            for s in range(room, points - 1, -1):
                row[s] += below[s - points]
    return sum(ways[need]) / math.comb(len(pool), need)


def _ranks(cards, out: np.ndarray, at: int) -> None:
    for c in cards:
        out[at + rank_of(c)] += 0.25


def state_features(view: View) -> np.ndarray:
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    discard = _discard_mask(view)
    pool = ((1 << DECK_SIZE) - 1) & ~view.my_hand & ~discard & ~view.opp_known
    pool_cards = cards_from_mask(pool)
    pool_n = len(pool_cards)
    pool_pts = [POINTS_OF[c] for c in pool_cards]
    pool_avg = sum(pool_pts) / pool_n if pool_n else 0.0
    pool_low = sum(1 for p in pool_pts if p <= 3) / pool_n if pool_n else 0.0
    hand = cards_from_mask(view.my_hand)
    my_pts = mask_points(view.my_hand)
    keep = best_keep_points(view.my_hand)
    known = cards_from_mask(view.opp_known)
    known_pts = sum(POINTS_OF[c] for c in known)
    need = view.opp_size - len(known)
    minimum = known_pts + sum(sorted(pool_pts)[:need]) if 0 < need <= pool_n else known_pts
    open_card = view.discard[-1]
    yaniv_legal = 1.0 if hand and my_pts <= YANIV_LIMIT else 0.0

    _ranks(hand, vec, 0)
    _ranks(view.discard[:-1], vec, RANKS)
    vec[2 * RANKS + rank_of(open_card)] = 1.0
    _ranks(known, vec, 3 * RANKS)
    _ranks(pool_cards, vec, 4 * RANKS)
    at = 5 * RANKS
    vec[at:at + 19] = (
        my_pts / 40.0,
        len(hand) / 5.0,
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
        (my_pts - keep) / 40.0,
        known_pts / 40.0,
        len(known) / 5.0,
        minimum / 40.0,
        1.0 if yaniv_legal and my_pts < minimum else 0.0,
        POINTS_OF[open_card] / 10.0,
        1.0 if view.my_score + my_pts in SCORE_CUTS else 0.0,
    )
    vec[at + 19 + min(5, max(1, view.opp_size)) - 1] = 1.0
    vec[at + 24] = (known_pts + max(0, need) * hidden_lean(view, pool_pts)) / 40.0
    vec[at + 25] = assaf_chance(view) if yaniv_legal else 0.0
    return vec


def action_key(action: Action) -> tuple:
    """What the net can tell apart about a move: the rank and number thrown, and the draw."""
    if action.kind == "yaniv":
        return ("yaniv",)
    mask = action.discard_mask
    return ("play", rank_of((mask & -mask).bit_length() - 1), mask.bit_count(), action.draw)


def net_actions(actions: list[Action]) -> tuple[list[Action], list[int]]:
    """One move per `action_key`, and for each original action the index of its move."""
    unique: list[Action] = []
    where: dict[tuple, int] = {}
    index = []
    for action in actions:
        key = action_key(action)
        if key not in where:
            where[key] = len(unique)
            unique.append(action)
        index.append(where[key])
    return unique, index


def action_matrix(view: View, actions: list[Action]) -> np.ndarray:
    """Move features, one row per action (pass `net_actions(...)[0]` to skip suit-only twins)."""
    out = np.zeros((len(actions), ACTION_DIM), dtype=np.float32)
    if not actions:
        return out
    open_card = view.discard[-1]
    open_pts = POINTS_OF[open_card]
    open_rank = rank_of(open_card)
    pool_stats = None
    per_set: dict[int, tuple] = {}
    base = RANKS + 1
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
            info = (rank_of(chosen[0]), len(chosen), remain_pts, len(left), sum(POINTS_OF[c] for c in chosen),
                    open_rank in ranks, len(set(ranks)) < len(ranks))
            per_set[mask] = info
        rank, count, remain_pts, remain_n, discard_pts, pairs_open, kept = info
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
        single = 1.0 if count == 1 else 0.0
        top_pts = POINTS_OF[action.top]
        out[row, rank] = 1.0
        out[row, RANKS] = count / 4.0
        out[row, base:] = (
            draw_stock, draw_open, 0.0, discard_pts / 40.0, remain_pts / 40.0, remain_n / 5.0,
            top_pts / 10.0, incoming / 10.0, pairs, 1.0 - single, single,
            1.0 if kept else 0.0, result_callable, 1.0 if top_pts <= 4 else 0.0,
        )
    return out


def action_features(view: View, action: Action) -> np.ndarray:
    return action_matrix(view, [action])[0]


def score_actions(net, view: View, actions: list[Action]) -> np.ndarray:
    """The net's score for every action. Suit-only twins share one score.

    A net with the older 389-number input reads features_v1 instead, so nets
    trained before the suit-free layout still play and can still teach.
    """
    if not actions:
        return np.zeros(0, dtype=np.float32)
    if net.dims[0] == INPUT_DIM:
        unique, index = net_actions(actions)
        state = np.repeat(state_features(view)[None, :], len(unique), axis=0)
        scores = net.forward(np.concatenate([state, action_matrix(view, unique)], axis=1))[:, 0]
        return scores[index]
    from . import features_v1

    if net.dims[0] != features_v1.INPUT_DIM:
        raise ValueError(f"a net with {net.dims[0]} inputs fits neither the {INPUT_DIM}-number layout "
                         f"nor the older {features_v1.INPUT_DIM}-number one")
    state = np.repeat(features_v1.state_features(view)[None, :], len(actions), axis=0)
    return net.forward(np.concatenate([state, features_v1.action_matrix(view, actions)], axis=1))[:, 0]

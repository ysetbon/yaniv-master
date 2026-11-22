"""A search player: imagine the opponent's hidden cards, play each move out.

For a decision, the player samples `samples` complete deals consistent with
what it can see (the opponent's known cards stay, the rest of the opponent's
hand and the stock order are drawn from the unseen cards). Each candidate move
is applied to every sampled deal and the round is played to its end by a
playout policy for both seats. The move with the best average value wins.

The value is the round's point margin from the mover's side (the opponent's
added points minus the mover's), plus `MATCH_BONUS` when the round ends the
match: a match won or lost outweighs any single round. Every candidate sees
the same sampled deals, so their difference is not luck of the draw.

The playout policy is the card counter at first. Once a net has learned from
the search, the net can play the playouts instead, which makes the search a
better teacher: the loop the search recipe in train.py runs.
"""

from __future__ import annotations

import math
import random
from typing import Callable

import numpy as np

from .engine import (
    MATCH_LIMIT,
    MAX_PLIES,
    POINTS_OF,
    Action,
    State,
    View,
    apply_action,
    cards_from_mask,
    legal_actions,
    make_view,
    resolve_showdown,
)
from .features import action_matrix, pool_mask, state_features

MATCH_BONUS = 100.0

# The opponent throws high cards and keeps low ones, so its hidden cards are
# lower than a uniform draw from the unseen cards: about 3 points a card by
# mid-round against a uniform guess of 6. Hidden cards are drawn with weight
# exp(-beta * points), beta growing with the throws made so far (ply // 2).
# Fitted on card-counter games so the sampled average matches the real one.
HIDDEN_BETA = (0.04, 0.09, 0.14, 0.18, 0.21, 0.24, 0.26)

Chooser = Callable[[View, list[Action], random.Random], Action]


def deal_from_view(view: View, rng: random.Random) -> State:
    """A full state consistent with `view`.

    The opponent's hidden cards lean low (HIDDEN_BETA); the stock is the rest
    of the unseen cards in random order.
    """
    pool = cards_from_mask(pool_mask(view))
    need = view.opp_size - view.opp_known.bit_count()
    beta = HIDDEN_BETA[min(view.ply // 2, len(HIDDEN_BETA) - 1)]
    # Weighted draw without replacement: the `need` largest keys u ** (1 / w).
    keyed = sorted(pool, key=lambda c: rng.random() ** math.exp(beta * POINTS_OF[c]), reverse=True)
    hidden, stock = keyed[:need], keyed[need:]
    rng.shuffle(stock)
    opp = view.opp_known
    for card in hidden:
        opp |= 1 << card
    if len(stock) != view.stock_size:
        raise RuntimeError(f"unseen cards {len(pool)} do not match hand {need} + stock {view.stock_size}")
    me, other = view.seat, 1 - view.seat
    hands = [0, 0]
    known = [0, 0]
    scores = [0, 0]
    hands[me], hands[other] = view.my_hand, opp
    known[me], known[other] = view.my_known, view.opp_known
    scores[me], scores[other] = view.my_score, view.opp_score
    return State(hands=hands, known=known, stock=stock, discard=list(view.discard), scores=scores,
                 turn=me, ply=view.ply)


def _copy(state: State) -> State:
    return State(hands=list(state.hands), known=list(state.known), stock=list(state.stock),
                 discard=list(state.discard), scores=list(state.scores), turn=state.turn, ply=state.ply)


def playout(state: State, policy: Chooser, rng: random.Random) -> State:
    """Play the round to its end with `policy` in both seats."""
    while not state.terminal:
        if state.ply >= MAX_PLIES:
            resolve_showdown(state)
            break
        view = make_view(state, state.turn)
        actions = legal_actions(view)
        if not actions:
            resolve_showdown(state)
            break
        apply_action(state, policy(view, actions, rng))
    return state


def round_value(state: State, me: int) -> float:
    """Point margin for `me`, plus MATCH_BONUS either way if the round ended the match."""
    added = state.result.added
    value = float(added[1 - me] - added[me])
    mine, theirs = state.scores[me], state.scores[1 - me]
    if max(mine, theirs) >= MATCH_LIMIT and mine != theirs:
        value += MATCH_BONUS if mine < theirs else -MATCH_BONUS
    return value


def sample_values(view: View, candidates: list[Action], policy: Chooser, samples: int,
                  rng: random.Random) -> np.ndarray:
    """Value of each candidate (columns) on each of `samples` shared sampled deals (rows)."""
    values = np.zeros((max(1, samples), len(candidates)))
    for row in range(samples):
        deal = deal_from_view(view, rng)
        for i, action in enumerate(candidates):
            state = _copy(deal)
            apply_action(state, action)
            playout(state, policy, rng)
            values[row, i] = round_value(state, view.seat)
    return values


def evaluate_moves(view: View, candidates: list[Action], policy: Chooser, samples: int,
                   rng: random.Random) -> np.ndarray:
    """Average value of each candidate over `samples` shared sampled deals."""
    return sample_values(view, candidates, policy, samples, rng).mean(axis=0)


def pick_move(values: np.ndarray, default: int, confidence: float) -> int:
    """The default move unless another beats it by `confidence` standard errors.

    With a few dozen sampled deals two close moves often swap places by
    luck. Switching on every such swap throws away a good default move, so a
    rival must beat the default on the same deals by a clear margin.
    """
    if values.shape[1] <= 1:
        return default
    diffs = values - values[:, [default]]
    means = diffs.mean(axis=0)
    n = values.shape[0]
    se = diffs.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.full(values.shape[1], np.inf)
    clear = [i for i in range(values.shape[1]) if i != default and means[i] > confidence * se[i] and means[i] > 0]
    return max(clear, key=lambda i: means[i]) if clear else default


def counter_prior(view: View, actions: list[Action]) -> list[Action]:
    """Plays ranked best first by the card counter's own scoring."""
    from .heuristic import _score_action

    pool = cards_from_mask(pool_mask(view))
    plays = [a for a in actions if a.kind == "play"]
    return sorted(plays, key=lambda a: _score_action(view, a, pool))


def net_prior(net) -> Callable[[View, list[Action]], list[Action]]:
    def rank(view: View, actions: list[Action]) -> list[Action]:
        plays = [a for a in actions if a.kind == "play"]
        if not plays:
            return []
        state = np.repeat(state_features(view)[None, :], len(plays), axis=0)
        scores = net.forward(np.concatenate([state, action_matrix(view, plays)], axis=1))[:, 0]
        return [plays[i] for i in np.argsort(-scores, kind="stable")]

    return rank


def candidate_moves(view: View, actions: list[Action], default: Action,
                    prior: Callable[[View, list[Action]], list[Action]], width: int) -> list[Action]:
    """The policy's own move first, then the `width` best plays by `prior`, then Yaniv when legal."""
    out = [default]
    for action in prior(view, actions)[:width] + [a for a in actions if a.kind == "yaniv"]:
        if action not in out:
            out.append(action)
    return out


def search_chooser(policy: Chooser, prior: Callable[[View, list[Action]], list[Action]], samples: int = 48,
                   width: int = 4, seed: int = 0, confidence: float = 1.5) -> Chooser:
    """A chooser that plays the policy's move unless the search finds a clearly better one."""
    own = random.Random(seed)

    def choose(view: View, actions: list[Action], rng=None) -> Action:
        default = policy(view, actions, own)
        candidates = candidate_moves(view, actions, default, prior, width)
        if len(candidates) <= 1:
            return default
        values = sample_values(view, candidates, policy, samples, own)
        return candidates[pick_move(values, 0, confidence)]

    return choose


def collect_search(net, n_rounds: int, seed: int, samples: int, width: int, progress=None,
                   match_scores: bool = True, confidence: float = 1.5) -> list[tuple[View, int]]:
    """Label positions for the net to copy: (view, index of the searched move in legal_actions).

    The net plays the card counter, seats alternating; 40% of rounds start
    from random match scores when `match_scores` is set, so positions near
    the 100 limit are covered. Every decision with a real choice, on both
    sides of the table, is searched with the net as prior and playout policy,
    and the game goes on with the move each side would really play.
    """
    from .engine import new_round
    from .heuristic import choose_heuristic
    from .train import _tick, net_chooser

    rng = random.Random(seed)
    play_net = net_chooser(net, 0.0)
    searcher = random.Random(seed ^ 0x5EA4C4)
    prior = net_prior(net)
    rows: list[tuple[View, int]] = []
    for i in range(n_rounds):
        _tick(progress, i + 1, n_rounds)
        scores = [rng.randint(0, 95), rng.randint(0, 95)] if match_scores and rng.random() < 0.4 else [0, 0]
        net_seat = i % 2
        state = new_round(scores, (i // 2) % 2, rng)
        while not state.terminal:
            if state.ply >= MAX_PLIES:
                resolve_showdown(state)
                break
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                resolve_showdown(state)
                break
            default = play_net(view, actions, searcher)
            candidates = candidate_moves(view, actions, default, prior, width)
            if len(candidates) > 1:
                values = sample_values(view, candidates, play_net, samples, searcher)
                rows.append((view, actions.index(candidates[pick_move(values, 0, confidence)])))
            mover = play_net if state.turn == net_seat else choose_heuristic
            apply_action(state, mover(view, actions, rng))
    return rows

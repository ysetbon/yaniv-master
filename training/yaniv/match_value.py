"""The chance of winning the match from any score, so the search can play for matches.

A round's points matter differently at different scores: at 92 a ten-point
round ends the match, at 20 it hardly matters, and landing exactly on 50 or
100 cuts a score by 50. So the search can value a move by how much it changes
the chance of winning the match, instead of by the round's point margin.

The table comes from how rounds really end: rounds of the current net against
the card counter give a list of (points I add, points they add). Pooled from
both seats, that describes two equal players, and repeated substitution
(value iteration) gives, for every pair of scores below 100, the chance that
the player on the first score wins.
"""

from __future__ import annotations

import numpy as np

from .engine import MATCH_LIMIT, SCORE_CUTS

SIZE = MATCH_LIMIT  # scores 0..99 are still playing


def _cut(score: np.ndarray) -> np.ndarray:
    out = score.copy()
    for at, to in SCORE_CUTS.items():
        out[score == at] = to
    return out


def win_table(outcomes: list[tuple[int, int]], iterations: int = 200) -> np.ndarray:
    """P[mine, theirs]: the chance the player on `mine` wins the match, for scores 0..99.

    `outcomes` are (points I add, points the opponent adds) for finished
    rounds; each is also counted the other way round, so the table describes
    two equal players. A match tied at the end counts as half a win.
    """
    pooled = [(a, b) for a, b in outcomes] + [(b, a) for a, b in outcomes]
    pairs, counts = np.unique(np.asarray(pooled, dtype=np.int64), axis=0, return_counts=True)
    probs = counts / counts.sum()
    mine = np.arange(SIZE)[:, None]
    theirs = np.arange(SIZE)[None, :]
    table = np.full((SIZE, SIZE), 0.5)
    for _ in range(iterations):
        new = np.zeros((SIZE, SIZE))
        for (da, db), p in zip(pairs, probs):
            a = _cut(np.broadcast_to(mine + da, (SIZE, SIZE)))
            b = _cut(np.broadcast_to(theirs + db, (SIZE, SIZE)))
            over = (a >= MATCH_LIMIT) | (b >= MATCH_LIMIT)
            ended = np.where(a < b, 1.0, np.where(a > b, 0.0, 0.5))
            playing = table[np.minimum(a, SIZE - 1), np.minimum(b, SIZE - 1)]
            new += p * np.where(over, ended, playing)
        if np.abs(new - table).max() < 1e-9:
            table = new
            break
        table = new
    return table


def win_chance(table: np.ndarray, mine: int, theirs: int) -> float:
    """The chance of winning from these scores; a finished match is 1, 0 or a half."""
    if mine >= MATCH_LIMIT or theirs >= MATCH_LIMIT:
        return 1.0 if mine < theirs else 0.0 if mine > theirs else 0.5
    return float(table[mine, theirs])


def round_outcomes(net, n_rounds: int, seed: int) -> list[tuple[int, int]]:
    """(net's points, counter's points) for `n_rounds` rounds of the net against the card counter."""
    import random

    from .engine import play_round
    from .heuristic import choose_heuristic
    from .train import net_chooser

    rng = random.Random(seed)
    play = net_chooser(net, 0.0)
    out = []
    for i in range(n_rounds):
        if i % 2 == 0:
            state = play_round(play, choose_heuristic, rng, [0, 0], (i // 2) % 2)
            out.append((state.result.added[0], state.result.added[1]))
        else:
            state = play_round(choose_heuristic, play, rng, [0, 0], (i // 2) % 2)
            out.append((state.result.added[1], state.result.added[0]))
    return out

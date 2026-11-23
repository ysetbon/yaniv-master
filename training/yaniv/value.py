"""A value net: how a round will end, read off a position instead of played out.

The search teacher (search.py) values a move by imagining the opponent's
hidden cards and playing the round to its end with the policy net, dozens of
times. Each playout is slow (a whole round of moves) and noisy (every future
draw is luck), so with 48 imagined deals two close moves rarely separate, and
the teacher mostly agrees with the net it is teaching.

This is the AlphaZero idea, adapted to a card game with hidden cards:

1. The policy net plays itself. Every position it reaches is stored with how
   the round really ended for each player (the same value the search uses:
   the chance of winning the match after the round, or the point margin).
2. A second net, the value net, learns to predict that from the position.
   It sees the whole imagined deal (both hands, what each player knows) but
   not the stock order, so it predicts the average over the future draws.
3. The search looks `depth` moves ahead with the policy net on each imagined
   deal, then asks the value net instead of playing to the end.

One evaluation is then a couple of moves and one net call instead of a whole
round: cheaper, so the search can imagine more deals and try more moves, and
far less noisy, because the luck of the future draws is averaged out. Each
pass trains the value net again on the newest policy, so the two improve
together.
"""

from __future__ import annotations

import random

import numpy as np

from .engine import MAX_PLIES, State, apply_action, legal_actions, make_view, new_round, resolve_showdown
from .features import STATE_DIM, state_features
from .network import MLP

VALUE_DIM = 2 * STATE_DIM + 1  # my view, the opponent's view, and whether I am to move
VALUE_SCALE = 100.0  # targets are search values (percentage points or point margins) / 100


def value_features(state: State, me: int) -> np.ndarray:
    """The position from `me`'s side: what each player sees, and whose turn it is.

    The opponent's view holds its real hand, so with both views the net has
    the whole deal except the order of the stock.
    """
    out = np.empty(VALUE_DIM, dtype=np.float32)
    out[:STATE_DIM] = state_features(make_view(state, me))
    out[STATE_DIM:2 * STATE_DIM] = state_features(make_view(state, 1 - me))
    out[-1] = 1.0 if state.turn == me else 0.0
    return out


def value_net(seed: int = 5) -> MLP:
    """193 inputs, two hidden layers of 128, one output: the expected value / 100."""
    net = MLP([VALUE_DIM, 128, 128, 1], seed=seed)
    net.W[-1] *= 100.0  # the policy-style tiny last layer would make the first passes learn slowly
    return net


def predict(net: MLP, states: list[State], me: int) -> np.ndarray:
    """Search values (same units as search.round_value) for `me` in each state."""
    if not states:
        return np.zeros(0)
    x = np.stack([value_features(s, me) for s in states])
    return net.forward(x)[:, 0].astype(np.float64) * VALUE_SCALE


def collect_value(policy_net, n_rounds: int, seed: int, table=None, match_scores: bool = True,
                  explore: float = 0.04, progress=None) -> tuple[np.ndarray, np.ndarray]:
    """Positions of the policy net playing itself, and how each round ended.

    Returns (features, targets): one row per position and player. 40% of
    rounds start from random match scores (when `match_scores`), as in the
    search's own rounds. With chance `explore` a move is the card counter's
    or a random play instead of the net's, so the value net also sees the
    positions the search's other candidate moves lead to; only positions
    after a round's last such move are kept, because the value net predicts
    how the round ends when the net plays on.
    """
    from .heuristic import choose_heuristic
    from .search import round_value
    from .train import _tick, net_chooser

    rng = random.Random(seed)
    play = net_chooser(policy_net, 0.0)
    feats: list[np.ndarray] = []
    targets: list[float] = []
    for i in range(n_rounds):
        _tick(progress, i + 1, n_rounds)
        scores = [rng.randint(0, 95), rng.randint(0, 95)] if match_scores and rng.random() < 0.4 else [0, 0]
        state = new_round(scores, i % 2, rng)
        seen: list[tuple[np.ndarray, np.ndarray]] = []
        while not state.terminal:
            if state.ply >= MAX_PLIES:
                resolve_showdown(state)
                break
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                resolve_showdown(state)
                break
            seen.append((value_features(state, 0), value_features(state, 1)))
            if rng.random() < explore:
                seen.clear()  # earlier positions' futures now include a move the net would not make
                plays = [a for a in actions if a.kind == "play"] or actions
                action = choose_heuristic(view, actions, rng) if rng.random() < 0.5 else rng.choice(plays)
            else:
                action = play(view, actions, rng)
            apply_action(state, action)
        if not state.terminal:
            resolve_showdown(state)
        if not seen:
            continue
        v0, v1 = round_value(state, 0, table), round_value(state, 1, table)
        for f0, f1 in seen:
            feats += [f0, f1]
            targets += [v0, v1]
    if not feats:
        return np.zeros((0, VALUE_DIM), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.stack(feats), (np.asarray(targets, dtype=np.float32) / VALUE_SCALE)


def train_value(net: MLP, x: np.ndarray, y: np.ndarray, epochs: int = 4, lr: float = 1e-3, batch: int = 256,
                seed: int = 0, progress=None) -> float:
    """Least squares on (features, target / 100). Returns the last epoch's mean squared error."""
    rng = np.random.default_rng(seed)
    n = len(y)
    loss = float("nan")
    steps = epochs * (-(-n // batch))
    done = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch):
            ids = order[start:start + batch]
            pred = net.forward(x[ids], cache=True)[:, 0]
            err = pred - y[ids]
            total += float(np.sum(err * err))
            net.zero_grad()
            net.backward_acc((2.0 * err)[:, None])
            net.step(len(ids), lr)
            done += 1
            if progress is not None and (done == steps or done % max(1, steps // 50) == 0):
                progress(done, steps)
        loss = total / max(1, n)
    return loss


def explained(net: MLP, x: np.ndarray, y: np.ndarray) -> float:
    """Share of the targets' variance the net explains on (x, y): 1 is perfect, 0 is no better than the mean."""
    if len(y) < 2:
        return float("nan")
    pred = net.forward(x)[:, 0]
    var = float(np.var(y))
    return 1.0 - float(np.mean((pred - y) ** 2)) / var if var > 0 else float("nan")

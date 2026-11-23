"""A learned decider and expert nets: a mixture of experts.

The decider is a small net that reads the table (the state part of the
input) and gives each expert a weight; the weights add up to 1. Every expert
scores the move from the full input, as a plain net does, and the move's
score is the experts' scores mixed by those weights:

    score(table, move) = sum over k of  weight_k(table) * expert_k(table, move)

Everything trains together, so each expert can specialise in the positions
the decider sends it, and the decider learns where those boundaries are.

It starts warm: every expert is a copy of a trained net, and the decider is
first taught the opening / middle / endgame rule (`stage_of_state`), so the
mixture begins playing exactly like the net it came from and only then
learns to split.
"""

from __future__ import annotations

import numpy as np

from .engine import MAX_PLIES
from .features import INPUT_DIM, RANKS, STATE_DIM
from .network import MLP

STAGES = ("opening", "middle", "endgame")
_SCALARS = 5 * RANKS  # where the 19 numbers start in the state features


def stage_of_state(states: np.ndarray) -> np.ndarray:
    """The rule decider, read from state feature rows: 0 opening, 1 middle, 2 endgame.

    Opening: fewer than 4 throws so far. Endgame: my hand is 12 points or
    less, the opponent holds 2 cards or fewer, or the stock is down to 10.
    """
    states = np.atleast_2d(states)
    my_pts = states[:, _SCALARS] * 40.0
    opp_size = states[:, _SCALARS + 2] * 5.0
    stock = states[:, _SCALARS + 3] * 40.0
    ply = states[:, _SCALARS + 10] * MAX_PLIES
    out = np.ones(len(states), dtype=np.int64)
    out[(my_pts <= 12.5) | (opp_size <= 2.5) | (stock <= 10.5)] = 2
    out[ply < 3.5] = 0
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MoE:
    """Same interface as MLP where the trainer, the search and the workers use it."""

    def __init__(self, gate: MLP, experts: list[MLP]):
        if gate.dims[0] != STATE_DIM or gate.dims[-1] != len(experts):
            raise ValueError("the decider reads the state and gives one weight per expert")
        self.gate = gate
        self.experts = experts
        self._weights: np.ndarray | None = None
        self._scores: np.ndarray | None = None

    @property
    def dims(self) -> list[int]:
        return self.experts[0].dims

    def describe(self) -> str:
        return f"a decider ({' → '.join(map(str, self.gate.dims))}) and {len(self.experts)} nets of " \
               f"{' → '.join(map(str, self.dims))}"

    def weights_for(self, states: np.ndarray) -> np.ndarray:
        return _softmax(self.gate.forward(states))

    def forward(self, x: np.ndarray, cache: bool = False) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        weights = _softmax(self.gate.forward(x[:, :STATE_DIM], cache=cache))
        scores = np.concatenate([e.forward(x, cache=cache) for e in self.experts], axis=1)
        if cache:
            self._weights, self._scores = weights, scores
        return (weights * scores).sum(axis=1, keepdims=True).astype(np.float32)

    def zero_grad(self) -> None:
        self.gate.zero_grad()
        for e in self.experts:
            e.zero_grad()

    def backward_acc(self, dy: np.ndarray) -> None:
        d = dy[:, 0]
        w, s = self._weights, self._scores
        for k, expert in enumerate(self.experts):
            expert.backward_acc((w[:, k] * d)[:, None])
        dw = s * d[:, None]  # dL/d weight_k
        self.gate.backward_acc(w * (dw - (w * dw).sum(axis=1, keepdims=True)))  # through the softmax

    def step(self, n_examples: int, lr: float) -> None:
        self.gate.step(n_examples, lr)
        for e in self.experts:
            e.step(n_examples, lr)

    def clone(self) -> "MoE":
        return MoE(self.gate.clone(), [e.clone() for e in self.experts])

    def to_numpy(self) -> "MoE":
        return self


def warm_moe(base: MLP, states: np.ndarray, experts: int = 3, hidden: int = 32, seed: int = 5,
             steps: int = 3000, lr: float = 1e-2) -> tuple[MoE, float]:
    """Every expert a copy of `base`; the decider taught the stage rule on `states`.

    Returns the mixture and how often the decider's top expert matches the rule.
    """
    if base.dims[0] != INPUT_DIM:
        raise ValueError(f"the experts start as copies of a {INPUT_DIM}-input net")
    gate = MLP([STATE_DIM, hidden, experts], seed=seed)
    target = stage_of_state(states) % experts
    rng = np.random.default_rng(seed)
    for _ in range(steps if len(states) else 0):
        ids = rng.integers(0, len(states), 256)
        probs = _softmax(gate.forward(states[ids], cache=True))
        probs[np.arange(len(ids)), target[ids]] -= 1.0
        gate.zero_grad()
        gate.backward_acc(probs)
        gate.step(len(ids), lr)
    agree = float((gate.forward(states).argmax(axis=1) == target).mean()) if len(states) else 0.0
    return MoE(gate, [base.clone() for _ in range(experts)]), agree

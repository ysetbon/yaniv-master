"""A kit for writing a net's weights by hand, unit by unit, with names.

    kit = Kit()
    kit.unit(1, "low_keep", {"pts_kept": -40.0}, bias=7.0, note="how far under 8 the kept cards are")
    kit.unit(2, "safe_call", {"yaniv": 1.0, "sure_yaniv": 1.0}, bias=-1.0, note="a Yaniv call that cannot be caught")
    kit.out({"low_keep": 0.5, "safe_call": 100.0})
    net = kit.build()          # a network.MLP, shape [125, ..., 1]

Every unit is ReLU(sum of weight * source + bias). A source is one of the 125
inputs (names in INPUTS) or a unit of an earlier layer. A source more than
one layer back is carried forward by pass-through units the kit adds; that
is exact because every input and every hidden unit is 0 or more. The output
is linear: sum of weight * source + bias, and the net plays the legal move
with the highest output.

Input scales (features.py): rank counts are count / 4; points are / 40
(`my_pts`, `pts_kept`, `pts_thrown`, `opp_min`, `opp_est`, ...) except single
cards, which are / 10 (`open_pts`, `top_pts`, `incoming`); card counts are
/ 5 (`my_cards`, `n_kept`, ...) except `n_thrown`, which is / 4; scores are
/ 120; `stock` and `unseen_n` are / 40; `ply` is / 48.
"""

from __future__ import annotations

import numpy as np

from .features import ACTION_DIM, INPUT_DIM, STATE_DIM
from .network import MLP

RANK_NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "JOKER"]
RANK_POINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 0]

INPUTS: list[str] = (
    [f"hand_{r}" for r in RANK_NAMES]        # ranks in my hand, count / 4
    + [f"gone_{r}" for r in RANK_NAMES]      # ranks thrown and out of play (under the open card), count / 4
    + [f"open_{r}" for r in RANK_NAMES]      # rank of the open card, one on
    + [f"known_{r}" for r in RANK_NAMES]     # ranks the opponent is known to hold, count / 4
    + [f"unseen_{r}" for r in RANK_NAMES]    # ranks still unseen (stock or the opponent's hidden cards), count / 4
    + [
        "my_pts",        # my hand's points / 40
        "my_cards",      # my cards / 5
        "opp_cards",     # the opponent's cards / 5
        "stock",         # cards in the stock / 40
        "my_score",      # my match score / 120
        "opp_score",     # the opponent's match score / 120
        "unseen_n",      # unseen cards / 40
        "unseen_avg",    # their average points / 10
        "unseen_low",    # share of them worth 3 or less
        "yaniv_legal",   # 1 if I may call Yaniv now
        "ply",           # throws made so far / 48
        "best_keep",     # fewest points I can hold after one throw / 40
        "best_shed",     # points that throw sheds / 40
        "known_pts",     # points of the opponent's known cards / 40
        "known_n",       # how many of its cards are known / 5
        "opp_min",       # the opponent's lowest possible total / 40
        "sure_yaniv",    # 1 if Yaniv is legal and my points are under opp_min: the call cannot be caught
        "open_pts",      # the open card's points / 10
        "lands_on_cut",  # 1 if my score + my points is exactly 50 or 100
        "opp_has_1", "opp_has_2", "opp_has_3", "opp_has_4", "opp_has_5",  # the opponent's card count, one on
        "opp_est",       # the opponent's estimated total, hidden cards leaning low / 40
        "assaf",         # exact chance a Yaniv call now is caught, hidden cards uniform (0 if Yaniv is not legal)
    ]
    + [f"throw_{r}" for r in RANK_NAMES]     # rank thrown, one on (all off for a Yaniv call)
    + [
        "n_thrown",      # cards thrown / 4
        "draw_deck",     # 1 if the move draws from the deck
        "take_open",     # 1 if the move takes the open card
        "yaniv",         # 1 if the move is the Yaniv call (every other move number is then 0)
        "pts_thrown",    # points thrown / 40
        "pts_kept",      # points kept, before the draw / 40
        "n_kept",        # cards kept / 5
        "top_pts",       # points of the card left face up / 10
        "incoming",      # points coming in: the open card's, or the unseen average for a deck draw / 10
        "pairs_open",    # 1 if the open card is taken and matches a kept card's rank
        "is_set",        # 1 if two or more cards are thrown
        "is_single",     # 1 if one card is thrown
        "keeps_pair",    # 1 if the kept cards hold two of one rank
        "callable_next", # 1 if Yaniv is sure to be legal next turn (kept + open card, or kept + the worst unseen card, is 7 or less)
        "low_top",       # 1 if the card left face up is worth 4 or less
    ]
)
assert len(INPUTS) == INPUT_DIM == STATE_DIM + ACTION_DIM
INDEX = {name: i for i, name in enumerate(INPUTS)}


class Kit:
    def __init__(self) -> None:
        self.layers: list[dict[str, tuple[dict, float, str]]] = []  # layers[k - 1][name] = (weights, bias, note)
        self.level: dict[str, int] = {name: 0 for name in INPUTS}
        self.output: tuple[dict, float] = ({}, 0.0)

    def unit(self, layer: int, name: str, weights: dict[str, float], bias: float = 0.0, note: str = "") -> str:
        """A ReLU unit in hidden layer `layer` (1 is the first). Sources must come from earlier layers."""
        if name in self.level:
            raise ValueError(f"{name} is already defined")
        for source in weights:
            if source not in self.level:
                raise ValueError(f"{name}: unknown source {source}")
            if self.level[source] >= layer:
                raise ValueError(f"{name} (layer {layer}) reads {source} from layer {self.level[source]}")
        while len(self.layers) < layer:
            self.layers.append({})
        self.layers[layer - 1][name] = (dict(weights), float(bias), note)
        self.level[name] = layer
        return name

    def out(self, weights: dict[str, float], bias: float = 0.0) -> None:
        """Add terms to the linear output (may be called several times; weights for one source add up)."""
        merged, old_bias = self.output
        for source, weight in weights.items():
            if source not in self.level:
                raise ValueError(f"output: unknown source {source}")
            merged[source] = merged.get(source, 0.0) + weight
        self.output = (merged, old_bias + bias)

    def build(self) -> MLP:
        depth = len(self.layers)
        layers = [dict(layer) for layer in self.layers]
        # Carry a source forward through every layer between it and its reader,
        # working back from the output so a carry is itself carried further.
        for at in range(depth, 0, -1):
            sources = set(self.output[0]) if at == depth else {s for w, _, _ in layers[at].values() for s in w}
            for source in sorted(sources):
                if self.level[source] < at and source not in layers[at - 1]:
                    layers[at - 1][source] = ({source: 1.0}, 0.0, "carried forward")
        dims = [INPUT_DIM] + [len(layer) for layer in layers] + [1]
        net = MLP(dims, seed=0)
        before = dict(INDEX)
        for k, layer in enumerate(layers):
            weight = np.zeros((dims[k], dims[k + 1]), dtype=np.float32)
            bias = np.zeros(dims[k + 1], dtype=np.float32)
            for col, (name, (weights, b, _)) in enumerate(layer.items()):
                for source, w in weights.items():
                    weight[before[source], col] = w
                bias[col] = b
            net.W[k], net.b[k] = weight, bias
            before = {name: col for col, name in enumerate(layer)}
        weight = np.zeros((dims[-2], 1), dtype=np.float32)
        for source, w in self.output[0].items():
            weight[before[source], 0] = w
        net.W[-1], net.b[-1] = weight, np.array([self.output[1]], dtype=np.float32)
        self.built_layers = layers
        return net

    def describe(self) -> str:
        """Every unit the author wrote, by layer, with its note."""
        lines = []
        for k, layer in enumerate(self.layers, start=1):
            lines.append(f"Layer {k}: {len(layer)} units")
            for name, (weights, bias, note) in layer.items():
                terms = " ".join(f"{w:+g}*{s}" for s, w in weights.items())
                lines.append(f"  {name} = relu({terms} {bias:+g})  # {note}")
        terms = " ".join(f"{w:+g}*{s}" for s, w in self.output[0].items())
        lines.append(f"Output = {terms} {self.output[1]:+g}")
        return "\n".join(lines)

    def trace(self, x: np.ndarray) -> dict[str, float]:
        """The value of every named unit (and the output, as "OUT") for one 125-number input row."""
        values = {name: float(x[i]) for name, i in INDEX.items()}
        for layer in self.layers:
            for name, (weights, bias, _) in layer.items():
                values[name] = max(0.0, sum(w * values[s] for s, w in weights.items()) + bias)
        values["OUT"] = sum(w * values[s] for s, w in self.output[0].items()) + self.output[1]
        return values

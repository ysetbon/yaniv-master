"""Two hand-written nets as ONE model file: the main net, then the 4-card net after the hand-over.

    python -m training.yaniv.handmade5_pack MAIN.json [ENDGAME.json] --out public/model/yaniv-net-handmade.json
    python -m training.yaniv.handmade5_pack MAIN.json --when mine --at 4 --check 20000

The player for the real game is hybrid.switch_chooser(main, endgame, at=4, when="both"): the main
net chooses until both hands hold 4 cards or fewer, then the 4-card hand-written net does. The
browser loads one file, so this writes the pair in the game's existing "decider and expert nets"
format (moe.py, network.save_net, src/lib/ai/net.ts): expert 0 is the main net, expert 1 the endgame
net, and the decider is a hand-written net that does nothing but the switch.

The decider (96 state inputs -> 2 units -> 2 outputs), with S = 8000 and G = 2000. For "both":

    my_big  = ReLU(S * (my cards  - at - 0.5))      0 if my hand holds `at` cards or fewer, else >= S/2
    opp_big = ReLU(S * (opp cards - at - 0.5))
    output for the main net    = my_big + opp_big   ("mine" / "opp": only that one unit)
    output for the endgame net = G

so the two outputs are at least 2000 apart, on the right side. ("either" turns it round: two units
that fire when a hand is small, added up for the endgame net, and G for the main net.) The game
turns the outputs into weights with a softmax, exp(output - largest) / sum. exp(-2000) is exactly 0
both in float32 (numpy; anything under about -104 is) and in float64 (the browser's Math.exp;
anything under about -745.2 is), so the weights are exactly (1, 0) or (0, 1), and the mixed score
1 * chosen + 0 * other is the chosen net's score, bit for bit, as long as the other net's score is
finite (0 * 1000, a Yaniv row, is 0).

The two nets must have one shape (moe.MoE stacks the experts' layers). A narrower layer gets
units whose weights are all 0: they give 0 and nothing reads them. A shallower net gets
pass-through layers (weight 1 from a unit to its copy) after one of its hidden layers, which is
exact because a hidden unit is never negative; the layer is chosen to make the file smallest.

Nothing here trains or searches for weights: every number is written from the rule above.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from .features import INPUT_DIM, STATE_DIM
from .handmade_kit import INPUTS
from .hybrid import WHEN
from .moe import MoE
from .network import MLP, load_net, save_net

STEP = 8000.0  # S: the slope of the card-count units, per card
GAP = 2000.0   # G: the constant output; the outputs end up at least this far apart
ENDGAME = Path("public/model/yaniv-net-4cards-yaniv7-handmade.json")
OUT = Path("public/model/yaniv-net-handmade.json")
PROTECTED = ("yaniv-net.json", "yaniv-net-4cards-yaniv7.json", "yaniv-net-4cards-yaniv7-handmade.json")
_MY, _OPP = INPUTS.index("my_cards"), INPUTS.index("opp_cards")  # both are count / 5


def decider(when: str = "both", at: int = 4) -> MLP:
    """The hand-written decider: output 0 for the main net, output 1 for the endgame net."""
    if when not in WHEN:
        raise ValueError(f"when={when!r}: give one of {WHEN}")
    if not 0 <= at <= 5:
        raise ValueError("the hand-over size is a card count from 0 to 5")
    gate = MLP([STATE_DIM, 2, 2], seed=0)
    for w in gate.W:
        w.fill(0.0)
    for b in gate.b:
        b.fill(0.0)
    # "either" is the mirror image: its units fire on a SMALL hand and feed the endgame net.
    sign, fed, constant = (-1.0, 1, 0) if when == "either" else (1.0, 0, 1)
    for unit, source in enumerate((_MY, _OPP)):
        gate.W[0][source, unit] = sign * 5.0 * STEP          # the input is cards / 5
        gate.b[0][unit] = -sign * STEP * (at + 0.5)
    for unit, used in enumerate((when != "opp", when != "mine")):
        gate.W[1][unit, fed] = 1.0 if used else 0.0
    gate.b[1][constant] = GAP
    return gate


def _plain(net, name: str) -> MLP:
    if hasattr(net, "experts") or not hasattr(net, "W"):
        raise ValueError(f"the {name} net must be a plain net, not a decider with experts")
    if net.dims[0] != INPUT_DIM or net.dims[-1] != 1:
        raise ValueError(f"the {name} net must read the {INPUT_DIM} inputs and give one score, not {net.dims}")
    return net


def _deepen(net: MLP, layers: int, after: int) -> MLP:
    """`net` with pass-through layers added after hidden layer `after` (1 = the first), `layers` in all."""
    extra = layers - len(net.W)
    width = net.dims[after]
    dims = net.dims[:after + 1] + [width] * extra + net.dims[after + 1:]
    out = MLP(dims, seed=0)
    eye = np.eye(width, dtype=np.float32)
    weights = net.W[:after] + [eye] * extra + net.W[after:]
    biases = net.b[:after] + [np.zeros(width, dtype=np.float32)] * extra + net.b[after:]
    for i in range(layers):
        out.W[i], out.b[i] = weights[i].astype(np.float32).copy(), biases[i].astype(np.float32).copy()
    return out


def _widen(net: MLP, dims: list[int]) -> MLP:
    """`net` with every layer as wide as `dims`: the new units have weight 0 in and out."""
    out = MLP(dims, seed=0)
    for i, (w, b) in enumerate(zip(net.W, net.b)):
        out.W[i].fill(0.0)
        out.b[i].fill(0.0)
        out.W[i][:w.shape[0], :w.shape[1]] = w
        out.b[i][:b.shape[0]] = b
    return out


def _size(dims: list[int]) -> int:
    return sum(a * b + b for a, b in zip(dims, dims[1:]))


def same_shape(first: MLP, second: MLP) -> tuple[MLP, MLP]:
    """Both nets, computing exactly what they did, in one common shape."""
    layers = max(len(first.W), len(second.W))
    shallow, deep = (first, second) if len(first.W) < len(second.W) else (second, first)
    if len(shallow.W) < layers:
        if len(shallow.W) < 2:
            raise ValueError("a net with no hidden layer cannot be given pass-through layers")
        # Put the pass-through layers after the hidden layer that makes the common shape smallest.
        def cost(after: int) -> int:
            padded = _deepen(shallow, layers, after).dims
            return _size([max(a, b) for a, b in zip(padded, deep.dims)])

        best = min(range(1, len(shallow.W)), key=cost)
        shallow = _deepen(shallow, layers, best)
        first, second = (shallow, deep) if len(first.W) < len(second.W) else (deep, shallow)
    dims = [max(a, b) for a, b in zip(first.dims, second.dims)]
    return _widen(first, dims), _widen(second, dims)


def pack(main: MLP, endgame: MLP, when: str = "both", at: int = 4) -> MoE:
    """One model that plays `main` until the hand-over and `endgame` after it (hybrid.switch_chooser)."""
    a, b = same_shape(_plain(main, "main"), _plain(endgame, "endgame"))
    return MoE(decider(when, at), [a, b])


def check(main: MLP, endgame: MLP, when: str = "both", at: int = 4, decisions: int = 20000, seed: int = 9,
          packed: MoE | None = None) -> dict:
    """Play real-game rounds and compare the packed model with the switch player at every decision.

    Both seats are the switch player; one move in ten is random, so the positions are not only the
    ones two good players reach. Rounds start from random match scores four times in ten.

    Two comparisons. "padded": against the switch player built from the packed file's own two
    experts (the nets in their common shape); this must be bit for bit. "original": against the
    switch player built from the nets as given. Numpy's matrix product rounds the last bit
    differently for matrices of different shapes (OpenBLAS picks other kernels), so a net and its
    padded copy can differ by a few units in the last place; when the two nets already have one
    shape nothing is padded and this is bit for bit too.
    """
    from .engine import HAND_SIZE, play_round
    from .features import score_actions, state_features
    from .hybrid import small_enough, switch_chooser
    from .train import net_chooser

    packed = packed or pack(main, endgame, when, at)
    original = switch_chooser(net_chooser(main, 0.0), net_chooser(endgame, 0.0), at, when)
    padded = switch_chooser(net_chooser(packed.experts[0], 0.0), net_chooser(packed.experts[1], 0.0), at, when)
    as_one = net_chooser(packed, 0.0)
    rng = random.Random(seed)
    out = {"decisions": 0, "after_handover": 0, "rounds": 0, "hand_size": HAND_SIZE, "rows": 0,
           "weights_not_exact": 0, "not_finite": 0,
           "moves_differ_padded": 0, "max_score_diff_padded": 0.0, "rows_differ_padded": 0,
           "moves_differ_original": 0, "max_score_diff_original": 0.0, "rows_differ_original": 0,
           "max_abs_score": 0.0}

    def choose(view, actions, rng_):
        small = int(small_enough(view, at, when))
        want = original(view, actions, None)
        ours = score_actions(packed, view, actions)
        weights = packed.weights_for(state_features(view)[None, :])[0]
        out["decisions"] += 1
        out["after_handover"] += small
        out["rows"] += len(ours)
        out["weights_not_exact"] += not (weights[small] == 1.0 and weights[1 - small] == 0.0)
        out["not_finite"] += not bool(np.isfinite(ours).all())
        out["max_abs_score"] = max(out["max_abs_score"], float(np.abs(ours).max()))
        got = as_one(view, actions, None)
        for name, move, nets in (("original", want, (main, endgame)),
                                 ("padded", padded(view, actions, None), packed.experts)):
            theirs = score_actions(nets[small], view, actions)
            out[f"moves_differ_{name}"] += move != got
            out[f"max_score_diff_{name}"] = max(out[f"max_score_diff_{name}"], float(np.abs(ours - theirs).max()))
            out[f"rows_differ_{name}"] += int((ours != theirs).sum())
        return rng.choice(actions) if rng.random() < 0.1 else want

    while out["decisions"] < decisions:
        scores = [rng.randint(0, 99), rng.randint(0, 99)] if rng.random() < 0.4 else [0, 0]
        play_round(choose, choose, rng, scores, out["rounds"] % 2)
        out["rounds"] += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a main net and the 4-card net into one model file")
    parser.add_argument("main", help="the main net (a plain net JSON file)")
    parser.add_argument("endgame", nargs="?", default=str(ENDGAME), help=f"the net after the hand-over (default {ENDGAME})")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--when", choices=WHEN, default="both",
                        help="hand over once both hands are small (default), or mine / the opponent's / either")
    parser.add_argument("--at", type=int, default=4, help="small = this many cards or fewer (default 4)")
    parser.add_argument("--check", type=int, default=0, metavar="N",
                        help="also compare with the switch player on N decisions of real-game rounds")
    parser.add_argument("--nets", metavar="PREFIX",
                        help="also write the two nets in their common shape, PREFIX-main.json and PREFIX-endgame.json: "
                             "the switch player made of those is the packed model bit for bit in Python too")
    args = parser.parse_args()
    out = Path(args.out)
    if out.name in PROTECTED:
        raise SystemExit(f"{out.name} is another player's file; write somewhere else")
    main_net, endgame = load_net(Path(args.main)), load_net(Path(args.endgame))
    packed = pack(main_net, endgame, args.when, args.at)
    rule = {"both": f"both hands hold {args.at} cards or fewer", "mine": f"my hand holds {args.at} cards or fewer",
            "opp": f"the opponent holds {args.at} cards or fewer",
            "either": f"either hand holds {args.at} cards or fewer"}[args.when]
    save_net(packed, out, meta={
        "rules": {"hand": 5, "yaniv_at": 7},
        "handmade": True,
        "source": "training/yaniv/handmade5_pack.py",
        "switch": {"when": args.when, "at": args.at, "main": args.main, "endgame": args.endgame,
                   "main_dims": main_net.dims, "endgame_dims": endgame.dims},
        "note": f"Two hand-written nets. Expert 0 ({args.main}) plays until {rule}; from then on expert 1 "
                f"({args.endgame}) plays. The decider is hand-written too and gives weights of exactly 1 and 0.",
    })
    print(f"wrote {out}: {packed.describe()}, {out.stat().st_size / 1e6:.2f} MB; "
          f"hands over once {rule}")
    if args.nets:
        for name, net in zip(("main", "endgame"), packed.experts):
            save_net(net, Path(f"{args.nets}-{name}.json"), meta={"handmade": True, "padded_copy_of": getattr(args, name)})
    if args.check:
        result = check(main_net, endgame, args.when, args.at, args.check, packed=load_net(out))
        print(json.dumps(result))
        if result["moves_differ_padded"] or result["rows_differ_padded"] or result["weights_not_exact"] or result["not_finite"]:
            raise SystemExit("the packed model does not play as the switch player does")
        if result["moves_differ_original"]:
            print(f"note: {result['moves_differ_original']} of {result['decisions']} moves differ from the nets in "
                  f"their own shapes: near ties decided by the last bit of numpy's matrix product (see check)")


if __name__ == "__main__":
    sys.exit(main())

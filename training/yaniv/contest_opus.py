"""Team opus's contest net: ONE hand-written net for the real game (5 cards each, Yaniv at 7 or less).

    python -m training.yaniv.contest_opus        # writes public/model/yaniv-net-contest-opus.json
    python -m training.yaniv.contest_opus -v     # also prints every unit with its purpose

Starts from handmade5.py (the design `counts`, see results/handmade-5x7.md); every change is a named block below
with its reason and its measured result (results/contest-opus.md). Nothing is trained, fitted or searched.
"""

from __future__ import annotations

import sys
from pathlib import Path

from training.yaniv import handmade5 as H5
from training.yaniv.handmade import add_score_units, add_yaniv_units, finish_yaniv, fold_output
from training.yaniv.handmade_kit import Kit
from training.yaniv.network import save_net

MODEL_PATH = Path(__file__).resolve().parents[2] / "public" / "model" / "yaniv-net-contest-opus.json"
MAX_HIDDEN = 1020


def build(extras=(), **over):
    """extras: functions f(kit) adding named units and output terms (the team's blocks; see EXTRAS)."""
    cfg = H5.config(**over)
    kit = Kit()
    H5.add_throw_units(kit, cfg)
    for f in extras:
        f(kit)
    reasons = add_yaniv_units(kit)
    reasons += add_score_units(kit)
    finish_yaniv(kit, reasons)
    fold_output(kit)
    net = kit.build()
    assert sum(net.dims[1:-1]) <= MAX_HIDDEN, net.dims
    return net, kit


from training.yaniv.contest_opus_endgame import add_endgame_units  # noqa: E402
from training.yaniv.contest_opus_mine import add_mine_units  # noqa: E402

# The blocks kept in the shipped net, each passed by play-outs against two opponents (results/contest-opus.md):
#  - add_endgame_units: rule S, the swap. Yaniv legal, the opponent holds 3-4 cards, catch chance >= 0.005 and the
#    open card 2+ points under my highest card: throw that card and take the open card instead of calling.
#    +4.8 to +5.3 a time against the counter, +3.5 to +8.9 against the trained net, ~0.007 times a round. 12 units.
#  - add_mine_units: TRIPLE. I hold a pair of R (R worth 4+) and the open card is an R: keep the pair, throw another
#    card worth 4+, take the open R and throw all three next turn (+5, one card charge, on that row). Changed
#    decisions: +3.4 a time against the counter, +2.9 against the trained net, ~0.045 times a round. 10 units.
EXTRAS: list = [add_endgame_units, add_mine_units]


def main(path: Path = MODEL_PATH) -> None:
    net, kit = build(EXTRAS)
    if "-v" in sys.argv:
        print(kit.describe())
    save_net(net, path, meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                              "source": "training/yaniv/contest_opus.py"})
    print(f"{path.name}: layers {net.dims}, {sum(net.dims[1:-1])} hidden units, written by hand")


if __name__ == "__main__":
    main(Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else MODEL_PATH)

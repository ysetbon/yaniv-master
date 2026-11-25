"""shaper: a hand-written net for the real game (5 cards, Yaniv at 7) - hand shaping with 5 cards.

    python -m training.yaniv.handmade5_designs.shaper [OUT.json] [NAME=value ...]

It plays while either player still holds 5 cards (the 4-card net handmade.py takes over after that), but it
is complete: it scores every legal move, calls Yaniv by handmade.py's rule and keeps its match-score units.

Nothing is trained, fitted or searched. The throw value is handmade.py's (held points, one-step plan, heavy-
card charge, catch bonus, open-card habits, feeding penalty), copied here so its constants can be set for the
5-card game, plus the SHAPING units below. Every constant is named in CFG with where its number comes from;
`NAME=value` on the command line builds a variant by hand (each variant measured is in
results/handmade-5x7-logs/shaper.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

from training.yaniv import handmade as H
from training.yaniv.handmade_kit import RANK_NAMES, RANK_POINTS, Kit
from training.yaniv.network import save_net

SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/shaper")
RANKS = RANK_NAMES

CFG = dict(
    # ---- handmade.py's throw constants (4-card values unless a note says otherwise)
    CARD=5.0,          # points charged per heavy card left
    CARD_SOON=2.0,     # charge taken back when the opponent has 1-2 cards
    TWO_LIGHT=0.5,     # a 2 counts as half a heavy card
    PAIR_OPEN=5.0,     # taking an open card onto a kept card of its rank
    OPEN_TELL=1.5,     # per point above 3 of an unpaired open card taken
    CALLABLE_VS4=3.0,  # Yaniv surely legal next turn while the opponent holds 4 cards
    LOW_TOP=1.0,       # leaving a card worth <= 4 face up
    # ---- shaping units (0 = unit not built)
    SET5=0.0,          # bonus for throwing a set while I hold 5 cards (the first shrink)
    OUTS=0.0,          # points per unit of sum over kept ranks of unseen cards of that rank / N_TYP
    TWOS55=0.0,        # bonus for throwing a set of 2s while both players hold 5 cards
    SET55=0.0,         # bonus for throwing any set while both players hold 5 cards
)


def add_throw_units(kit: Kit, c: dict) -> None:
    """handmade.add_throw_units with the constants taken from c (same units, same names)."""
    kit.unit(1, "light_thrown", {"n_thrown": 4.0, "throw_A": 4.0, "throw_JOKER": 4.0}, bias=-4.0,
             note="cards thrown if the thrown rank is A or joker (light cards leaving), else 0")
    kit.unit(1, "twos_thrown", {"n_thrown": 4.0, "throw_2": 4.0}, bias=-4.0, note="cards thrown if they are 2s")
    kit.unit(1, "took_light_open", {"take_open": 1.0, "open_A": 1.0, "open_JOKER": 1.0}, bias=-1.0,
             note="1 if the move takes an open A or joker")
    kit.unit(1, "took_two_open", {"take_open": 1.0, "open_2": 1.0}, bias=-1.0, note="1 if the move takes an open 2")
    kit.unit(1, "deck_heavy", {"draw_deck": 2.0 + H.DECK_HEAVY, "unseen_low": -1.0}, bias=-2.0,
             note="deck draw: chance the drawn card is heavy")
    tl = c["TWO_LIGHT"]
    heavy = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
             "hand_2": -4.0 * tl, "twos_thrown": tl,
             "take_open": 1.0, "took_light_open": -1.0, "took_two_open": -tl,
             "deck_heavy": 1.0}
    kit.unit(2, "heavy_soon", {**heavy, "opp_has_1": 6.0, "opp_has_2": 6.0}, bias=-6.0,
             note="heavy cards after the move, only when the opponent has 1-2 cards")
    out = {"pts_kept": -40.0, "incoming": -10.0, "heavy_soon": c["CARD_SOON"]}
    for s, w in heavy.items():
        out[s] = out.get(s, 0.0) - c["CARD"] * w

    kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
             note="points above 3 of an open card taken that pairs nothing")
    kit.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, bias=-1.0,
             note="Yaniv surely legal next turn while the opponent still has 4 cards")
    out.update({"pairs_open": c["PAIR_OPEN"], "open_tell": -c["OPEN_TELL"], "callable_vs4": c["CALLABLE_VS4"],
                "low_top": -c["LOW_TOP"]})

    for r, p in zip(RANKS, RANK_POINTS):
        kit.unit(1, f"feed_{r}", {f"throw_{r}": 1.0, f"known_{r}": 4.0}, bias=-1.0,
                 note=f"leaves a {r} face up while the opponent is known to hold one (it takes a set)")
        out[f"feed_{r}"] = -(H.FEED[0] + H.FEED[1] * p)
    kit.out(out)

    H.add_plan_units(kit)
    H.add_catch_units(kit)


def add_shape_units(kit: Kit, c: dict) -> None:
    if c["SET5"]:
        kit.unit(1, "set_from5", {"is_set": 1.0, "my_cards": 5.0}, bias=-5.0,
                 note="1 if a set is thrown while I hold 5 cards: the hand shrinks below 5 for good")
        kit.out({"set_from5": c["SET5"]})
    if c["TWOS55"]:
        kit.unit(1, "twos_55", {"is_set": 1.0, "throw_2": 1.0, "opp_has_5": 1.0, "my_cards": 5.0}, bias=-7.0,
                 note="1 if a set of 2s is thrown while both players hold 5 cards (nobody can call soon, and "
                      "a 5-card hand holding 2-2 cannot get to 7 points: the 2s are heavy cards here)")
        kit.out({"twos_55": c["TWOS55"]})
    if c["SET55"]:
        kit.unit(1, "set_55", {"is_set": 1.0, "opp_has_5": 1.0, "my_cards": 5.0}, bias=-6.0,
                 note="1 if a set is thrown while both players hold 5 cards")
        kit.out({"set_55": c["SET55"]})
    if c["OUTS"]:
        total = {}
        for r, p in zip(RANKS, RANK_POINTS):
            if p == 0:
                continue
            g = {f"grp_keep_{r}": 1.0 / p, f"grp_left_{r}": 1.0 / p}        # cards of rank r kept (layer 1)
            kit.unit(2, f"kept1_{r}_a", g, note=f"my {r}s kept")
            kit.unit(2, f"kept1_{r}_b", g, bias=-1.0, note=f"my {r}s kept beyond the first: a - b = [I keep a {r}]")
            kit.unit(3, f"outs_{r}", {f"kept1_{r}_a": 1.0, f"kept1_{r}_b": -1.0, f"unseen_{r}": 1.0}, bias=-1.0,
                     note=f"unseen {r}s / 4 if I keep a {r}: cards that would pair it")
            total[f"outs_{r}"] = c["OUTS"] * 4.0 / H.N_TYP
        kit.out(total)


def build(**over):
    c = {**CFG, **over}
    kit = Kit()
    add_throw_units(kit, c)
    add_shape_units(kit, c)
    reasons = H.add_yaniv_units(kit)
    reasons += H.add_score_units(kit)
    H.finish_yaniv(kit, reasons)
    H.fold_output(kit)
    return kit.build(), kit


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if "=" not in a]
    over = {k: float(v) for k, v in (a.split("=") for a in sys.argv[1:] if "=" in a)}
    unknown = set(over) - set(CFG)
    if unknown:
        raise SystemExit(f"unknown constants: {sorted(unknown)}")
    out = Path(args[0]) if args else SCRATCH / "shaper.json"
    net, kit = build(**over)
    save_net(net, out, meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                             "source": "training/yaniv/handmade5_designs/shaper.py", "constants": {**CFG, **over}})
    print(f"{out}: layers {net.dims}, {sum(net.dims[1:-1])} hidden units, written by hand")

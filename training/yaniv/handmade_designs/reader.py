"""reader: a hand-written net built from what the trained 4-card net does differently from the card counter.

Every weight is chosen by reasoning (see SCRATCH/logs/reader-findings.md for the behaviour it copies in words).

Throws: a move's value is minus the hand it leaves, counted as
    points after the move (kept points + the card coming in)
  + C per "heavy" card after the move (any card but A, 2 or a joker),
with C = 5 (3 when the opponent is down to 1-2 cards and may call soon). The card charge is why the
trained net throws a pair of 3s or more instead of a higher single (the hand shrinks for good) yet keeps
a pair of aces or 2s; a bonus for taking an open card that pairs a held card (a set to throw next turn);
and a small charge for taking an unpaired open card worth 4+ (it tells the opponent what I hold, and a deck
draw may pair). Net effect: unpaired open cards are taken only at <= 3 points (the counter takes <= 5).
Yaniv: call when the call cannot be caught, or when `assaf` is below a threshold T that falls with the
ply (opponents keep low cards, so the uniform `assaf` understates the risk more and more), rises with the
gap between the opponent's estimated hand and mine, and is a little higher at 6-7 points. T is about
0.02-0.04, nearly flat in my points (the counter allows up to 0.40 at 1 point, 0.18 at 3 points).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("YANIV_HAND_SIZE", "4")
os.environ.setdefault("YANIV_CALL_AT", "7")

from training.yaniv.handmade_kit import Kit  # noqa: E402
from training.yaniv.network import save_net  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-yaniv-master/ecf1221d-8c18-559f-9f3b-91b1597261e5/scratchpad"

CARD = 5.0          # points charged per heavy card left in the hand
CARD_SOON = 2.0     # charge taken back when the opponent has 1-2 cards (points matter more than cards then)
PAIR_OPEN = 9.0     # taking an open card that pairs a kept card: a set to throw next turn
TRIPLE_OPEN = 5.0   # extra when it makes a triple (one more card goes with the set)
CALLABLE_VS4 = 3.0  # bonus for a move that makes Yaniv legal next turn while the opponent holds 4 cards
LOW_TOP = 1.0       # leaving a card worth <= 4 face up feeds the counter (it takes those 95% of the time)
TWO_LIGHT = 0.5     # a 2 counts as half a heavy card (A and joker as none)
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "JOKER"]
OPEN_TELL = 1.5     # per point above 3 of an unpaired open card taken
T_EARLY = 0.025      # Yaniv threshold on assaf: T_FLOOR + relu(T_EARLY - T_SLOPE*ply) + gap and 6-7 point terms
T_SLOPE = 0.00125
T_FLOOR = 0.015
GAP_SLOPE = 0.002    # threshold change per point of (opponent estimate - my points) ...
GAP_CENTER = 5.0     # ... around a gap of 5 ...
GAP_CAP = 15.0       # ... with the gap capped here
DECK_HEAVY = 1.07   # chance a deck card is heavy ~ 1 - share of unseen worth <= 3, plus the 3s


def build():
    k = Kit()
    # ---- layer 1 ----
    k.unit(1, "light_thrown", {"n_thrown": 4.0, "throw_A": 4.0, "throw_JOKER": 4.0}, bias=-4.0,
           note="cards thrown if the thrown rank is A or joker (light cards leaving), else 0")
    k.unit(1, "twos_thrown", {"n_thrown": 4.0, "throw_2": 4.0}, bias=-4.0, note="cards thrown if they are 2s, else 0")
    k.unit(1, "took_light_open", {"take_open": 1.0, "open_A": 1.0, "open_JOKER": 1.0}, bias=-1.0,
           note="1 if the move takes an open A or joker")
    k.unit(1, "took_two_open", {"take_open": 1.0, "open_2": 1.0}, bias=-1.0, note="1 if the move takes an open 2")
    for r in RANKS:
        k.unit(1, f"triple_{r}", {f"open_{r}": 1.0, f"hand_{r}": 4.0, "take_open": 1.0, f"throw_{r}": -4.0}, bias=-3.5,
               note=f"0.5 if the move takes an open {r} onto two kept {r}s (a triple to throw next turn)")
    k.unit(1, "deck_heavy", {"draw_deck": 2.0 + DECK_HEAVY, "unseen_low": -1.0}, bias=-2.0,
           note="for a deck draw: chance the drawn card is heavy (1.07 - share of unseen worth <= 3)")
    k.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
           note="points above 3 of an open card taken that pairs nothing")
    k.unit(1, "early", {"ply": -T_SLOPE * 48}, bias=T_EARLY,
           note="Yaniv threshold part that falls with the ply: T_EARLY at ply 0, 0 from ply T_EARLY/T_SLOPE")
    k.unit(1, "gap", {"opp_est": 40.0, "my_pts": -40.0}, note="opponent's estimated total minus mine, >= 0")
    k.unit(1, "gap_cap", {"opp_est": 40.0, "my_pts": -40.0}, bias=-GAP_CAP, note="the gap above GAP_CAP (capped)")
    k.unit(1, "high_call", {"my_pts": 40.0}, bias=-5.0, note="my points above 5 (6-7 call a little more readily)")
    k.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, bias=-1.0,
           note="the move makes Yaniv sure to be legal next turn while the opponent still has 4 cards (a near-safe call)")
    k.unit(1, "sure_call", {"sure_yaniv": 1.0, "yaniv": 1.0}, bias=-1.0, note="the Yaniv move when it cannot be caught")

    # heavy cards after the move, as a linear expression of inputs and layer-1 units
    heavy = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
             "hand_2": -4.0 * TWO_LIGHT, "twos_thrown": TWO_LIGHT,
             "take_open": 1.0, "took_light_open": -1.0, "took_two_open": -TWO_LIGHT, "deck_heavy": 1.0}

    # ---- layer 2 ----
    k.unit(2, "heavy_soon", {**heavy, "opp_has_1": 6.0, "opp_has_2": 6.0}, bias=-6.0,
           note="heavy cards after the move, only when the opponent has 1-2 cards (else 0)")
    t_terms = {"early": 1.0, "gap": GAP_SLOPE, "gap_cap": -GAP_SLOPE, "high_call": 0.005}
    margin = {s: 500.0 * w for s, w in t_terms.items()}
    margin.update({"assaf": -500.0, "yaniv": 100.0})
    k.unit(2, "call", margin, bias=-100.0 + 500.0 * (T_FLOOR - GAP_SLOPE * GAP_CENTER),
           note="Yaniv move with assaf below T = T_FLOOR + relu(T_EARLY - T_SLOPE*ply) + GAP_SLOPE*(gap - GAP_CENTER, gap capped) "
                "+ 0.005*(pts-5)+; 500*(T - assaf), gated by the Yaniv flag")
    k.unit(2, "call_sat", {**margin}, bias=-100.0 + 500.0 * (T_FLOOR - GAP_SLOPE * GAP_CENTER) - 1.0, note="call - 1 (to saturate the step)")

    # ---- output ----
    out = {"pts_kept": -40.0, "incoming": -10.0, "pairs_open": PAIR_OPEN, "open_tell": -OPEN_TELL,
           "heavy_soon": CARD_SOON, "yaniv": -500.0, "call": 1000.0, "call_sat": -1000.0, "sure_call": 1000.0,
           "callable_vs4": CALLABLE_VS4, "low_top": -LOW_TOP}
    for r in RANKS:
        out[f"triple_{r}"] = 2.0 * TRIPLE_OPEN
    for s, w in heavy.items():
        out[s] = out.get(s, 0.0) - CARD * w
    k.out(out)
    return k.build(), k


if __name__ == "__main__":
    net, kit = build()
    print(kit.describe())
    name = sys.argv[1] if len(sys.argv) > 1 else "reader"
    save_net(net, Path(SCRATCH) / "nets" / f"{name}.json")

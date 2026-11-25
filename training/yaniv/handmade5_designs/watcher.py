"""watcher: a hand-written net for the 5-card phases of the real game (5 cards each, Yaniv at 7), written from
what good players were SEEN to do there.

    python -m training.yaniv.handmade5_designs.watcher [OUT.json]

Method (results/handmade-5x7-logs/watcher.md has the tables): on 84,000 real-game decisions where somebody
held 5 cards I recorded what the trained net, the 4-card hand-written net (training/yaniv/handmade.py, "the
bar") and the card counter would each play, tabulated the choices by situation, and where the trained net and
the bar disagreed I played the round out both ways on the same deal (matched rollouts) to see who is right.
The two nets agree on 83-92% of throws; the rules below are the places where the rollouts said the bar is
wrong in a 5-card phase. Nothing is copied from the trained net's weights and nothing is fitted: each rule is
a sentence, each constant a round number.

The net is the bar (same units, imported from handmade.py: held points, the one-step plan, the heavy-card
charge, the catch bonus, the Yaniv reasons, the match-score units) with these terms changed:

  RULE 1 (a pair of 2s goes before a high single while the opponent holds 4-5 cards). The trained net throws
     22 over a higher single 86-96% of the time at 5 v 5 and ~55% at 5 v 4, the bar never; on 2,143 matched
     rollouts throwing the 22 was worth +5.1 points at 5 v 5, +3.6 at 4 v 5, +2.7 at 5 v 4 and 0 at 5 v 3.
     Reason: an opponent with 4-5 cards is far from calling, so what I hold now costs little and a hand one
     card smaller is worth more than 6 points less. In words: "while the opponent holds 4 or 5 cards a 2 is a
     heavy card like any other" (the bar counts it as half a heavy card). Aces and jokers stay light: nobody
     throws AA and the rollouts agree (see the log).
  RULE 2 (do not hang on to a high card just because the opponent holds its twin). The bar charges up to 6
     points for leaving face up a rank the opponent is known to hold, so it throws a 5 and keeps a J for the
     rest of the round; the trained net and the counter both throw the J, and on 274 matched rollouts that
     was worth +4.0 points. In a 5-card phase the charge is cut to a tie-break (FEED5).
  (Things watched and NOT changed, because the rollouts did not separate the two nets: the trained net throws
  the lighter of two pairs first (+0.7 +- 0.6), takes unpaired open 4s late in the round (-0.2 +- 0.8), and
  sometimes throws a lower single keeping a high one (-0.4 +- 0.7).)
"""

from __future__ import annotations

import sys
from pathlib import Path

from training.yaniv import handmade as hm
from training.yaniv.handmade_kit import RANK_NAMES, RANK_POINTS, Kit
from training.yaniv.network import save_net

SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/watcher")

# ---------------------------------------------------------------- constants
# Everything not named here is the bar's (handmade.py: CARD 5, CARD_SOON 2, PAIR_OPEN 5, OPEN_TELL 1.5, the
# plan's EST_* table, the catch bonus, the Yaniv thresholds, the score units).
TWO_LIGHT = hm.TWO_LIGHT   # 0.5: a 2 counts as half a heavy card ... (bar)
TWO_FULL = 1.0             # ... but as a whole one while both players hold 5 cards (RULE 1: on every position
                           # with 22 + a single worth 5+, throwing the 22 was worth +4.7 points at 5 v 5,
                           # +0.5 at 5 v 4, 0 at 4 v 5, -3 at 5 v 3; the trained net does it 90% / 55% / 60% / 15%)
FEED5 = (0.5, 0.0)         # (flat, per point) for leaving face up a rank the opponent is known to hold: only
                           # a tie-break between equal cards (RULE 2; the bar's is (1.0, 0.5) = 6 for a ten)


def add_throw_units(kit: Kit) -> None:
    """handmade.add_throw_units with RULE 1 and RULE 2 (the rest is the bar's, unit for unit)."""
    CARD, CARD_SOON = hm.CARD, hm.CARD_SOON
    kit.unit(1, "light_thrown", {"n_thrown": 4.0, "throw_A": 4.0, "throw_JOKER": 4.0}, bias=-4.0,
             note="cards thrown if the thrown rank is A or joker (light cards leaving), else 0")
    kit.unit(1, "twos_thrown", {"n_thrown": 4.0, "throw_2": 4.0}, bias=-4.0, note="cards thrown if they are 2s")
    kit.unit(1, "took_light_open", {"take_open": 1.0, "open_A": 1.0, "open_JOKER": 1.0}, bias=-1.0,
             note="1 if the move takes an open A or joker")
    kit.unit(1, "took_two_open", {"take_open": 1.0, "open_2": 1.0}, bias=-1.0, note="1 if the move takes an open 2")
    kit.unit(1, "deck_heavy", {"draw_deck": 2.0 + hm.DECK_HEAVY, "unseen_low": -1.0}, bias=-2.0,
             note="deck draw: chance the drawn card is heavy")
    heavy = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
             "hand_2": -4.0 * TWO_LIGHT, "twos_thrown": TWO_LIGHT,
             "take_open": 1.0, "took_light_open": -1.0, "took_two_open": -TWO_LIGHT,
             "deck_heavy": 1.0}
    kit.unit(2, "heavy_soon", {**heavy, "opp_has_1": 6.0, "opp_has_2": 6.0}, bias=-6.0,
             note="heavy cards after the move, only when the opponent has 1-2 cards")
    out = {"pts_kept": -40.0, "incoming": -10.0, "heavy_soon": CARD_SOON}
    for s, w in heavy.items():
        out[s] = out.get(s, 0.0) - CARD * w

    # RULE 1: the 2s I hold after the move, counted only while the opponent holds 4-5 cards
    # (my_cards is cards / 5, so 30 my_cards - 24 is 6 with 5 cards and 0 or less with fewer)
    kit.unit(2, "twos_5v5", {"hand_2": 4.0, "twos_thrown": -1.0, "took_two_open": 1.0,
                             "opp_has_5": 6.0, "my_cards": 30.0}, bias=-36.0,
             note="2s in my hand after the move, only while both players hold 5 cards (nobody is near calling)")
    out["twos_5v5"] = -CARD * (TWO_FULL - TWO_LIGHT)     # the rest of a heavy card's charge

    kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
             note="points above 3 of an open card taken that pairs nothing")
    kit.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, bias=-1.0,
             note="Yaniv surely legal next turn while the opponent still has 4 cards")
    out.update({"pairs_open": hm.PAIR_OPEN, "open_tell": -hm.OPEN_TELL, "callable_vs4": hm.CALLABLE_VS4,
                "low_top": -hm.LOW_TOP})

    # RULE 2: feeding is only a tie-break
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        kit.unit(1, f"feed_{r}", {f"throw_{r}": 1.0, f"known_{r}": 4.0}, bias=-1.0,
                 note=f"leaves a {r} face up while the opponent is known to hold one")
        out[f"feed_{r}"] = -(FEED5[0] + FEED5[1] * p)
    kit.out(out)

    hm.add_plan_units(kit)
    hm.add_catch_units(kit)


def build():
    kit = Kit()
    add_throw_units(kit)
    reasons = hm.add_yaniv_units(kit)
    reasons += hm.add_score_units(kit)
    hm.finish_yaniv(kit, reasons)
    hm.fold_output(kit)
    return kit.build(), kit


if __name__ == "__main__":
    net, kit = build()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else SCRATCH / "watcher.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_net(net, out, meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                             "source": "training/yaniv/handmade5_designs/watcher.py"})
    print(f"{out}: layers {net.dims}, {sum(net.dims[1:-1])} hidden units, written by hand")

"""watcher: a hand-written net for the 5-card phases of the real game (5 cards each, Yaniv at 7), written from
what good players were SEEN to do there.

    python -m training.yaniv.handmade5_designs.watcher [OUT.json]

Method (results/handmade-5x7-logs/watcher.md has the tables and every script's name): on 84,000 real-game
decisions where somebody held 5 cards I recorded what the trained net, the 4-card hand-written net
(training/yaniv/handmade.py, "the bar") and the card counter would each play, tabulated the choices by
situation, and where the trained net and the bar disagreed I played the round out both ways on the same deal
(matched rollouts, the mover continuing as the bar against the counter) to see who is right. Nothing is
copied from the trained net's weights and nothing is fitted: a rule is a sentence, a constant a round number.

What the watching said: the two nets play the same move on 83% of throws at 5 v 5 and about 90% in the other
5-card phases; half of the differences are which of two ten-point cards to throw (rollouts: no difference),
and in every other kind of difference but one the bar's move was as good or better (it is right to keep a
high card whose twin the opponent holds, right not to take unpaired open 4s and 5s, right to throw the
heavier of two pairs first when the opponent is under 5 cards). So this net IS the bar (same units, imported
from handmade.py: held points, the one-step plan, the heavy-card charge, the catch bonus, the Yaniv reasons,
the match-score units) with the one rule where the trained net was right:

  RULE 1: while I hold 5 cards and the opponent 4 or 5, a pair of 2s goes before a high single.
     Seen: with one pair of 2s and a higher single the trained net throws the 22 86-96% of the time at 5 v 5
     and 46-61% at 5 v 4 (10-29% at 5 v 3); the bar and the counter never do. Rollouts, 22 thrown instead of
     the bar's single: +5.1 +- 0.7 points at 5 v 5 and +2.7 +- 0.8 at 5 v 4 on the positions where the trained
     net does it; on every position with 22 and a single worth 5+: +4.7 +- 0.5 at 5 v 5, +0.6 +- 0.5 at 5 v 4,
     0 at 4 v 5, -3 at 5 v 3. A pair of aces stays (nobody throws it; rollouts -1.4 at 5 v 5, -5 elsewhere).
     Reason: an opponent with 4-5 cards is at least two turns from calling (it calls next turn 1% / 5% of the
     time), so the points I hold now cost little, and a hand one card smaller reaches a callable total a turn
     sooner. In units: in that phase a 2 is a heavy card like any other (TWO_FULL), and a heavy card is charged
     CARD_BIG = 8 points instead of 5, so that 22 (4 points out, one card fewer) beats throwing a ten (10
     points out) even when the plan sees a second ten to throw next turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

from training.yaniv import handmade as hm
from training.yaniv.handmade_kit import RANK_NAMES, RANK_POINTS, Kit
from training.yaniv.network import save_net

SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/watcher")

# ---------------------------------------------------------------- constants
# Everything not named here is the bar's (handmade.py: CARD 5, CARD_SOON 2, PAIR_OPEN 5, OPEN_TELL 1.5, FEED,
# the plan's EST_* table, the catch bonus, the Yaniv thresholds, the score units).
BIG_OPP = (4, 5)           # RULE 1 holds while I hold 5 cards and the opponent holds one of these counts
                           # (rollouts: + at 5 v 5 and 5 v 4, 0 at 4 v 5, - at 5 v 3)
TWO_LIGHT = hm.TWO_LIGHT   # 0.5: a 2 counts as half a heavy card (the bar) ...
TWO_FULL = 1.0             # ... but as a whole one in RULE 1's phase (the trained net throws 22 there, see above)
CARD_BIG = 8.0             # charge per heavy card in RULE 1's phase (the bar's CARD is 5): a card must be worth
                           # more than the 6 points between 22 and a ten; 8 leaves 2 points to spare


def add_throw_units(kit: Kit) -> None:
    """handmade.add_throw_units with RULE 1 added (the rest is the bar's, unit for unit)."""
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

    # RULE 1: while I hold 5 cards and the opponent 4-5 (nobody is near calling) a 2 is a heavy card like any
    # other, and every heavy card is charged CARD_BIG instead of CARD.
    # (my_cards is cards / 5, so 6 [opp has 4-5] + 30 my_cards - 36 is 0 in that phase and -6 or less outside)
    phase = {**{f"opp_has_{n}": 6.0 for n in BIG_OPP}, "my_cards": 30.0}
    kit.unit(2, "twos_big", {"hand_2": 4.0, "twos_thrown": -1.0, "took_two_open": 1.0, **phase}, bias=-36.0,
             note="2s in my hand after the move, only while I hold 5 cards and the opponent 4-5")
    out["twos_big"] = -CARD * (TWO_FULL - TWO_LIGHT)     # the rest of a heavy card's charge
    if CARD_BIG != CARD:
        # heavy cards after the move with the 2s counted whole; a card taken onto its twin is not counted (it
        # leaves with the twin next turn: the bar's PAIR_OPEN = CARD says the same)
        full = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
                "take_open": 1.0, "took_light_open": -1.0, "deck_heavy": 1.0, "pairs_open": -1.0}
        kit.unit(2, "heavy_big", {**full, **phase}, bias=-36.0,
                 note="heavy cards after the move (2s whole, a paired open card not counted), same phase")
        out["heavy_big"] = -(CARD_BIG - CARD)

    kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
             note="points above 3 of an open card taken that pairs nothing")
    kit.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, bias=-1.0,
             note="Yaniv surely legal next turn while the opponent still has 4 cards")
    out.update({"pairs_open": hm.PAIR_OPEN, "open_tell": -hm.OPEN_TELL, "callable_vs4": hm.CALLABLE_VS4,
                "low_top": -hm.LOW_TOP})

    # feeding penalty: the bar's (watched: the trained net ignores it and loses 4 points each time)
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        kit.unit(1, f"feed_{r}", {f"throw_{r}": 1.0, f"known_{r}": 4.0}, bias=-1.0,
                 note=f"leaves a {r} face up while the opponent is known to hold one")
        out[f"feed_{r}"] = -(hm.FEED[0] + hm.FEED[1] * p)
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

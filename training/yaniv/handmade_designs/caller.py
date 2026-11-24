"""A hand-written net whose focus is the Yaniv call (4 cards, Yaniv at 7).

Two separate parts, so the Yaniv logic can be plugged into another design:

  add_throw_units(kit)   a plain throw policy: keep few points, draw cheaply,
                         take a card that pairs, keep pairs, don't leave a low card up.
  add_yaniv_units(kit)   the call decision, as a 0/1 step on the Yaniv row only.

How the two are kept apart (the guarantee). Every throw-part term reads move
numbers (pts_kept, incoming, pairs_open, ...), all of which are 0 on the Yaniv
row, so the throw part scores the Yaniv row 0 and any throw between about -60
and +10 (points kept + incoming are at most ~50, bonuses at most ~10). The
Yaniv part adds -BIG * yaniv + 2 * BIG * step to the output, where step is 1
when calling is right and 0 when it is not, and is forced to 0 on throw rows
by a gate (G * yaniv - G inside its relus). With BIG = 1000 the Yaniv row
scores +1000 (above every throw) or -1000 (below every throw). Only a margin
within 1/K of the decision line gives a fractional step; those are coin flips
anyway.

The call rule (rule "caller"). Diagnostics on card-counter self-play (see the
report) show `assaf` (uniform hidden cards) badly understates the real catch
chance: assaf 0.006 -> 5% caught, 0.03 -> 18%, 0.07 -> 26%, 0.17 -> 32%,
0.24 -> 57%. `opp_est` (hidden cards leaning low) ranks positions much
better. Comparing the call's points with a played-out continuation, the call
pays once opp_est - my_pts clears about 3 points at 4-5 points in hand and
about 4 at 6-7 (a caught call costs my_pts + 30, so a higher hand needs a
wider margin). So:

    margin (points) = 40*opp_est - 40*my_pts - (0.4*my_pts_in_points + 1.3)
                      - 10 * assaf + 100 * sure_yaniv
    call iff margin > 0

i.e. threshold 2.5 at 3 points, 2.9 at 4, 3.3 at 5, 3.7 at 6, 4.1 at 7; a
sure Yaniv always calls. The small assaf term is a guard for positions where
the opponent's estimate is uncertain but plain counting already sees danger.

Rule "counter" writes the card counter's own call (heuristic._call_threshold
on the exact assaf, plus the sure Yaniv) as units, as a baseline.
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
BIG = 1000.0   # the Yaniv row lands at +BIG or -BIG; every throw scores within about [-60, +10]
GATE = 5000.0  # larger than any K * margin, so the step is exactly 0 on throw rows


def add_throw_units(kit: Kit, low_top: float = 1.0, keep_pair: float = 1.0, pair_open: float = 0.7,
                    callable_next: float = 1.5, open_cost: float = 0.0, pair_open_early: float = 0.0,
                    open_cost_by_cards: dict[int, float] | None = None, graded_top: bool = False,
                    rank_supply: float = 0.0) -> None:
    """Score a throw by the points I expect to hold after it (in points), plus small bonuses.

    All terms read move numbers only, so they are 0 on the Yaniv row.
    """
    # Points held after the throw and the draw: kept cards plus what comes in
    # (the open card's points, or the unseen average for a deck draw).
    kit.out({"pts_kept": -40.0, "incoming": -10.0})
    # Taking an open card that pairs a kept card: it goes out with its twin next
    # turn, so most of its points are not really "held". Gate: the open card's
    # points when pairs_open is 1, else 0 (incoming <= 1, so 10*incoming <= 10).
    kit.unit(1, "paired_open_pts", {"incoming": 10.0, "pairs_open": 10.0}, bias=-10.0,
             note="points of an open card taken to pair a kept card (0 otherwise)")
    kit.out({"paired_open_pts": pair_open})
    if pair_open_early:
        # While the opponent still holds 4 cards nobody can call soon, so what counts is the hand after the next
        # throw: a pair taken now is worth more. Gate on pairs_open AND opp_has_4.
        kit.unit(1, "paired_open_pts_early", {"incoming": 10.0, "pairs_open": 10.0, "opp_has_4": 10.0}, bias=-20.0,
                 note="points of an open card taken to pair a kept card, while the opponent holds 4 cards")
        kit.out({"paired_open_pts_early": pair_open_early})
    if open_cost:
        # A deck draw is worth more than its average: a high draw is simply the next card thrown, a low one is kept,
        # and the opponent does not learn it. So taking a plain open card needs to beat the average by a margin.
        kit.out({"take_open": -open_cost})
    # A kept pair sheds two cards in one throw later.
    kit.out({"keeps_pair": keep_pair})
    # A throw that makes Yaniv surely legal next turn.
    kit.out({"callable_next": callable_next})
    if open_cost_by_cards:
        # The same cost, by the opponent's card count: the closer a call, the more the points held now matter.
        for n, cost in open_cost_by_cards.items():
            kit.unit(1, f"take_open_vs{n}", {"take_open": 1.0, f"opp_has_{n}": 1.0}, bias=-1.0,
                     note=f"takes the open card while the opponent holds {n} cards")
            kit.out({f"take_open_vs{n}": -cost})
    if graded_top:
        # Leaving a low card face up hands the opponent a cheap card: 0.4 a point under 4.5 (A 1.4, 2 1.0, 3 0.6,
        # 4 0.2), close to the counter's own top penalty. Gated by "is a throw" so the Yaniv row reads 0.
        kit.unit(1, "cheap_top", {"top_pts": -10.0, "draw_deck": 4.5, "take_open": 4.5},
                 note="how far under 4.5 the card left face up is")
        kit.out({"cheap_top": -0.4})
    else:
        # Leaving a card worth 4 or less face up hands the opponent a cheap card.
        kit.out({"low_top": -low_top})
    if rank_supply:
        # Throw the rank with fewer unseen copies and keep the one a draw can still pair (ties among 10s, mostly).
        names = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "JOKER"]
        for r in names:
            kit.unit(1, f"unseen_of_thrown_{r}", {f"unseen_{r}": 4.0, f"throw_{r}": 4.0}, bias=-4.0,
                     note=f"unseen copies of rank {r} if it is the rank thrown")
            kit.out({f"unseen_of_thrown_{r}": rank_supply})


def _step(kit: Kit, layer: int, name: str, margin: dict[str, float], bias: float, k: float) -> None:
    """Output += 2*BIG*step - BIG*yaniv, step = clip(k*margin + 1, 0, 1) on the Yaniv row, 0 on throws.

    `margin` (with `bias`) is linear in its sources; calling is right when it is >= 0.
    """
    hi = {s: k * w for s, w in margin.items()}
    hi["yaniv"] = hi.get("yaniv", 0.0) + GATE
    kit.unit(layer, f"{name}_a", hi, bias=k * bias + 1.0 - GATE, note="relu(k*margin + 1), Yaniv row only")
    kit.unit(layer, f"{name}_b", dict(hi), bias=k * bias - GATE, note="relu(k*margin), Yaniv row only")
    kit.out({f"{name}_a": 2 * BIG, f"{name}_b": -2 * BIG, "yaniv": -BIG})


def add_yaniv_units(kit: Kit, rule: str = "caller", slope: float = 0.4, base: float = 1.3, assaf_w: float = 10.0,
                    by_cards: dict[str, float] | None = None, flat_cut: float = 0.02) -> None:
    """The Yaniv row scores +BIG when calling is right and -BIG when it is not.

    `by_cards` shifts the margin (in points) by the opponent's card count, e.g. {"opp_has_2": 1.5}.
    """
    if rule == "caller":
        # margin in points: opp_est - my_pts - (slope*my_pts + base) - assaf_w*assaf, +100 for a sure Yaniv.
        margin = {"opp_est": 40.0, "my_pts": -40.0 * (1.0 + slope), "assaf": -assaf_w, "sure_yaniv": 100.0}
        margin.update(by_cards or {})
        _step(kit, 1, "call", margin, bias=-base, k=4.0)
        return
    if rule == "counter":
        # heuristic._call_threshold(points) as 0.04 + a sum of relu(c - points): exact at every whole point.
        p = {"my_pts": -40.0}
        for c in (7, 5, 4, 3, 2, 1):
            kit.unit(1, f"below_{c}", p, bias=float(c), note=f"relu({c} - my points)")
        thr = {"below_7": 0.02, "below_5": 0.02, "below_4": 0.02, "below_3": 0.04, "below_2": 0.02, "below_1": -0.12}
        margin = dict(thr)
        margin["assaf"] = -1.0
        margin["sure_yaniv"] = 1.0  # k*margin stays under GATE
        _step(kit, 2, "ctr_call", margin, bias=0.04, k=1000.0)
        return
    if rule == "flat":
        # One cut on the raw assaf, read off the calibration table: below 0.02 the real catch rate is ~5%.
        _step(kit, 1, "flat_call", {"assaf": -1.0, "sure_yaniv": 1.0}, bias=flat_cut, k=1000.0)
        return
    raise ValueError(rule)


VARIANTS = {
    # name: (rule, yaniv kwargs, throw kwargs)
    "caller_t0": ("caller", {}, {}),
    "caller_ctr": ("counter", {}, {}),
    # The counter's 2-card hands are caught less often than opp_est says, its 3- and 4-card hands more often.
    "caller_n2": ("caller", {"by_cards": {"opp_has_2": 1.5, "opp_has_3": -0.75, "opp_has_4": -0.75}}, {}),
    # Throw policy T1: a plain open card must beat the deck average by 1.5 points; a pairing open card counts
    # 0.5 more of its points while the opponent holds 4 cards.
    "caller_t1": ("caller", {"by_cards": {"opp_has_2": 1.5, "opp_has_3": -0.75, "opp_has_4": -0.75}},
                  {"open_cost": 1.5, "pair_open_early": 0.5}),
    "caller_t1_ctr": ("counter", {}, {"open_cost": 1.5, "pair_open_early": 0.5}),
    # T2: the open-card cost by the opponent's card count (2 at 4 cards, 0.5 at 3, 0 at 2 or 1), and a graded
    # penalty for the card left face up.
    "caller_t2": ("caller", {"by_cards": {"opp_has_2": 1.5, "opp_has_3": -0.75, "opp_has_4": -0.75}},
                  {"open_cost_by_cards": {4: 2.0, 3: 0.5}, "pair_open_early": 0.5, "graded_top": True}),
    # C2 on T1: with the better throw, waiting is worth more. Break-even of opp_est - my_pts, re-measured on T1's
    # own games, is about 3.9 / 4.7 / 5.5 / 6.3 at 4 / 5 / 6 / 7 points, flat in the opponent's card count.
    "caller_c2": ("caller", {"slope": 0.8, "base": 0.7}, {"open_cost": 1.5, "pair_open_early": 0.5}),
    # C1 exactly (no card-count shift) on T1: the 2-card shift did not reproduce on T1's games.
    "caller_t1_c1": ("caller", {}, {"open_cost": 1.5, "pair_open_early": 0.5}),
    # T3 = T1 + throw the rank with fewer unseen copies (-0.3 a copy): keep the rank a deck draw can still pair.
    "caller_t3": ("caller", {}, {"open_cost": 1.5, "pair_open_early": 0.5, "rank_supply": -0.3}),
    # Is opp_est needed? A single flat cut on assaf (call iff assaf < 0.02, or a sure Yaniv), on T1.
    "caller_flat": ("flat", {}, {"open_cost": 1.5, "pair_open_early": 0.5}),
}


BEST = "caller_t1_c1"


def build(variant: str = BEST):
    rule, ykw, tkw = VARIANTS[variant]
    kit = Kit()
    add_throw_units(kit, **tkw)
    add_yaniv_units(kit, rule, **ykw)
    return kit.build(), kit


if __name__ == "__main__":
    # No arguments: the best design, saved as caller.json. With names: those variants, each under its own name.
    names = sys.argv[1:] or [BEST]
    for name in names:
        net, kit = build(name)
        out = Path(SCRATCH) / "nets" / f"{'caller' if not sys.argv[1:] else name}.json"
        save_net(net, out)
        print(f"saved {out} dims={net.dims}")

"""Thread `survey`: designs on contest_fable's pieces.  build(**kw) -> (net, kit).

TRIPLE (kw TRIP, TRIP_MIN, TRIP_OPP): the open card has the rank of a PAIR I hold and the opponent still has 4-5
cards: keep the pair, take the open card (three of a kind leave together next turn: one card fewer than throwing
the pair now and a single later).  hm5 throws the pair and draws the deck.  One layer-1 unit per rank worth
TRIP_MIN points or more:
    trip_r = relu(3 open_r + 2 take_open + 4 hand_r - 10 throw_r + [opp has 4 or 5] - 7)
           = 1 if I hold two r, take the open r, do not throw r, and the opponent has 4-5 cards (2 with three r).
Measured (survey.md): pays for ranks worth 7+ against 4 or 5 cards and for 5s/6s against 5 cards; not against 1-3 cards.
"""

from __future__ import annotations

from training.yaniv import contest_fable as cf
from training.yaniv.handmade_kit import RANK_NAMES as RANKS, RANK_POINTS, Kit

TRIP = 40.0          # larger than the points the pair-throw line is ahead by on hm5's own count (~3p + 5)
TRIP_MIN = 7         # ranks worth this much or more
TRIP_OPP = (4, 5)    # opponent card counts where it applies
TRIP_MIN5 = 5        # ranks worth this much or more when the opponent has 5 cards (5s and 6s: only then)


def add_triple_units(kit: Kit, trip=TRIP, trip_min=TRIP_MIN, trip_opp=TRIP_OPP, trip_min5=None) -> None:
    trip_min5 = trip_min if trip_min5 is None else trip_min5
    out = {}
    gate = {f"opp_has_{c}": 1.0 for c in trip_opp}
    for r, p in zip(RANKS, RANK_POINTS):
        if p < min(trip_min, trip_min5):
            continue
        gate = {f"opp_has_{c}": 1.0 for c in (trip_opp if p >= trip_min else (5,))}
        kit.unit(1, f"trip_{r}", {f"open_{r}": 3.0, "take_open": 2.0, f"hand_{r}": 4.0, f"throw_{r}": -10.0, **gate},
                 bias=-7.0, note=f"takes an open {r} onto a kept pair of {r}s, opponent has {trip_opp} cards")
        out[f"trip_{r}"] = trip
    kit.out(out)


def build(TRIP=TRIP, TRIP_MIN=TRIP_MIN, TRIP_OPP=TRIP_OPP, TRIP_MIN5=TRIP_MIN5, **over):
    cfg = cf.config(**over)
    kit = Kit()
    cf.add_throw_units(kit, cfg)
    if TRIP:
        add_triple_units(kit, TRIP, TRIP_MIN, tuple(TRIP_OPP), TRIP_MIN5)
    reasons = cf.add_yaniv_units(kit)
    reasons += cf.add_score_units(kit)
    cf.finish_yaniv(kit, reasons)
    cf.fold_output(kit)
    net = kit.build()
    hidden = sum(net.dims[1:-1])
    if hidden > cf.MAX_HIDDEN:
        raise ValueError(f"{hidden} hidden units")
    return net, kit


if __name__ == "__main__":
    net, kit = build()
    base, _ = cf.build()
    print(f"layers {net.dims}: {sum(net.dims[1:-1])} hidden units ({sum(net.dims[1:-1]) - sum(base.dims[1:-1])} more than contest_fable.build())")

"""port: the 4-card hand-written net (training/yaniv/handmade.py) carried to the 5-card phases of the real game,
with every constant that was read off 4-card play looked at again and, where a real-game measurement or a reason
says so, made to depend on the opponent's card count (inputs opp_has_1..5, one on).

    python -m training.yaniv.handmade5_designs.port [OUT.json]      # default OUT: the scratch dir below
    python -m training.yaniv.handmade5_designs.port -v              # also prints every unit with its purpose

Nothing is trained, fitted or searched. The structure is handmade.py's, unchanged: held points, a one-step plan
weighted by P(the opponent calls before my next turn), a charge per heavy card kept, a catch bonus, the small
terms, the three Yaniv reasons and the match-score units (those last two are imported from handmade.py as they
are). What this file adds is that four tables are now written per opponent card count:

  PI[c][bucket]     P(the counter calls on its next turn | it holds c cards, its estimated total's bucket)
  CARD_BY[c]        the charge per heavy card kept
  CATCH_AT[c]       how low my hand must be to catch its call: a distribution over thresholds (the counter's
                    total when it calls rises with its card count: 2.7 / 4.5 / 5.5 / 6.1 / 6.3 on average)
  CALLABLE[c]       bonus for making Yaniv surely legal next turn
  OPEN_TELL_BY[c], LOW_TOP_BY[c]  the two open-card habits

With `FOUR_CARD = True` every table holds the 4-card net's numbers and the net plays exactly as handmade.py's
(checked move for move); the values below are the real-game ones. Where each number comes from is in the
comment beside it; the measurements and every variant tried are in results/handmade-5x7-logs/port.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

from training.yaniv import handmade as H
from training.yaniv.handmade_kit import RANK_NAMES as RANKS
from training.yaniv.handmade_kit import RANK_POINTS, Kit
from training.yaniv.network import save_net

SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/port")
COUNTS = (1, 2, 3, 4, 5)

FOUR_CARD = False   # True: every table below is replaced by the 4-card net's constants (the reproduction check)

# ---------------------------------------------------------------- constants kept from handmade.py (see there)
TWO_LIGHT = H.TWO_LIGHT      # a 2 counts as half a heavy card
PAIR_OPEN = H.PAIR_OPEN      # taking an open card onto a kept card of its rank
DECK_HEAVY = H.DECK_HEAVY    # chance a deck card is heavy ~ 1.07 - share of unseen worth <= 3
FEED = H.FEED                # leaving face up a rank the opponent is known to hold
N_TYP = H.N_TYP              # typical unseen count: P(deck card worth v) ~ unseen worth v / N_TYP
EST_CUTS = H.EST_CUTS        # buckets of the opponent's estimated total: < 5, 5-8, 8-11, 11-14, 14-17, 17+
W_FLOOR = 0.05               # w = P(opponent calls before my next turn) + 0.05, as in handmade.py
CATCH = H.CATCH              # an Assaf against the opponent: +30 to it

# ---------------------------------------------------------------- tables by the opponent's card count
# P(the counter calls on its next turn), by its card count and estimated-total bucket. MEASURED in the real game
# (3,000 rounds of the 4-card net against the counter, port.md table 1); cells with no data are filled
# monotonically. handmade.py's single row was 0.55 / 0.30 / 0.18 / 0.08 / 0.04 / 0.01: the real game agrees
# with it within about 0.05 wherever a bucket is common, because opp_est already carries the card count.
PI = {
    1: [0.63, 0.62, 0.50, 0.30, 0.15, 0.05],   # 163 positions, all under 8
    2: [0.43, 0.33, 0.27, 0.10, 0.04, 0.01],   # 5-8: 1098 positions
    3: [0.30, 0.24, 0.17, 0.11, 0.05, 0.01],
    4: [0.25, 0.18, 0.13, 0.09, 0.04, 0.015],
    5: [0.20, 0.15, 0.10, 0.07, 0.03, 0.007],  # a 5-card hand needs <= 7 points in 5 cards: 0.7% over all
}
# Charge per heavy card kept. handmade.py: 5, and 3 when the opponent has 1-2 cards. See port.md for the 5-card value.
CARD_BY = {1: 3.0, 2: 3.0, 3: 5.0, 4: 5.0, 5: 5.0}
# P(the counter's total when it calls == m), by its card count: MEASURED (port.md table 2), rounded to 0.05.
# My hand catches the call when it is <= the caller's total, so P(catch) = sum_m CATCH_AT[c][m] [my hand <= m].
# handmade.py used one threshold, 5, for every count.
CATCH_AT = {
    1: {1: 0.25, 2: 0.25, 3: 0.20, 4: 0.20, 5: 0.10},
    2: {2: 0.10, 3: 0.15, 4: 0.20, 5: 0.20, 6: 0.20, 7: 0.15},
    3: {3: 0.10, 4: 0.10, 5: 0.20, 6: 0.30, 7: 0.30},
    4: {4: 0.10, 5: 0.15, 6: 0.30, 7: 0.45},
    5: {5: 0.15, 6: 0.30, 7: 0.55},
}
# Extra share of a heavy card that a 2 counts for, on top of TWO_LIGHT (so 0.5 makes a 2 a full heavy card).
# handmade.py: 0 everywhere (a 2 is half a heavy card). See port.md for the 5-card value.
TWO_EXTRA = {5: 0.0}
# The same for an ace or a joker (handmade.py: 0, they are never charged).
ACE_EXTRA = {5: 0.0}
# Charge per unseen copy of the rank of a single card thrown (a "live" rank can still be paired from the deck or
# from the opponent's throws), by the opponent's card count. handmade.py has no such term (0).
LIVE = {}
# Bonus for making Yaniv surely legal next turn. handmade.py: 3 against 4 cards only.
CALLABLE = {4: 3.0}
# Charge per point above 3 of an unpaired open card taken (handmade.py: 1.5) and for leaving a card worth <= 4
# face up (handmade.py: 1.0), by the opponent's card count.
OPEN_TELL_BY = {c: H.OPEN_TELL for c in COUNTS}
LOW_TOP_BY = {c: H.LOW_TOP for c in COUNTS}

if FOUR_CARD:
    PI = {c: list(H.EST_PI) for c in COUNTS}
    CARD_BY = {1: H.CARD - H.CARD_SOON, 2: H.CARD - H.CARD_SOON, 3: H.CARD, 4: H.CARD, 5: H.CARD}
    CATCH_AT = {c: {H.CATCH_M: 1.0} for c in COUNTS}
    CALLABLE = {4: H.CALLABLE_VS4}


# ---------------------------------------------------------------- helpers
def _groups(table: dict) -> list[tuple[tuple[int, ...], object]]:
    """Card counts that share a table row, so one set of units serves them: [((counts...), row), ...]."""
    out: list[tuple[list[int], object]] = []
    for c in COUNTS:
        row = table.get(c)
        if row is None:
            continue
        for counts, r in out:
            if r == row:
                counts.append(c)
                break
        else:
            out.append(([c], row))
    return [(tuple(counts), row) for counts, row in out]


def _has(counts: tuple[int, ...], weight: float) -> dict:
    """weight x [the opponent's card count is one of `counts`] (the opp_has_* inputs are one-hot)."""
    return {f"opp_has_{c}": weight for c in counts}


def _tag(counts: tuple[int, ...]) -> str:
    return "".join(str(c) for c in counts)


# ---------------------------------------------------------------- throws
def add_throw_units(kit: Kit) -> None:
    # ---- held points and the heavy-card charge: handmade.py's units, the charge by opponent card count ----
    kit.unit(1, "light_thrown", {"n_thrown": 4.0, "throw_A": 4.0, "throw_JOKER": 4.0}, bias=-4.0,
             note="cards thrown if the thrown rank is A or joker (light cards leaving), else 0")
    kit.unit(1, "twos_thrown", {"n_thrown": 4.0, "throw_2": 4.0}, bias=-4.0, note="cards thrown if they are 2s")
    kit.unit(1, "took_light_open", {"take_open": 1.0, "open_A": 1.0, "open_JOKER": 1.0}, bias=-1.0,
             note="1 if the move takes an open A or joker")
    kit.unit(1, "took_two_open", {"take_open": 1.0, "open_2": 1.0}, bias=-1.0, note="1 if the move takes an open 2")
    kit.unit(1, "deck_heavy", {"draw_deck": 2.0 + DECK_HEAVY, "unseen_low": -1.0}, bias=-2.0,
             note="deck draw: chance the drawn card is heavy")
    heavy = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
             "hand_2": -4.0 * TWO_LIGHT, "twos_thrown": TWO_LIGHT,
             "take_open": 1.0, "took_light_open": -1.0, "took_two_open": -TWO_LIGHT,
             "deck_heavy": 1.0}                               # heavy cards after the move (0 .. 5)
    out = {"pts_kept": -40.0, "incoming": -10.0}
    top = max(CARD_BY.values())
    for s, w in heavy.items():
        out[s] = out.get(s, 0.0) - top * w                    # the largest charge, linear ...
    for counts, card in _groups(CARD_BY):
        if card != top:                                       # ... and the difference given back per count
            kit.unit(2, f"heavy_vs{_tag(counts)}", {**heavy, **_has(counts, 6.0)}, bias=-6.0,
                     note=f"heavy cards after the move, only when the opponent has {counts} cards")
            out[f"heavy_vs{_tag(counts)}"] = top - card
    twos = {"hand_2": 4.0, "twos_thrown": -1.0, "took_two_open": 1.0}    # 2s I hold after the move
    for counts, extra in _groups(TWO_EXTRA):
        if extra:
            kit.unit(2, f"twos_vs{_tag(counts)}", {**twos, **_has(counts, 6.0)}, bias=-6.0,
                     note=f"2s held after the move, only when the opponent has {counts} cards")
            out[f"twos_vs{_tag(counts)}"] = -extra * max(CARD_BY[c] for c in counts)
    aces = {"hand_A": 4.0, "hand_JOKER": 4.0, "light_thrown": -1.0, "took_light_open": 1.0}   # aces and jokers held
    for counts, extra in _groups(ACE_EXTRA):
        if extra:
            kit.unit(2, f"aces_vs{_tag(counts)}", {**aces, **_has(counts, 6.0)}, bias=-6.0,
                     note=f"aces and jokers held after the move, only when the opponent has {counts} cards")
            out[f"aces_vs{_tag(counts)}"] = -extra * max(CARD_BY[c] for c in counts)

    # ---- open-card habits, callable-next, top card ----
    kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
             note="points above 3 of an open card taken that pairs nothing")   # (name read by the score units)
    out["pairs_open"] = PAIR_OPEN
    base = max(OPEN_TELL_BY.values())
    out["open_tell"] = -base
    for counts, tell in _groups(OPEN_TELL_BY):
        if tell != base:
            kit.unit(1, f"open_tell_vs{_tag(counts)}",
                     {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0, **_has(counts, 10.0)}, bias=-23.0,
                     note=f"open_tell, only when the opponent has {counts} cards")
            out[f"open_tell_vs{_tag(counts)}"] = base - tell
    for counts, low in _groups(LOW_TOP_BY):
        if low:
            kit.unit(1, f"low_top_vs{_tag(counts)}", {"low_top": 1.0, **_has(counts, 1.0)}, bias=-1.0,
                     note=f"leaves a card worth <= 4 face up while the opponent has {counts} cards")
            out[f"low_top_vs{_tag(counts)}"] = -low
    for counts, bonus in _groups(CALLABLE):
        kit.unit(1, f"callable_vs{_tag(counts)}", {"callable_next": 1.0, **_has(counts, 1.0)}, bias=-1.0,
                 note=f"Yaniv surely legal next turn while the opponent has {counts} cards")
        out[f"callable_vs{_tag(counts)}"] = bonus

    # ---- live ranks: a tie-breaker between singles of equal points ----
    for counts, live in _groups(LIVE):
        for r in RANKS:
            kit.unit(1, f"live_{r}_vs{_tag(counts)}",
                     {f"unseen_{r}": 1.0, f"throw_{r}": 1.0, "is_single": 1.0, **_has(counts, 1.0)}, bias=-3.0,
                     note=f"unseen copies / 4 of the single {r} thrown, when the opponent has {counts} cards")
            out[f"live_{r}_vs{_tag(counts)}"] = -4.0 * live

    # ---- feeding penalty ----
    for r, p in zip(RANKS, RANK_POINTS):
        kit.unit(1, f"feed_{r}", {f"throw_{r}": 1.0, f"known_{r}": 4.0}, bias=-1.0,
                 note=f"leaves a {r} face up while the opponent is known to hold one (it takes a set)")
        out[f"feed_{r}"] = -(FEED[0] + FEED[1] * p)
    kit.out(out)

    add_plan_units(kit)
    add_catch_units(kit)


def add_plan_units(kit: Kit) -> None:
    """+ (1 - w) x G, G = points of my best next throw (+ E[relu(drawn card - G)] for a deck draw),
    w = PI[opponent's cards][bucket of its estimated total] + W_FLOOR."""
    groups = []
    for r, p in zip(RANKS, RANK_POINTS):
        if p == 0:
            continue
        kit.unit(1, f"grp_keep_{r}", {f"hand_{r}": 4.0 * p, f"throw_{r}": -40.0},
                 note=f"points of my {r}s if none are thrown")
        kit.unit(1, f"grp_left_{r}", {f"hand_{r}": 4.0 * p, "n_thrown": -4.0 * p, f"throw_{r}": 40.0}, bias=-40.0,
                 note=f"points of my {r}s left if {r}s are thrown")
        kit.unit(1, f"grp_open_{r}", {f"open_{r}": p, "take_open": p}, bias=-p,
                 note=f"{p} if the move takes an open {r}")
        groups.append({f"grp_keep_{r}": 1.0, f"grp_left_{r}": 1.0, f"grp_open_{r}": 1.0})
    G, L = H._max_all(kit, 2, "gmax", groups)
    gt = dict(G)
    for t in range(1, 11):
        kit.unit(L + 1, f"g_under{t}_a", {s: -w for s, w in G.items()}, bias=float(t), note=f"relu({t} - G)")
        kit.unit(L + 1, f"g_under{t}_b", {s: -w for s, w in G.items()}, bias=float(t - 1), note=f"relu({t - 1} - G)")
        pge = {f"unseen_{r}": 4.0 / N_TYP for r, p in zip(RANKS, RANK_POINTS) if p >= t}
        kit.unit(L + 2, f"deck_over{t}", {**pge, f"g_under{t}_a": 1.0, f"g_under{t}_b": -1.0, "draw_deck": 1.0},
                 bias=-2.0, note=f"deck draw and G < {t}: chance the drawn card is worth {t}+")
        gt[f"deck_over{t}"] = 1.0
    # (1 - w) x gt: start from the smallest w of the table, then take back gt x (extra w) per count and bucket.
    w_min = min(row[-1] for row in PI.values()) + W_FLOOR
    out = {s: 1.0 - w_min for s in gt}
    for counts, row in _groups(PI):
        tag = _tag(counts)
        if row[-1] + W_FLOOR != w_min:
            kit.unit(L + 3, f"plan_vs{tag}", {**gt, **_has(counts, 50.0)}, bias=-50.0,
                     note=f"G (+ deck gain) when the opponent has {counts} cards")
            out[f"plan_vs{tag}"] = -(row[-1] + W_FLOOR - w_min)
        for i, t in enumerate(EST_CUTS):
            if row[i] == row[i + 1]:
                continue
            e = H._est_under(kit, t)
            kit.unit(L + 3, f"plan_vs{tag}_est{t}",
                     {**gt, **{s: 50.0 * w for s, w in e.items()}, **_has(counts, 50.0)}, bias=-100.0,
                     note=f"G (+ deck gain) when the opponent has {counts} cards and its estimate is under {t}")
            out[f"plan_vs{tag}_est{t}"] = -(row[i] - row[i + 1])
    kit.out(out)


def add_catch_units(kit: Kit) -> None:
    """CATCH x PI[c][bucket] x P(my hand after the draw <= the caller's total), the caller's total drawn from
    CATCH_AT[c]. For threshold m: deck draw P = sum_k [kept <= k] P(card worth exactly m - k); open card exact."""
    thresholds = sorted({m for row in CATCH_AT.values() for m in row})
    p_low = {}
    for m in thresholds:
        terms = {}
        for k in range(m + 1):
            le = H._step(kit, f"kept_le{k}", {"pts_kept": -40.0}, float(k), f"kept points <= {k}")
            pv = {f"unseen_{r}": 4.0 / N_TYP for r, p in zip(RANKS, RANK_POINTS) if p == m - k}
            kit.unit(2, f"low{m}_k{k}", {**pv, **le, "draw_deck": 1.0}, bias=-2.0,
                     note=f"deck draw, kept <= {k}: chance the card is worth exactly {m - k}")
            terms[f"low{m}_k{k}"] = 1.0
        held = H._step(kit, f"held_le{m}", {"pts_kept": -40.0, "incoming": -10.0}, float(m),
                       f"kept points + incoming card <= {m}")
        kit.unit(2, f"low{m}_open", {**held, "take_open": 1.0}, bias=-1.0, note=f"open card taken, holding <= {m}")
        terms[f"low{m}_open"] = 1.0
        p_low[m] = terms                                   # P(my hand after the draw <= m), in 0..1
    out = {}
    for c in COUNTS:
        mix: dict = {}
        for m, share in CATCH_AT[c].items():               # P(my hand <= the caller's total)
            for s, w in p_low[m].items():
                mix[s] = mix.get(s, 0.0) + share * w
        row = PI[c] + [0.0]
        for i, t in enumerate(EST_CUTS + [None]):
            if row[i] == row[i + 1]:
                continue
            e = H._est_under(kit, t) if t is not None else {}
            name = f"catch_vs{c}_est{t}" if t is not None else f"catch_vs{c}"
            kit.unit(3, name, {**mix, **e, f"opp_has_{c}": 1.0}, bias=-2.0 if t is not None else -1.0,
                     note=f"P(my hand after the draw catches a call) when the opponent has {c} cards"
                          + (f" and its estimate is under {t}" if t is not None else ""))
            out[name] = CATCH * (row[i] - row[i + 1])
    kit.out(out)


# ---------------------------------------------------------------- assembly
def build():
    kit = Kit()
    add_throw_units(kit)
    reasons = H.add_yaniv_units(kit)        # the three Yaniv reasons, unchanged (port.md: measured, no change needed)
    reasons += H.add_score_units(kit)       # match-score units, unchanged
    H.finish_yaniv(kit, reasons)
    H.fold_output(kit)
    return kit.build(), kit


if __name__ == "__main__":
    net, kit = build()
    args = [a for a in sys.argv[1:] if a != "-v"]
    if "-v" in sys.argv:
        print(kit.describe())
    out = Path(args[0]) if args else SCRATCH / "port.json"
    save_net(net, out, meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                             "source": "training/yaniv/handmade5_designs/port.py"})
    print(f"{out}: layers {net.dims}, {sum(net.dims[1:-1])} hidden units, written by hand")

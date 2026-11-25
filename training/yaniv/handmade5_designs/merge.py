"""merge: the round-2 merge of the four 5-card designs (port, counts, watcher, shaper).

    python -m training.yaniv.handmade5_designs.merge [OUT.json] [NAME=value ...]     # -v prints every unit

A hand-written net for the real game (5 cards each, Yaniv at 7). It is meant to play while either player still
holds 5 cards (the 4-card net, training/yaniv/handmade.py, takes over once both hands hold 4 or fewer), but it
scores every legal move in every position. Nothing is trained, fitted or searched.

THE NET is the 4-card net, unit for unit (held points, the one-step plan, the heavy-card charge, the catch
bonus, the open-card habits, the feeding penalty, the Yaniv reasons, the match-score units), plus two rules
that hold while the OPPONENT still has 5 cards. A 5-card hand almost never calls (0.6% a turn), so until the
opponent sheds a card the points I hold cost nearly nothing; what counts is how fast my own hand shrinks.

  RULE 1 (twos_whole, 1 unit): a 2 I keep is charged as a whole heavy card, not half of one. A pair of 2s
     then leaves before a high single, instead of being kept to the end as two "low" cards (a 5-card hand
     holding 2-2 cannot reach 7 points anyway). All four designers found this one.
  RULE 2 (live_*, 12 units): throwing a single card costs LIVE x its points / 10 for every copy of its rank
     that is still unseen: a high single I keep is paired from the open pile one turn in eight against the
     counter (it throws its high cards first; port's measurement), and a pair leaves in one throw. It decides
     which single goes (the rank that can no longer be paired) and tips a low set ahead of a live high single.

Paired playouts against the 4-card net (where the two nets play a different move, both moves are played to
the end of the round on the same hidden cards): rule 1 +0.15 +- 0.04 points a round, rule 2 another
+0.14 +- 0.03 (+0.07 +- 0.025 on a second base), the two together +0.26 +- 0.05 and +0.43 +- 0.05 on two
samples of 20,000 rounds; nearly all of it while both players hold 5 cards.

WHAT THE MERGE LEFT OUT, each measured on top of the two rules (results/handmade-5x7-logs/merge.md has every
variant; the constants are kept in CFG at their "off" value so a variant can be rebuilt by hand):
  - watcher's heavy-card charge of 8 against 5 cards (and the measurer's 7.5 / 5.5): +0.07 on rule 1 alone,
    +0.00 once rule 2 is in. The same decisions (a low set before a high single), bought once already.
  - shaper's +2 for any set at 5 v 5: -0.02 +- 0.01 on top.
  - counts' ALPHA (heavy points outside my best group weigh 0.15 against 5 cards): at 5 v 5 it throws the
    pair of 2s by another route (0.0 +- 0.05 on top of rule 1), and with rule 1 it loses 0.2 a round when I
    hold 3-4 cards against 5 (2s become dearer to keep than kings). Against 4 / 3 cards: +0.00 +- 0.03.
  - counts' form of rule 2 (1 point a copy whatever the card, sets too, also against 4 and 3 cards):
    +0.01 +- 0.02 where port's form gives +0.07; against 4 cards -0.03, port's form there +0.02 +- 0.01.
  - gating rule 1 on my own card count (5 v 5 only: shaper; I hold 5 and the opponent 4-5: watcher):
    5 v 4 adds +0.00 +- 0.03; dropping the rule when I hold 4 or fewer loses 0.015 +- 0.014. "The opponent
    holds 5" is the plainest gate that is level with the others.
  - anything for the cells where I hold 4 or fewer against 5: no variant, nor the trained net's or the
    counter's moves, gains there (all within +-0.03).

ONE FIX to the 4-card net's Yaniv threshold (cap_high_call): it changes no move, see there.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from training.yaniv import handmade as H
from training.yaniv.handmade_kit import RANK_NAMES as RANKS
from training.yaniv.handmade_kit import RANK_POINTS, Kit
from training.yaniv.network import save_net

SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/a532e2e0-83d9-59dd-8a1a-414723f6a44f/scratchpad/merge")

CFG = dict(
    # ---- RULE 1: a 2 counts as a whole heavy card (the 4-card net: half) while the opponent holds one of
    # these card counts. (5,): +0.15 +- 0.04 a round over the 4-card net. (4, 5) with TWO_ME5: nothing at 5 v 4.
    TWO_OPP=(5,),
    TWO_ME5=False,         # True: only while I hold 5 cards myself (shaper, watcher). -0.015 +- 0.014: not used
    # ---- RULE 2: charge for throwing a card, per unseen copy of its rank, by the opponent's card count.
    # 0.5 x points / 10 is port's number: a kept 10 is paired by the next open card 12.7% of the time against
    # 5 cards (6-9 points: 7.4%, 3-5: 3.7%), a pair leaving together is worth about a card (~8 points), so
    # 0.13 x 8 = 1 point for a 10 with three copies unseen, 0.4 a copy; 0.5 written. Not re-tuned here.
    LIVE={5: 0.5},
    LIVE_BY_POINTS=True,   # x the card's points / 10. False (counts: flat): +0.01 +- 0.02, not used
    LIVE_SINGLES=True,     # single throws only. False (counts: sets too): -0.01 +- 0.01, not used
    # ---- measured and left out (see the module note); each is built only when it differs from this value
    CARD_BIG=5.0,          # heavy-card charge in RULE 1's phase (watcher: 8). +0.00 on top of rule 2
    ALPHA={},              # counts: weight of the heavy points held outside my best group, by the opponent's
                           # card count ({5: 0.15, 4: 0.35, 3: 0.6, 2: 0.85, 1: 0.85}). 0 to -0.2 on top
    TWOS55=0.0,            # shaper: bonus for throwing a set of 2s while both hold 5 cards (6) = rule 1
    SET55=0.0,             # shaper: bonus for throwing any set while both hold 5 cards (2). -0.02 on top
    # ---- the fix to the 4-card net's Yaniv threshold, see cap_high_call
    CAP_HIGH_CALL=True,
)


def _phase(c: dict) -> tuple[dict, float]:
    """Gate weights and bias: 0 inside RULE 1's phase, -6 or less outside (my_cards is cards / 5)."""
    gate = {f"opp_has_{n}": 6.0 for n in c["TWO_OPP"]}
    if c["TWO_ME5"]:
        return {**gate, "my_cards": 30.0}, -36.0
    return gate, -6.0


def add_throw_units(kit: Kit, c: dict) -> None:
    """handmade.add_throw_units, unit for unit, plus the rules above."""
    kit.unit(1, "light_thrown", {"n_thrown": 4.0, "throw_A": 4.0, "throw_JOKER": 4.0}, bias=-4.0,
             note="cards thrown if the thrown rank is A or joker (light cards leaving), else 0")
    kit.unit(1, "twos_thrown", {"n_thrown": 4.0, "throw_2": 4.0}, bias=-4.0, note="cards thrown if they are 2s")
    kit.unit(1, "took_light_open", {"take_open": 1.0, "open_A": 1.0, "open_JOKER": 1.0}, bias=-1.0,
             note="1 if the move takes an open A or joker")
    kit.unit(1, "took_two_open", {"take_open": 1.0, "open_2": 1.0}, bias=-1.0, note="1 if the move takes an open 2")
    kit.unit(1, "deck_heavy", {"draw_deck": 2.0 + H.DECK_HEAVY, "unseen_low": -1.0}, bias=-2.0,
             note="deck draw: chance the drawn card is heavy")
    heavy = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
             "hand_2": -4.0 * H.TWO_LIGHT, "twos_thrown": H.TWO_LIGHT,
             "take_open": 1.0, "took_light_open": -1.0, "took_two_open": -H.TWO_LIGHT,
             "deck_heavy": 1.0}                                 # heavy cards after the move, a 2 counted half
    kit.unit(2, "heavy_soon", {**heavy, "opp_has_1": 6.0, "opp_has_2": 6.0}, bias=-6.0,
             note="heavy cards after the move, only when the opponent has 1-2 cards")
    out = {"pts_kept": -40.0, "incoming": -10.0, "heavy_soon": H.CARD_SOON}
    for s, w in heavy.items():
        out[s] = out.get(s, 0.0) - H.CARD * w

    # ---- RULE 1: the other half of the charge for each 2 held after the move, inside the phase
    gate, off = _phase(c)
    if c["TWO_OPP"]:
        kit.unit(2, "twos_whole", {"hand_2": 4.0, "twos_thrown": -1.0, "took_two_open": 1.0, **gate}, bias=off,
                 note=f"2s held after the move, while the opponent holds {c['TWO_OPP']} cards"
                      + (" and I hold 5" if c["TWO_ME5"] else ""))
        out["twos_whole"] = -H.CARD * (1.0 - H.TWO_LIGHT)
        if c["CARD_BIG"] != H.CARD:
            full = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
                    "take_open": 1.0, "took_light_open": -1.0, "deck_heavy": 1.0, "pairs_open": -1.0}
            kit.unit(2, "heavy_big", {**full, **gate}, bias=off,
                     note="heavy cards after the move (2s whole, a paired open card not counted), same phase")
            out["heavy_big"] = -(c["CARD_BIG"] - H.CARD)

    kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
             note="points above 3 of an open card taken that pairs nothing")
    kit.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, bias=-1.0,
             note="Yaniv surely legal next turn while the opponent still has 4 cards")
    out.update({"pairs_open": H.PAIR_OPEN, "open_tell": -H.OPEN_TELL, "callable_vs4": H.CALLABLE_VS4,
                "low_top": -H.LOW_TOP})
    for r, p in zip(RANKS, RANK_POINTS):
        kit.unit(1, f"feed_{r}", {f"throw_{r}": 1.0, f"known_{r}": 4.0}, bias=-1.0,
                 note=f"leaves a {r} face up while the opponent is known to hold one (it takes a set)")
        out[f"feed_{r}"] = -(H.FEED[0] + H.FEED[1] * p)

    # ---- RULE 2: unseen copies of the rank thrown
    by_value: dict = {}
    for n, live in c["LIVE"].items():
        if live:
            by_value.setdefault(live, []).append(n)
    for live, counts in by_value.items():
        tag = "".join(str(n) for n in counts)
        for r, p in zip(RANKS, RANK_POINTS):
            weight = live * (p / 10.0 if c["LIVE_BY_POINTS"] else 1.0)
            if weight < 0.1:
                continue
            w = {f"unseen_{r}": 1.0, f"throw_{r}": 1.0, **{f"opp_has_{n}": 1.0 for n in counts}}
            if c["LIVE_SINGLES"]:
                w["is_single"] = 1.0
            kit.unit(1, f"live_{r}_vs{tag}", w, bias=float(len(counts) - len(w)),
                     note=f"unseen copies / 4 of the {r} thrown" + (" alone" if c["LIVE_SINGLES"] else "")
                          + f", while the opponent holds {tuple(counts)} cards")
            out[f"live_{r}_vs{tag}"] = -4.0 * weight

    # ---- shaper's bonuses
    if c["TWOS55"]:
        kit.unit(1, "twos_55", {"is_set": 1.0, "throw_2": 1.0, "opp_has_5": 1.0, "my_cards": 5.0}, bias=-7.0,
                 note="1 if a set of 2s is thrown while both players hold 5 cards")
        out["twos_55"] = c["TWOS55"]
    if c["SET55"]:
        kit.unit(1, "set_55", {"is_set": 1.0, "opp_has_5": 1.0, "my_cards": 5.0}, bias=-6.0,
                 note="1 if a set is thrown while both players hold 5 cards")
        out["set_55"] = c["SET55"]

    # ---- ALPHA, first half: give back (1 - alpha) x the heavy points held (the plan takes the same share of G)
    if any(alpha != 1.0 for alpha in c["ALPHA"].values()):
        kit.unit(1, "aces_thrown", {"n_thrown": 4.0, "throw_A": 4.0}, bias=-4.0, note="cards thrown if they are aces")
        kit.unit(1, "took_ace_open", {"take_open": 1.0, "open_A": 1.0}, bias=-1.0, note="1 if the move takes an open A")
    pts = {"pts_kept": 40.0, "incoming": 10.0, "hand_A": -4.0, "aces_thrown": 1.0, "took_ace_open": -1.0,
           "hand_2": -8.0, "twos_thrown": 2.0, "took_two_open": -2.0}
    for n, alpha in c["ALPHA"].items():
        if alpha != 1.0:
            kit.unit(2, f"heavy_pts_vs{n}", {**pts, f"opp_has_{n}": 60.0}, bias=-60.0,
                     note=f"points held after the move outside aces and 2s, while the opponent holds {n} cards")
            out[f"heavy_pts_vs{n}"] = 1.0 - alpha
    kit.out(out)

    add_plan_units(kit, c)
    H.add_catch_units(kit)


def add_plan_units(kit: Kit, c: dict) -> None:
    """handmade.add_plan_units: + (1 - w(opp_est)) x G, G = points of my best next throw (+ the deck gain);
    with ALPHA[n] the coefficient is (ALPHA[n] - w) while the opponent holds n cards."""
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
        pge = {f"unseen_{r}": 4.0 / H.N_TYP for r, p in zip(RANKS, RANK_POINTS) if p >= t}
        kit.unit(L + 2, f"deck_over{t}", {**pge, f"g_under{t}_a": 1.0, f"g_under{t}_b": -1.0, "draw_deck": 1.0},
                 bias=-2.0, note=f"deck draw and G < {t}: chance the drawn card is worth {t}+")
        gt[f"deck_over{t}"] = 1.0
    out = {s: 1.0 - H.EST_W[-1] for s in gt}
    for i, t in enumerate(H.EST_CUTS):
        e = H._est_under(kit, t)
        kit.unit(L + 3, f"plan_est{t}", {**gt, **{s: 50.0 * w for s, w in e.items()}}, bias=-50.0,
                 note=f"G (+ deck gain) when the opponent's estimate is under {t}")
        out[f"plan_est{t}"] = -(H.EST_W[i] - H.EST_W[i + 1])
    for n, alpha in c["ALPHA"].items():
        if alpha != 1.0:
            kit.unit(L + 3, f"plan_vs{n}", {**gt, f"opp_has_{n}": 50.0}, bias=-50.0,
                     note=f"G (+ deck gain) while the opponent holds {n} cards")
            out[f"plan_vs{n}"] = -(1.0 - alpha)
    kit.out(out)


def cap_high_call(kit: Kit) -> None:
    """handmade.py's Yaniv threshold has a term HIGH_CALL x (my points above 5), meant for calls at 6-7 points
    but not capped. With 5 cards a hand reaches 50 points, the term alone passes the unit's -100 gate on THROW
    rows above about 35 points, and every throw then gets the Yaniv reason's +2000. It is the same for all
    throws of a position and Yaniv is not legal there, so no move changes, but scores are no longer "within
    -100..+60" and near-ties are decided at float32's resolution around 2000. Cap the term at 7 points (the
    Yaniv limit): on a Yaniv row nothing changes, on a throw row the unit stays at 0."""
    kit.unit(1, "high_call_over", {"my_pts": 40.0}, bias=-7.0, note="my points above 7 (caps high_call)")
    for name in ("call", "call_sat"):
        kit.layers[kit.level[name] - 1][name][0]["high_call_over"] = -500.0 * H.HIGH_CALL


def build(**over):
    unknown = set(over) - set(CFG)
    if unknown:
        raise KeyError(f"unknown constants: {sorted(unknown)}")
    c = {**CFG, **over}
    kit = Kit()
    add_throw_units(kit, c)
    reasons = H.add_yaniv_units(kit)
    if c["CAP_HIGH_CALL"]:
        cap_high_call(kit)
    reasons += H.add_score_units(kit)
    H.finish_yaniv(kit, reasons)
    H.fold_output(kit)
    return kit.build(), kit


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if "=" not in a and a != "-v"]
    over = {k: ast.literal_eval(v) for k, v in (a.split("=", 1) for a in sys.argv[1:] if "=" in a)}
    net, kit = build(**over)
    if "-v" in sys.argv:
        print(kit.describe())
    out_path = Path(args[0]) if args else SCRATCH / "merge.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_net(net, out_path, meta={"rules": {"hand": 5, "yaniv_at": 7}, "handmade": True,
                                  "source": "training/yaniv/handmade5_designs/merge.py",
                                  "constants": {k: str(v) for k, v in {**CFG, **over}.items()}})
    print(f"{out_path}: layers {net.dims}, {sum(net.dims[1:-1])} hidden units, written by hand")

"""merge_small: a small, readable hand-written Yaniv net (4 cards, Yaniv at 7) that merges round-1 ideas.

Every weight is written by hand from reasoning; nothing is trained, fitted or searched. Base: reader.py.

HOW A MOVE IS VALUED (output in points; higher is better)

Throws. A throw is worth minus the hand it leaves:
  - the points I hold after the move (kept points + the card coming in: the open card's points, or the
    unseen average for a deck draw) - that is what I pay if the opponent calls before my next turn;
  - plus CARD points per "heavy" card left (anything but A, 2, joker; a 2 counts half). The charge is the
    future: a hand of fewer cards stays smaller for the rest of the round, so a pair of 3s or more is
    thrown before a higher single, while a pair of aces or 2s is kept. When the opponent is near a call
    the future matters less, so the charge falls (by the opponent's card count, as in reader);
  - a bonus for taking an open card that pairs a kept card (a set to throw next turn) and more for a triple;
  - a charge per point above 3 of an unpaired open card taken (a deck draw is usually as good and stays
    hidden), so plain open cards are taken only at 3 points or less;
  - +3 for a move that makes Yaniv surely legal next turn while the opponent still holds 4 cards;
  - -1 for leaving a card worth 4 or less face up (the card counter takes those).
  Optional merged pieces (see the constants): a catch bonus, a feeding penalty, an information cost.

Yaniv. The Yaniv row is -YBIG or +YBIG and nothing else can move it across the throws (they all score
within about -100..+40): +YBIG exactly when one of the "call" reasons is on, through a saturating unit:
  - the call cannot be caught (sure_yaniv);
  - the uniform catch chance `assaf` is under a threshold T that falls with the ply (late in a round the
    opponent's hidden cards are low, so `assaf` understates the risk more) and rises with the gap between
    the opponent's estimated total and mine: T = 0.015 + relu(0.025 - 0.00125 ply) + 0.002 (gap - 5)
    (gap capped at 15) + 0.005 relu(pts - 5);
  - (score units) a caught call would land my score exactly on 50 or 100, which is cut, so call.

Score units (add_score_units, copied from scorer.py): when taking the open card (or drawing) can leave my
score + held points exactly on 50 or 100, a call by the opponent would cut my score by 50: bonus
~50 x P(opponent calls next turn).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("YANIV_HAND_SIZE", "4")
os.environ.setdefault("YANIV_CALL_AT", "7")

from training.yaniv.handmade_kit import RANK_NAMES, RANK_POINTS, Kit  # noqa: E402
from training.yaniv.network import save_net  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-yaniv-master/ecf1221d-8c18-559f-9f3b-91b1597261e5/scratchpad"

# ---------------------------------------------------------------- throw constants
CARD = 5.0          # points charged per heavy card left: the trained net throws a pair of 3s+ over a higher single
CARD_SOON = 2.0     # charge taken back when the opponent has 1-2 cards (it may call soon: points matter more)
TWO_LIGHT = 0.5     # a 2 counts as half a heavy card (the trained net throws 22 over a 3, keeps it over a 5+)
PAIR_OPEN = 9.0     # taking an open card that pairs a kept card: a set to throw next turn (trained takes ~90%)
TRIPLE_OPEN = 5.0   # extra when it makes a triple (one more card goes out with the set)
OPEN_TELL = 1.5     # per point above 3 of an unpaired open card taken (trained takes unpaired 4s 35%, 5s 7%)
CALLABLE_VS4 = 3.0  # Yaniv surely legal next turn while the opponent holds 4 cards: a call then is near-safe
LOW_TOP = 1.0       # leaving a card worth <= 4 face up feeds the counter (it takes those 95% of the time)
DECK_HEAVY = 1.07   # chance a deck card is heavy ~ 1.07 - share of unseen worth <= 3 (the 3s count as heavy)

# ---------------------------------------------------------------- Yaniv constants
YBIG = 1000.0       # the Yaniv row is -YBIG or +YBIG; throws score within about -100..+40
T_EARLY = 0.025     # threshold on assaf: T_FLOOR + relu(T_EARLY - T_SLOPE * ply) + gap and 6-7 point terms
T_SLOPE = 0.00125   # ... 0.04 at ply 0, the floor from ply 20 (opponents' hidden cards are lower late)
T_FLOOR = 0.015
GAP_SLOPE = 0.002   # threshold change per point of (opponent estimate - my points) ...
GAP_CENTER = 5.0    # ... around a gap of 5 ...
GAP_CAP = 15.0      # ... with the gap capped here
HIGH_CALL = 0.005   # per point above 5: the trained net calls a little more readily at 6-7 points
C1_SLOPE = 0.4      # caller.py's margin: call only if opp_est - pts >= C1_SLOPE * pts + C1_BASE (2.9 at 4, 4.1 at 7)
C1_BASE = 1.3

RANKS = RANK_NAMES
DEFAULTS: dict = dict(
    pair_pts=0.0,     # extra bonus per point of an open card taken that pairs a kept card
    catch=None,       # catch bonus: list of (opp_est at or under, points) cumulative steps; m = CATCH_M
    catch_m=5,        # "low enough to catch the opponent's call": my hand after the draw <= this
    feed=None,        # feeding penalty (flat, per point) for leaving face up a rank the opponent is known to hold
    info=None,        # (opp_est at or under, cost per point) of an open card taken: the opponent sees it
    soon="cards",     # how the card charge relief is triggered: "cards" (opp 1-2 cards) or [(est<=t, relief), ...]
    rule="reader",    # Yaniv rule: "reader", "clone", "caller", "reader_gap" (reader AND gap >= rule_gap)
    rule_gap=2.0,
    score=False,      # add the match-score units
)
N_TYP = 38.0        # typical unseen count mid-round, to turn unseen counts into a deck-draw chance


def _step(kit: Kit, name: str, weights: dict, bias: float, note: str) -> dict:
    """Two layer-1 units whose difference is 1 if (weights . x + bias) >= 0, ramping to 0 within -1."""
    kit.unit(1, name + "_a", weights, bias + 1.0, note=note + " (ramp)")
    kit.unit(1, name + "_b", weights, bias, note=note + " (ramp shifted)")
    return {name + "_a": 1.0, name + "_b": -1.0}


def _est_le(kit: Kit, t: float) -> dict:
    """[opp_est <= t points] (width 0.1 point)."""
    name = f"est_le{t:g}"
    if name + "_a" in kit.level:
        return {name + "_a": 1.0, name + "_b": -1.0}
    return _step(kit, name, {"opp_est": -400.0}, 10.0 * t, f"opponent's estimated total <= {t:g}")


# ---------------------------------------------------------------- throws
def add_throw_units(kit: Kit, o: dict) -> None:
    # ---- layer 1 ----
    kit.unit(1, "light_thrown", {"n_thrown": 4.0, "throw_A": 4.0, "throw_JOKER": 4.0}, bias=-4.0,
             note="cards thrown if the thrown rank is A or joker (light cards leaving), else 0")
    kit.unit(1, "twos_thrown", {"n_thrown": 4.0, "throw_2": 4.0}, bias=-4.0, note="cards thrown if they are 2s")
    kit.unit(1, "took_light_open", {"take_open": 1.0, "open_A": 1.0, "open_JOKER": 1.0}, bias=-1.0,
             note="1 if the move takes an open A or joker")
    kit.unit(1, "took_two_open", {"take_open": 1.0, "open_2": 1.0}, bias=-1.0, note="1 if the move takes an open 2")
    for r in RANKS:
        kit.unit(1, f"triple_{r}", {f"open_{r}": 1.0, f"hand_{r}": 4.0, "take_open": 1.0, f"throw_{r}": -4.0},
                 bias=-3.5, note=f"0.5 if the move takes an open {r} onto two kept {r}s (a triple next turn)")
    kit.unit(1, "deck_heavy", {"draw_deck": 2.0 + DECK_HEAVY, "unseen_low": -1.0}, bias=-2.0,
             note="deck draw: chance the drawn card is heavy")
    kit.unit(1, "open_tell", {"open_pts": 10.0, "take_open": 10.0, "pairs_open": -10.0}, bias=-13.0,
             note="points above 3 of an open card taken that pairs nothing")
    kit.unit(1, "callable_vs4", {"callable_next": 1.0, "opp_has_4": 1.0}, bias=-1.0,
             note="Yaniv surely legal next turn while the opponent still has 4 cards")

    # heavy cards after the move, linear in inputs and layer-1 units
    heavy = {"n_kept": 5.0, "hand_A": -4.0, "hand_JOKER": -4.0, "light_thrown": 1.0,
             "hand_2": -4.0 * TWO_LIGHT, "twos_thrown": TWO_LIGHT,
             "take_open": 1.0, "took_light_open": -1.0, "took_two_open": -TWO_LIGHT, "deck_heavy": 1.0}
    out = {"pts_kept": -40.0, "incoming": -10.0, "pairs_open": PAIR_OPEN, "open_tell": -OPEN_TELL,
           "callable_vs4": CALLABLE_VS4, "low_top": -LOW_TOP}
    for r in RANKS:
        out[f"triple_{r}"] = 2.0 * TRIPLE_OPEN
    for s, w in heavy.items():
        out[s] = out.get(s, 0.0) - CARD * w

    # ---- layer 2: card-charge relief when the opponent may call soon ----
    if o["soon"] == "cards":
        kit.unit(2, "heavy_soon", {**heavy, "opp_has_1": 6.0, "opp_has_2": 6.0}, bias=-6.0,
                 note="heavy cards after the move, only when the opponent has 1-2 cards")
        out["heavy_soon"] = CARD_SOON
    else:
        for t, relief in o["soon"]:
            dz = _est_le(kit, t)
            kit.unit(2, f"heavy_est{t:g}", {**heavy, **{s: 6.0 * w for s, w in dz.items()}}, bias=-6.0,
                     note=f"heavy cards after the move, only when the opponent's estimate is <= {t:g}")
            out[f"heavy_est{t:g}"] = relief
    kit.out(out)

    if o["pair_pts"]:
        # a pairing open card goes out with its twin next turn, so its points are held for one turn only
        kit.unit(1, "paired_open_pts", {"incoming": 10.0, "pairs_open": 10.0}, bias=-10.0,
                 note="points of an open card taken to pair a kept card (0 otherwise)")
        kit.out({"paired_open_pts": o["pair_pts"]})

    if o["feed"]:
        flat, per = o["feed"]
        for r, p in zip(RANKS, RANK_POINTS):
            kit.unit(1, f"feed_{r}", {f"throw_{r}": 1.0, f"known_{r}": 4.0}, bias=-1.0,
                     note=f"leaves a {r} face up while the opponent is known to hold one (it takes a set)")
            kit.out({f"feed_{r}": -(flat + per * p)})

    if o["info"]:
        t, per = o["info"]
        dz = _est_le(kit, t)
        kit.unit(2, "shown_pts", {"incoming": 10.0, "take_open": 10.0, **{s: 10.0 * w for s, w in dz.items()}},
                 bias=-20.0, note=f"points of an open card taken, when the opponent's estimate is <= {t:g}")
        kit.out({"shown_pts": -per})

    if o["catch"]:
        # Catch bonus: if the opponent calls next turn and my hand is at or under its total, the call is an
        # Assaf (+30 to it). Value ~ 30 x P(it calls next) x P(my hand after the draw <= m).
        # P for a deck draw = sum_k [kept == k] P(card worth <= m - k) = sum_k [kept <= k] P(card worth m - k),
        # with P(card worth v) ~ unseen cards worth v / N_TYP; for an open card it is exact.
        m = o["catch_m"]
        K = {"pts_kept": 40.0}
        le = {k: _step(kit, f"kept_le{k}", {"pts_kept": -40.0}, float(k), f"kept points <= {k}") for k in range(m + 1)}
        held_le = _step(kit, f"held_le{m}", {"pts_kept": -40.0, "incoming": -10.0}, float(m),
                        f"kept points + incoming card <= {m}")
        del K
        for t, w in o["catch"]:
            dz = _est_le(kit, t)
            for k in range(m + 1):
                v = m - k
                pv = {f"unseen_{r}": 4.0 / N_TYP for r, p in zip(RANKS, RANK_POINTS) if p == v}
                kit.unit(2, f"catch{t:g}_k{k}", {**pv, **le[k], **dz, "draw_deck": 1.0}, bias=-3.0,
                         note=f"deck draw, kept <= {k}, opp_est <= {t:g}: chance the card is worth exactly {v}")
                kit.out({f"catch{t:g}_k{k}": w})
            kit.unit(2, f"catch{t:g}_open", {**held_le, **dz, "take_open": 1.0}, bias=-2.0,
                     note=f"open card taken, holding <= {m}, opp_est <= {t:g}")
            kit.out({f"catch{t:g}_open": w})


# ---------------------------------------------------------------- Yaniv
def add_yaniv_units(kit: Kit, o: dict) -> list[dict]:
    """Return the 'call' reasons, each a linear combination of units that is 0 or 1 (only on the Yaniv row)."""
    reasons = []
    kit.unit(1, "sure_call", {"sure_yaniv": 1.0, "yaniv": 1.0}, bias=-1.0, note="the call cannot be caught")
    reasons.append({"sure_call": 1.0})
    rule = o["rule"]
    if rule in ("reader", "reader_gap", "reader_c1"):
        kit.unit(1, "early", {"ply": -T_SLOPE * 48}, bias=T_EARLY,
                 note="threshold part that falls with the ply: T_EARLY at ply 0, 0 from ply T_EARLY/T_SLOPE")
        kit.unit(1, "gap", {"opp_est": 40.0, "my_pts": -40.0}, note="opponent's estimated total minus mine, >= 0")
        kit.unit(1, "gap_cap", {"opp_est": 40.0, "my_pts": -40.0}, bias=-GAP_CAP, note="the gap above GAP_CAP")
        kit.unit(1, "high_call", {"my_pts": 40.0}, bias=-5.0, note="my points above 5")
        t_terms = {"early": 1.0, "gap": GAP_SLOPE, "gap_cap": -GAP_SLOPE, "high_call": HIGH_CALL}
        margin = {s: 500.0 * w for s, w in t_terms.items()}
        margin.update({"assaf": -500.0, "yaniv": 100.0})
        b = -100.0 + 500.0 * (T_FLOOR - GAP_SLOPE * GAP_CENTER)
        kit.unit(2, "call", margin, bias=b, note="Yaniv row: 500 (T - assaf)")
        kit.unit(2, "call_sat", dict(margin), bias=b - 1.0, note="call - 1: call - call_sat = [assaf < T]")
        if rule == "reader":
            reasons.append({"call": 1.0, "call_sat": -1.0})
        elif rule == "reader_c1":
            # AND caller's margin: opp_est - pts >= C1_SLOPE pts + C1_BASE (a caught call costs pts + 30, so a
            # higher hand needs a wider margin over the opponent's likely total)
            g = _step(kit, "c1gap", {"opp_est": 400.0, "my_pts": -400.0 * (1.0 + C1_SLOPE)}, -10.0 * C1_BASE,
                      "opponent's estimate - my points >= C1_SLOPE * my points + C1_BASE")
            kit.unit(3, "call_c1", {"call": 1.0, "call_sat": -1.0, **g}, bias=-1.0,
                     note="assaf under T AND the opponent's estimate clears my points by caller's margin")
            reasons.append({"call_c1": 1.0})
        else:
            g = _step(kit, "gapmin", {"opp_est": 40.0, "my_pts": -40.0}, -o["rule_gap"],
                      "opponent's estimate at least rule_gap above my points")
            kit.unit(3, "call_gap", {"call": 1.0, "call_sat": -1.0, **g}, bias=-1.0,
                     note="assaf under T AND the opponent looks rule_gap+ points higher")
            reasons.append({"call_gap": 1.0})
    elif rule == "clone":
        # the counter's thr(pts), call if assaf <= 0.5 thr, or (assaf <= 1.75 thr and opp_est - pts >= 3)
        for c in range(1, 6):
            kit.unit(1, f"pts_over{c}", {"my_pts": 40.0}, bias=-float(c), note=f"relu(my points - {c})")
        thr = {"bias": 0.40, 1: -0.12, 2: 0.02, 3: 0.04, 4: 0.02, 5: 0.02}
        Kc = 2000.0

        def assaf_le(name, scale):
            w = {f"pts_over{c}": Kc * scale * thr[c] for c in range(1, 6)}
            w.update({"assaf": -Kc, "yaniv": YBIG * 10})
            kit.unit(2, name + "_a", w, bias=Kc * scale * thr["bias"] + 1.0 - YBIG * 10, note="ramp: scaled thr - assaf")
            kit.unit(2, name + "_b", dict(w), bias=Kc * scale * thr["bias"] - YBIG * 10, note="ramp shifted")
            return {name + "_a": 1.0, name + "_b": -1.0}

        strict = assaf_le("cstrict", 0.5)
        loose = assaf_le("cloose", 1.75)
        g = _step(kit, "gap3", {"opp_est": 40.0, "my_pts": -40.0}, -3.0, "opponent's estimate 3+ above my points")
        kit.unit(3, "c_loose_gap", {**loose, **g}, bias=-1.0, note="loose threshold AND gap >= 3")
        kit.unit(3, "c_strict", strict, note="strict threshold")
        reasons += [{"c_loose_gap": 1.0}, {"c_strict": 1.0}]
    elif rule == "caller":
        # margin (points) = opp_est - pts - (0.4 pts + 1.3) - 10 assaf > 0
        w = {"opp_est": 40.0 * 10, "my_pts": -40.0 * 1.4 * 10, "assaf": -100.0, "yaniv": YBIG * 10}
        kit.unit(1, "c1_a", w, bias=-13.0 + 1.0 - YBIG * 10, note="C1 margin ramp on the Yaniv row")
        kit.unit(1, "c1_b", dict(w), bias=-13.0 - YBIG * 10, note="ramp shifted")
        reasons.append({"c1_a": 1.0, "c1_b": -1.0})
    else:
        raise ValueError(rule)
    return reasons


def finish_yaniv(kit: Kit, reasons: list[dict]) -> None:
    """Yaniv row = -YBIG + 2 YBIG * min(1, sum of reasons): no reason can push it past +YBIG, and throws
    (the reasons are all 0 on throw rows) never get near either end."""
    total: dict = {}
    for r in reasons:
        for s, w in r.items():
            total[s] = total.get(s, 0.0) + w
    layer = max(kit.level[s] for s in total) + 1
    kit.unit(layer, "ycall_a", total, note="sum of call reasons (0 on throw rows)")
    kit.unit(layer, "ycall_b", dict(total), bias=-1.0, note="ycall_a - ycall_b = min(1, reasons)")
    kit.out({"yaniv": -YBIG, "ycall_a": 2 * YBIG, "ycall_b": -2 * YBIG})


# ---------------------------------------------------------------- score units
GATE = 200.0        # gate strength for the score units: larger than any value a gated sum takes
OPEN_CUT = {50: (1.5, 18.0), 100: (1.0, 9.0)}  # (flat, x threat) points for an open take landing on 50 / 100:
                    # ~50 x P(opponent calls next turn) at 50, half at 100 (scorer.py's measurement)
DECK_CUT = (20.0, 10.0)  # the same for a deck draw, x P(the drawn card lands exactly on the cut)


def _threat(kit: Kit) -> dict:
    """threat in 0..1: chance-like weight that the opponent calls next turn (1 at opp_est <= 9, 0 at >= 15)."""
    if "threat_a" not in kit.level:
        kit.unit(1, "threat_a", {"opp_est": -40.0}, bias=15.0, note="relu(15 - opponent's estimated total)")
        kit.unit(1, "threat_b", {"opp_est": -40.0}, bias=9.0, note="(threat_a - threat_b)/6: 1 at <= 9, 0 at >= 15")
    return {"threat_a": 1.0 / 6, "threat_b": -1.0 / 6}


def add_score_units(kit: Kit, o: dict) -> list[dict]:
    """Match-score units (copied from scorer.add_score_units, same weights). Returns extra Yaniv reasons.

    1. open card: my score + held points after taking it == 50 or 100 (exact: s = 120 my_score + 40 pts_kept
       + 10 incoming, a tent of width 1). If the opponent calls next turn my score is cut by 50:
       bonus (flat + x threat) from OPEN_CUT.
    2. deck draw: lands on the cut if the drawn card is worth exactly target - score - kept;
       bonus DECK_CUT x P(that value) x threat, the product as relu(P + 0.45 threat - 0.45) (P <= ~0.45).
    3. assaf on cut: my score + my points + 30 == 50 or 100: even a caught call lowers my score (it is
       cut), and an uncaught one costs nothing, so call. This is a Yaniv REASON (0/1) fed into the same
       saturating unit as the other reasons, so it always wins (+YBIG) and can never push past it.
    Throw-side bonuses are at most ~20 points, far inside +-YBIG, so they cannot flip a Yaniv decision.
    """
    threat = _threat(kit)
    s = {"my_score": 120.0, "pts_kept": 40.0, "incoming": 10.0, "take_open": GATE}
    for at, (flat, x_threat) in OPEN_CUT.items():
        pieces = (("lo", -(at - 1), 1.0), ("mid", -at, -2.0), ("hi", -(at + 1), 1.0))
        for name, off, wt in pieces:
            kit.unit(1, f"opencut{at}_{name}", s, bias=off - GATE,
                     note=f"tent piece: my score + held points after taking the open card == {at}")
            kit.out({f"opencut{at}_{name}": flat * wt})
        w = {f"opencut{at}_{name}": GATE * wt for name, _, wt in pieces}
        w.update(threat)
        kit.unit(2, f"opencut{at}_threat", w, bias=-GATE, note=f"threat when the open-card take lands on {at}")
        kit.out({f"opencut{at}_threat": x_threat})
    s = {"my_score": 120.0, "pts_kept": 40.0, "draw_deck": GATE}
    for at, weight in zip((50, 100), DECK_CUT):
        chance = {}
        for v in range(0, 11):
            pieces = (("lo", -(at - v - 1), 1.0), ("mid", -(at - v), -2.0), ("hi", -(at - v + 1), 1.0))
            for name, off, wt in pieces:
                kit.unit(1, f"deckcut{at}_{v}_{name}", s, bias=off - GATE,
                         note=f"tent piece: deck draw and my score + kept == {at} - {v}")
            w = {f"deckcut{at}_{v}_{name}": GATE * wt for name, _, wt in pieces}
            for r, p in zip(RANK_NAMES, RANK_POINTS):
                if p == v:
                    w[f"unseen_{r}"] = 4.0 / 34.0
            kit.unit(2, f"deckcut{at}_{v}", w, bias=-GATE, note=f"chance the deck card is worth {v} (lands on {at})")
            chance[f"deckcut{at}_{v}"] = 1.0
        kit.unit(3, f"deckcut{at}", {**chance, **{k: 0.45 * x for k, x in threat.items()}}, bias=-0.45,
                 note=f"~ chance the deck draw lands on {at} x threat")
        kit.out({f"deckcut{at}": weight})
    reason = {}
    a = {"my_score": 120.0, "my_pts": 40.0, "yaniv": GATE}
    for at in (50, 100):
        for name, off, wt in (("lo", -(at - 31), 1.0), ("mid", -(at - 30), -2.0), ("hi", -(at - 29), 1.0)):
            kit.unit(1, f"assafcut{at}_{name}", a, bias=off - GATE,
                     note=f"tent piece: Yaniv row and my score + my points + 30 == {at}")
            reason[f"assafcut{at}_{name}"] = wt
    return [reason]


def build(**overrides):
    o = dict(DEFAULTS)
    o.update(overrides)
    kit = Kit()
    add_throw_units(kit, o)
    reasons = add_yaniv_units(kit, o)
    if o["score"]:
        reasons += add_score_units(kit, o)
    finish_yaniv(kit, reasons)
    return kit.build(), kit


if __name__ == "__main__":
    net, kit = build()
    print(kit.describe())
    print(net.dims, file=sys.stderr)
    name = sys.argv[1] if len(sys.argv) > 1 else "merge_small"
    save_net(net, Path(SCRATCH) / "nets" / f"{name}.json")

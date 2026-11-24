"""A hand-written net with match-score awareness ("scorer").

Every weight below is written by hand from reasoning; nothing is trained or searched.

The base round policy is a readable copy of the card counter's idea:
- throws (`add_throw_units`): minimise the points held after the draw, minus the
  points the NEXT throw will be able to shed (the highest card, or a pair), plus
  the counter's penalty for leaving a low card face up;
- the Yaniv call (`add_yaniv_units`): call when the call cannot be caught, or when
  the uniform Assaf chance is at most a threshold that falls with my points
  (the counter's own table: 0.40 at 0-1 points ... 0.04 at 7).

Everything that depends on the match score is in `add_score_units`, so it can be
plugged into another design. It needs only the kit and the input names; the Yaniv
override terms assume the design's Yaniv row is decided by output differences of
order 100 (see the note there).

Output units are points (badness of the hand, negated): higher is better.

    python -m training.yaniv.handmade_designs.scorer      # writes SCRATCH/nets/scorer.json and scorer_base.json
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("YANIV_HAND_SIZE", "4")
os.environ.setdefault("YANIV_CALL_AT", "7")

from training.yaniv.handmade_kit import RANK_NAMES, RANK_POINTS, Kit  # noqa: E402

SCRATCH = "/tmp/claude-0/-home-user-yaniv-master/ecf1221d-8c18-559f-9f3b-91b1597261e5/scratchpad"

BIG = 200.0  # gate strength: larger than any value a gated sum can take


# ---------------------------------------------------------------- base: throws
def add_throw_units(kit: Kit, shed_weight: float = 0.78) -> None:
    """Score a throw by -(points held after the draw) + shed_weight * (points the next throw can shed) - top penalty.

    The counter's badness is w*points + (1-w)*keep with keep = points - best shed, i.e.
    points - (1-w)*shed; w = 0.22 when the opponent holds 4 cards, more when fewer
    (it may call sooner, so the current total matters more).
    """
    # Layer 1: cards of rank r thrown, and the open card when it is taken.
    for r in RANK_NAMES:
        kit.unit(1, f"thrown_{r}", {"n_thrown": 1.0, f"throw_{r}": 1.0}, bias=-1.0,
                 note=f"cards of rank {r} thrown / 4 (n_thrown when this rank is thrown, else 0)")
        kit.unit(1, f"takes_{r}", {"take_open": 1.0, f"open_{r}": 1.0}, bias=-1.0,
                 note=f"1 if the move takes the open card and it is a {r}")
    # Layer 2: "after the draw I hold a card worth >= v" for v = 1..10, and pairs.
    # count of rank r after = 4*(hand_r - thrown_r) + takes_r  (hand_r is carried by the kit)
    def after(r: str, scale: float) -> dict:
        return {f"hand_{r}": 4.0 * scale, f"thrown_{r}": -4.0 * scale, f"takes_{r}": scale}

    for v in range(1, 11):
        w: dict = {}
        for r, p in zip(RANK_NAMES, RANK_POINTS):
            if p >= v:
                for s, x in after(r, 1.0).items():
                    w[s] = w.get(s, 0.0) + x
        kit.unit(2, f"ge{v}_a", w, note=f"cards worth {v}+ held after the draw (drawn deck card not counted)")
        kit.unit(2, f"ge{v}_b", w, bias=-1.0, note=f"same minus 1, so ge{v}_a - ge{v}_b = [any card worth {v}+]")
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        if p == 0:
            continue
        kit.unit(2, f"pair_{r}_a", after(r, 1.0), bias=-1.0, note=f"{r}s held after the draw minus 1")
        kit.unit(2, f"pair_{r}_b", after(r, 1.0), bias=-2.0, note="minus 2: pair_a - pair_b = [two or more]")
    # Layer 3: highest single card H = sum_v [any >= v]; a pair of rank r sheds 2p, so extra = relu(2p*pair - H).
    high = {}
    for v in range(1, 11):
        high[f"ge{v}_a"] = 1.0
        high[f"ge{v}_b"] = -1.0
    for r, p in zip(RANK_NAMES, RANK_POINTS):
        if p == 0:
            continue
        w = {k: -x for k, x in high.items()}
        w[f"pair_{r}_a"] = 2.0 * p
        w[f"pair_{r}_b"] = -2.0 * p
        kit.unit(3, f"pairx_{r}", w, note=f"how much more a pair of {r}s sheds than the highest single")
    kit.unit(3, "high", high, note="highest card held after the draw (the next throw sheds at least this)")
    shed = {"high": 1.0, **{f"pairx_{r}": 1.0 for r, p in zip(RANK_NAMES, RANK_POINTS) if p}}
    # Layer 4: shed again, but only when the opponent holds 3 or fewer cards (it may call soon: shed counts less).
    kit.unit(4, "shed_opp3", {**shed, "opp_has_3": 40.0}, bias=-40.0, note="shed if the opponent has 3 cards")
    kit.unit(4, "shed_opp12", {**shed, "opp_has_1": 40.0, "opp_has_2": 40.0}, bias=-40.0,
             note="shed if the opponent has 1-2 cards")
    # A deck draw may itself become the highest card: E[relu(c - high)] over unseen cards is about
    # 3.6 / 2.2 / 0.7 / 0 for high = 3 / 5 / 8 / 10; 0.5 * relu(9 - high) follows that, a bit low (late in a round the unseen cards are lower).
    kit.unit(4, "deck_up", {"high": -0.5, "draw_deck": BIG}, bias=0.5 * 9.0 - BIG,
             note="expected extra shed from the deck card being higher than what I keep (deck draws only)")
    # Top penalty (counter's _top_penalty): t = 10*top_pts; 0.35 if t<=5, +0.65 if t<=3, +0.8 if t<=1.
    for name, edge in (("top5", 6.0), ("top3", 4.0), ("top1", 2.0)):
        gate = {"top_pts": -10.0, "draw_deck": BIG, "take_open": BIG}  # every throw draws; the Yaniv row is gated off
        kit.unit(1, f"{name}_a", gate, bias=edge - BIG, note=f"relu({edge:g} - face-up points) on throw rows")
        kit.unit(1, f"{name}_b", gate, bias=edge - 1.0 - BIG,
                 note=f"{name}_a - {name}_b = 1 if the face-up card is worth {edge - 1:g} or less")
    kit.out({
        "pts_kept": -40.0, "incoming": -10.0,              # points held after the draw
        **{k: shed_weight * x for k, x in shed.items()},     # minus what the next throw sheds
        "deck_up": shed_weight,
        "shed_opp3": -0.18, "shed_opp12": -0.40,            # w = 0.40 / 0.62 when the opponent has fewer cards
        "top5_a": -0.35, "top5_b": 0.35, "top3_a": -0.65, "top3_b": 0.65, "top1_a": -0.8, "top1_b": 0.8,
    })


# ---------------------------------------------------------------- base: Yaniv
def add_yaniv_units(kit: Kit) -> None:
    """Yaniv row: -100, plus 200 when the call is good (thr(points) - assaf >= 0, the counter's table).

    Throw rows score about -45..+10, so -100 never wins and +100 always does.
    thr(p) = 0.40 - 0.12 relu(p-1) + 0.02 relu(p-2) + 0.04 relu(p-3) + 0.02 relu(p-4) + 0.02 relu(p-5)
    gives 0.40, 0.40, 0.28, 0.18, 0.12, 0.08, 0.06, 0.04 for p = 0..7.
    """
    for k in range(1, 6):
        kit.unit(1, f"pts_over{k}", {"my_pts": 40.0}, bias=-float(k), note=f"relu(my points - {k})")
    thr = {"pts_over1": -0.12, "pts_over2": 0.02, "pts_over3": 0.04, "pts_over4": 0.02, "pts_over5": 0.02}
    K = 200.0  # assaf resolution 0.005
    w = {s: K * x for s, x in thr.items()}
    w.update({"assaf": -K, "yaniv": BIG})
    kit.unit(2, "call_a", w, bias=K * 0.40 + 1.0 - BIG, note="K*(thr - assaf) + 1 on the Yaniv row")
    kit.unit(2, "call_b", w, bias=K * 0.40 - BIG, note="call_a - call_b = 1 when assaf <= thr(points)")
    kit.unit(1, "sure_call", {"sure_yaniv": 1.0, "yaniv": 1.0}, bias=-1.0, note="the call cannot be caught")
    kit.out({"yaniv": -100.0, "call_a": 200.0, "call_b": -200.0, "sure_call": 200.0})


# ---------------------------------------------------------------- score units
def add_score_units(kit: Kit, open_cut=None, assaf_cut: bool = True, desperate: float = 0.0,
                    bust=0.0, deck_cut=(20.0, 10.0)) -> None:
    """Match-score adjustments. Units of the output: points (the base's scale).

    1. open_cut: taking the open card leaves exactly 50 or 100 on (my score + held points),
       so if the opponent calls next turn my score is cut by 50. Exact for open-card
       draws: s = 120*my_score + 40*pts_kept + 10*incoming; a tent of width 1 at 50 and 100.
    2. assaf_cut: Yaniv is legal and my score + my points + 30 is exactly 50 or 100: even a
       caught call lowers my score (cut), so call. +400 on the Yaniv row.
    3. desperate: my score >= 80 and I am >= 15 behind: losing more points barely matters,
       a round won matters a lot, so call Yaniv at a higher Assaf chance (thr + `desperate`).
    4. bust: (optional) taking/drawing a hand that puts my score + held points at 100+ means
       being called ends the match; penalty in points.
    """
    if open_cut is None:  # (flat, x threat) points: ~50 * P(opponent calls next) at 50, half that at 100
        open_cut = {50: (1.5, 18.0), 100: (1.0, 9.0)}
    if isinstance(open_cut, (int, float)):
        open_cut = {50: (open_cut, 0.0), 100: (open_cut, 0.0)}
    if any(f or t for f, t in open_cut.values()):
        s = {"my_score": 120.0, "pts_kept": 40.0, "incoming": 10.0, "take_open": BIG}
        if any(t for _, t in open_cut.values()):
            # threat: how likely the opponent calls on its next turn. Measured (diagnostic): it calls next
            # ~40% of the time when its estimated total is <= 9, ~14% at 10-14, ~3% above 14.
            kit.unit(1, "threat_a", {"opp_est": -40.0}, bias=15.0, note="relu(15 - opponent's estimated total)")
            kit.unit(1, "threat_b", {"opp_est": -40.0}, bias=9.0, note="(threat_a - threat_b)/6: 1 at <= 9, 0 at >= 15")
        for at, (flat, threat) in open_cut.items():
            pieces = (("lo", -(at - 1), 1.0), ("mid", -at, -2.0), ("hi", -(at + 1), 1.0))
            for name, off, wt in pieces:
                kit.unit(1, f"opencut{at}_{name}", s, bias=off - BIG,
                         note=f"tent piece: my score + held points after taking the open card == {at}")
                kit.out({f"opencut{at}_{name}": flat * wt})
            if threat:
                w = {f"opencut{at}_{name}": BIG * wt for name, _, wt in pieces}
                w.update({"threat_a": 1.0 / 6, "threat_b": -1.0 / 6})
                kit.unit(2, f"opencut{at}_threat", w, bias=-BIG,
                         note=f"threat (0..1) when the open-card take lands on {at}, else 0")
                kit.out({f"opencut{at}_threat": threat})
    if deck_cut:
        # Deck draw: lands on the cut if the drawn card is worth exactly need = target - score - kept.
        # P(need = v) = unseen cards worth v / unseen count, taken as / 34 (a typical unseen count).
        # Bonus = weight * P * threat, the product approximated by relu(P + 0.45*threat - 0.45)
        # (exact at the corners; P is at most ~0.45, for v = 10).
        if "threat_a" not in kit.level:
            kit.unit(1, "threat_a", {"opp_est": -40.0}, bias=15.0, note="relu(15 - opponent's estimated total)")
            kit.unit(1, "threat_b", {"opp_est": -40.0}, bias=9.0, note="(threat_a - threat_b)/6: 1 at <= 9, 0 at >= 15")
        s = {"my_score": 120.0, "pts_kept": 40.0, "draw_deck": BIG}
        for at, weight in zip((50, 100), deck_cut):
            if not weight:
                continue
            chance = {}
            for v in range(0, 11):
                pieces = (("lo", -(at - v - 1), 1.0), ("mid", -(at - v), -2.0), ("hi", -(at - v + 1), 1.0))
                for name, off, wt in pieces:
                    kit.unit(1, f"deckcut{at}_{v}_{name}", s, bias=off - BIG,
                             note=f"tent piece: deck draw and my score + kept == {at} - {v}")
                w = {f"deckcut{at}_{v}_{name}": BIG * wt for name, _, wt in pieces}
                for r, p in zip(RANK_NAMES, RANK_POINTS):
                    if p == v:
                        w[f"unseen_{r}"] = 4.0 / 34.0
                kit.unit(2, f"deckcut{at}_{v}", w, bias=-BIG, note=f"chance the deck card is worth {v} (when it lands on {at})")
                chance[f"deckcut{at}_{v}"] = 1.0
            kit.unit(3, f"deckcut{at}", {**chance, "threat_a": 0.45 / 6, "threat_b": -0.45 / 6}, bias=-0.45,
                     note=f"~ chance the deck draw lands on {at} x threat")
            kit.out({f"deckcut{at}": weight})
    if bust:
        flat, threat = bust if isinstance(bust, tuple) else (bust, 0.0)
        s = {"my_score": 120.0, "pts_kept": 40.0, "incoming": 10.0, "yaniv": -BIG}
        kit.unit(1, "bust_a", s, bias=-100.0, note="my score + held points after the draw - 100 (throws only)")
        kit.unit(1, "bust_b", s, bias=-101.0, note="bust_a - bust_b = 1 when that passes 100 (exactly 100 is cut)")
        kit.out({"bust_a": -flat, "bust_b": flat})
        if threat:
            if "threat_a" not in kit.level:
                kit.unit(1, "threat_a", {"opp_est": -40.0}, bias=15.0, note="relu(15 - opponent's estimated total)")
                kit.unit(1, "threat_b", {"opp_est": -40.0}, bias=9.0, note="(threat_a - threat_b)/6")
            kit.unit(2, "bust_threat", {"bust_a": BIG, "bust_b": -BIG, "threat_a": 1 / 6, "threat_b": -1 / 6},
                     bias=-BIG, note="threat when the held points take my score to 100+")
            kit.out({"bust_threat": -threat})
    if assaf_cut:
        a = {"my_score": 120.0, "my_pts": 40.0, "yaniv": BIG}
        for at in (50, 100):
            for name, off, wt in (("lo", -(at - 31), 1.0), ("mid", -(at - 30), -2.0), ("hi", -(at - 29), 1.0)):
                kit.unit(1, f"assafcut{at}_{name}", a, bias=off - BIG,
                         note=f"tent piece: Yaniv row and my score + my points + 30 == {at}")
                kit.out({f"assafcut{at}_{name}": 400.0 * wt})
    if desperate:
        kit.unit(1, "high_a", {"my_score": 120.0}, bias=-79.0, note="my score - 79")
        kit.unit(1, "high_b", {"my_score": 120.0}, bias=-80.0, note="high_a - high_b = 1 when my score >= 80")
        kit.unit(1, "behind_a", {"my_score": 120.0, "opp_score": -120.0}, bias=-14.0, note="how far behind - 14")
        kit.unit(1, "behind_b", {"my_score": 120.0, "opp_score": -120.0}, bias=-15.0,
                 note="behind_a - behind_b = 1 when I am 15 or more behind")
        kit.unit(2, "desperate", {"high_a": 1.0, "high_b": -1.0, "behind_a": 1.0, "behind_b": -1.0}, bias=-1.0,
                 note="1 when my score >= 80 and I am >= 15 behind, else 0")
        thr = {"pts_over1": -0.12, "pts_over2": 0.02, "pts_over3": 0.04, "pts_over4": 0.02, "pts_over5": 0.02}
        if "pts_over1" not in kit.level:
            for k in range(1, 6):
                kit.unit(1, f"pts_over{k}", {"my_pts": 40.0}, bias=-float(k), note=f"relu(my points - {k})")
        K = 200.0
        w = {s: K * x for s, x in thr.items()}
        w.update({"assaf": -K, "yaniv": BIG, "desperate": BIG})
        kit.unit(3, "dcall_a", w, bias=K * (0.40 + desperate) + 1.0 - 2 * BIG,
                 note="K*(thr + extra - assaf) + 1 on the Yaniv row when desperate")
        kit.unit(3, "dcall_b", w, bias=K * (0.40 + desperate) - 2 * BIG, note="dcall_a - dcall_b: the risky call")
        kit.out({"dcall_a": 200.0, "dcall_b": -200.0})


def build(score: bool = True, **score_opts):
    kit = Kit()
    add_throw_units(kit)
    add_yaniv_units(kit)
    if score:
        add_score_units(kit, **score_opts)
    return kit.build(), kit


if __name__ == "__main__":
    from training.yaniv.network import save_net

    net, kit = build(score=False)
    save_net(net, Path(SCRATCH) / "nets" / "scorer_base.json")
    net, kit = build()
    save_net(net, Path(SCRATCH) / "nets" / "scorer.json")
    print(kit.describe())
    print(net.dims)

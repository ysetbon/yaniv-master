"""Thread `match`: match-score units on top of contest_fable's pieces. Weights written by hand.

    build()                       # contest_fable's net plus the units switched on below
    build(BUSTCALL=None, ...)     # a piece is off when its constant is None

Every piece reads the match scores (my_score, opp_score, / 120) and is exactly 0 while my score is low, so play
from 0-0 is unchanged (checked by the rounds duel). See results/contest-fable-logs/match.md for the measurements.

Pieces
  BUSTCALL  Yaniv reason: my score + my points > 100 (if the opponent wins this round I am out whatever happens:
            being caught costs no more than losing the round) and assaf < BUSTCALL.
  FATAL     Yaniv veto: my score + my points < 100 but + 30 > 100 (a caught call ends the match, a lost round
            would not) and assaf > FATAL.
  BUST      throw: my score + held points after the move > 100 (kept + open card; kept + unseen average for a deck
            draw): -(flat + x threat [+ per opponent-score point above BUST_T0]).
  DENY      Yaniv veto when the call would hand the opponent its own cut (add_deny_units).
  OPEN, DECK, THI   contest_fable's cut-landing units (add_score_units) with their constants as arguments, per cut
            (50 / 100): OPEN = (flat, x threat) for an open take that lands exactly, DECK = weight on
            ~P(the deck card lands exactly) x threat, THI = the opponent estimate where the threat reaches 0.
"""

from __future__ import annotations

from training.yaniv import contest_fable as cf
from training.yaniv.handmade_kit import RANK_NAMES as RANKS, RANK_POINTS, Kit

G = 300.0
K = 500.0


def _and_assaf(kit: Kit, name: str, gate: dict, thr: float, below: bool, note: str) -> dict:
    """[yaniv row] x [gate] x [assaf < thr] (or > thr): a 0..1 combination, 0 on throw rows."""
    sign = 1.0 if below else -1.0
    w = {"assaf": -sign * K, "yaniv": G, **{s: G * x for s, x in gate.items()}}
    b = sign * K * thr - 2.0 * G
    kit.unit(2, name + "_a", w, bias=b, note=note)
    kit.unit(2, name + "_b", dict(w), bias=b - 1.0, note=note + " (saturation)")
    return {name + "_a": 1.0, name + "_b": -1.0}


def add_match_units(kit: Kit, BUSTCALL, FATAL, BUST, BUST_T) -> list[dict]:
    reasons = []
    now = {"my_score": 240.0, "my_pts": 80.0}
    if BUSTCALL is not None or FATAL is not None:
        over = cf._step(kit, "over_now", now, -201.0, "my score + my points >= 101 (a lost round ends the match)")
    if BUSTCALL is not None:
        reasons.append(_and_assaf(kit, "bustcall", over, BUSTCALL, True,
                                  f"Yaniv row, out if I lose the round, assaf < {BUSTCALL}"))
    if FATAL is not None:
        fat = cf._step(kit, "fatal_now", now, -141.0, "my score + my points + 30 >= 101 (a caught call ends the match)")
        gate = dict(fat)
        for s, x in over.items():
            gate[s] = gate.get(s, 0.0) - x
        veto = _and_assaf(kit, "fatal", gate, FATAL, False, f"Yaniv row, a caught call is fatal, assaf > {FATAL}")
        reasons.append({s: -5.0 * x for s, x in veto.items()})
    if BUST is not None:
        flat, x_threat = BUST
        after = cf._step(kit, "over_after", {"my_score": 240.0, "pts_kept": 80.0, "incoming": 20.0}, -201.0,
                         "my score + held points after the move >= 101")
        kit.out({s: -flat * x for s, x in after.items()})
        threat = cf._threat(kit)
        if x_threat:
            kit.unit(2, "over_threat", {**after, **threat}, bias=-1.0, note="over 100 after the move x threat")
            kit.out({"over_threat": -x_threat})
        if BUST_T is not None:
            t0, per = BUST_T
            # per point of the opponent's score above t0, when over and under threat (staying alive at 95-99 is
            # worth more the closer the opponent is to 100 itself)
            kit.unit(2, "over_opp", {**{s: 100.0 * x for s, x in after.items()}, **{s: 100.0 * x for s, x in threat.items()},
                                     "opp_score": 120.0}, bias=-200.0 - t0,
                     note=f"opponent's score above {t0} when over 100 after the move, under threat")
            kit.out({"over_opp": -per})
    return reasons


def add_deny_units(kit: Kit, DENY) -> list[dict]:
    """Yaniv veto when my call would hand the opponent ITS cut: its score + its hand lands exactly on 50 / 100.
    DENY = (wide, hid): veto when at most `hid` of its cards are hidden from me and
      A. |its score + opp_est - cut| < 0.5 (with no hidden card opp_est is its exact total), or
      B. |its score + opp_est - cut| < wide and at least one card is hidden (wide None: A only).
    Measured against hm5, which sits on its cut on purpose (match.md 2.5)."""
    wide, hid = DENY
    d = {"opp_score": 240.0, "opp_est": 80.0}                     # 2 x (its score + its estimated total)
    hidden = {"opp_cards": 10.0, "known_n": -10.0}                # 2 x hidden cards
    few = cf._step(kit, "hid_le", {k: -v for k, v in hidden.items()}, 2.0 * hid, f"at most {hid} opponent cards hidden")
    some = cf._step(kit, "hid_ge1", hidden, -2.0, "at least one opponent card hidden")
    a_in, b_in = {}, {}
    for cut in (50, 100):
        for name, off, sign, acc in (("lo", 0.5, 1.0, a_in), ("hi", -0.5, -1.0, a_in)) + (
                (("wlo", wide, 1.0, b_in), ("whi", -wide, -1.0, b_in)) if wide else ()):
            st = cf._step(kit, f"deny{cut}_{name}", d, -2.0 * cut + 2.0 * off,
                          f"opponent's score + estimate >= {cut} - {off:g}")
            for u, x in st.items():
                acc[u] = acc.get(u, 0.0) + sign * x
    veto = {}
    kit.unit(2, "deny_exact", {**a_in, **few, "yaniv": 1.0}, bias=-2.0,
             note="Yaniv row: the opponent's total sits on its cut (within 0.5), few hidden cards")
    veto["deny_exact"] = -5.0
    if wide:
        kit.unit(2, "deny_near", {**b_in, **few, **some, "yaniv": 1.0}, bias=-3.0,
                 note=f"Yaniv row: the opponent's total within {wide:g} of its cut, 1..{hid} hidden cards")
        veto["deny_near"] = -5.0
    return [veto]


def _threat_to(kit: Kit, hi: float) -> dict:
    """Threat in 0..1 that the opponent calls next turn: 1 at opp_est <= 9, 0 at >= hi (contest_fable's: hi 15)."""
    if hi == 15:
        return cf._threat(kit)
    cf._threat(kit)
    if f"threat_a{hi:g}" not in kit.level:
        kit.unit(1, f"threat_a{hi:g}", {"opp_est": -40.0}, bias=hi, note=f"relu({hi:g} - opponent's estimated total)")
    return {f"threat_a{hi:g}": 1.0 / (hi - 9.0), "threat_b": -1.0 / (hi - 9.0)}


def add_score_units(kit: Kit, OPEN, DECK, THI) -> list[dict]:
    """contest_fable.add_score_units with its constants as arguments: OPEN[cut] = (flat, x threat),
    DECK[cut] = weight on ~P(the deck card lands on the cut) x threat, THI[cut] = the opponent estimate at which
    the threat reaches 0 (contest_fable: 15 for both). Same units otherwise (one more layer-1 unit per new THI)."""
    GATE = cf.GATE
    s = {"my_score": 120.0, "pts_kept": 40.0, "incoming": 10.0, "take_open": GATE}
    for at, (flat, x_threat) in OPEN.items():
        threat = _threat_to(kit, THI[at])
        pieces = (("lo", -(at - 1), 1.0), ("mid", -at, -2.0), ("hi", -(at + 1), 1.0))
        for name, off, wt in pieces:
            kit.unit(1, f"opencut{at}_{name}", s, bias=off - GATE,
                     note=f"tent piece: my score + held points after taking the open card == {at}")
            kit.out({f"opencut{at}_{name}": flat * wt})
        w = {f"opencut{at}_{name}": GATE * wt for name, _, wt in pieces}
        w.update(threat)
        kit.unit(2, f"opencut{at}_threat", w, bias=-GATE, note=f"threat when the open-card take lands on {at}")
        kit.out({f"opencut{at}_threat": x_threat})
        if cf.TELL_ON_CUT:
            w = {f"opencut{at}_{name}": 10.0 * wt for name, _, wt in pieces}
            w["open_tell"] = 1.0
            kit.unit(2, f"opencut{at}_tell", w, bias=-10.0, note=f"open_tell when the open-card take lands on {at}")
            kit.out({f"opencut{at}_tell": cf.OPEN_TELL})
    s = {"my_score": 120.0, "pts_kept": 40.0, "draw_deck": GATE}
    for at in (50, 100):
        threat = _threat_to(kit, THI[at])
        chance = {}
        for v in range(0, 11):
            pieces = (("lo", -(at - v - 1), 1.0), ("mid", -(at - v), -2.0), ("hi", -(at - v + 1), 1.0))
            for name, off, wt in pieces:
                kit.unit(1, f"deckcut{at}_{v}_{name}", s, bias=off - GATE,
                         note=f"tent piece: deck draw and my score + kept == {at} - {v}")
            w = {f"deckcut{at}_{v}_{name}": GATE * wt for name, _, wt in pieces}
            for r, p in zip(RANKS, RANK_POINTS):
                if p == v:
                    w[f"unseen_{r}"] = 4.0 / 34.0
            kit.unit(2, f"deckcut{at}_{v}", w, bias=-GATE, note=f"chance the deck card is worth {v} (lands on {at})")
            chance[f"deckcut{at}_{v}"] = 1.0
        kit.unit(3, f"deckcut{at}", {**chance, **{k: 0.45 * x for k, x in threat.items()}}, bias=-0.45,
                 note=f"~ chance the deck draw lands on {at} x threat")
        kit.out({f"deckcut{at}": DECK[at]})
    reason = {}
    a = {"my_score": 120.0, "my_pts": 40.0, "yaniv": GATE}
    for at in (50, 100):
        for name, off, wt in (("lo", -(at - 31), 1.0), ("mid", -(at - 30), -2.0), ("hi", -(at - 29), 1.0)):
            kit.unit(1, f"assafcut{at}_{name}", a, bias=off - GATE,
                     note=f"tent piece: Yaniv row and my score + my points + 30 == {at}")
            reason[f"assafcut{at}_{name}"] = wt
    return [reason]


DEFAULTS = {"BUSTCALL": None, "FATAL": None, "BUST": None, "BUST_T": None, "DENY": None,
            "OPEN": {50: cf.OPEN_CUT[50], 100: cf.OPEN_CUT[100]}, "DECK": {50: cf.DECK_CUT[0], 100: cf.DECK_CUT[1]},
            "THI": {50: 15.0, 100: 15.0}}


def build(**kw):
    opt = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    for k in list(kw):
        if k in opt:
            v = kw.pop(k)
            if isinstance(opt[k], dict):          # JSON keys are strings: {"100": ...}
                opt[k].update({int(c): tuple(x) if isinstance(x, list) else x for c, x in v.items()})
            else:
                opt[k] = tuple(v) if isinstance(v, list) else v
    cfg = cf.config(**kw)
    kit = Kit()
    cf.add_throw_units(kit, cfg)
    reasons = cf.add_yaniv_units(kit)
    reasons += add_score_units(kit, opt["OPEN"], opt["DECK"], opt["THI"])
    reasons += add_match_units(kit, opt["BUSTCALL"], opt["FATAL"], opt["BUST"], opt["BUST_T"])
    if opt["DENY"] is not None:
        reasons += add_deny_units(kit, opt["DENY"])
    cf.finish_yaniv(kit, reasons)
    cf.fold_output(kit)
    net = kit.build()
    hidden = sum(net.dims[1:-1])
    if hidden > cf.MAX_HIDDEN:
        raise ValueError(f"{hidden} hidden units: the contest allows {cf.MAX_HIDDEN}")
    return net, kit


if __name__ == "__main__":
    import json
    import sys

    net, kit = build(**(json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}))
    print(net.dims, sum(net.dims[1:-1]), "hidden units")

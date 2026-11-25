"""Thread `match`: probes for contest_fable_lab.paired(mode="matches") and the equity valuation of its records.

Only plays and counts. Alternatives are defined a priori (see results/contest-fable-logs/match.md):

  call      Yaniv is legal: the other choice (call <-> the hero's best throw).
  nobust    my score + held points after the chosen move (kept + open card, or kept + unseen average for a deck
            draw) is over 100 and some throw leaves it under 100: the hero's best such throw.
  minhold   my score >= 60: the throw that holds the fewest points (hero's order breaks ties).
  cutopen   an open-card take that lands my score + held points exactly on 50 / 100, not chosen.
  cutdeck10 a deck draw with my score + kept == cut - 10 (a ten-point card lands on the cut), not chosen.
  cutkeep   my score + hand is on the cut now: the throw that changes my points least (stay near the cut).

    python -m training.yaniv.contest_fable_match_probe hm5 hm5 3000 1 [probe]
"""

from __future__ import annotations

import math
import pickle
import sys

import numpy as np

from . import contest_fable_lab as lab
from .engine import POINTS_OF, mask_points
from .features import net_actions, state_features

HERO = {"name": "hm5"}
_T = {}


def table():
    if "t" not in _T:
        _T["t"] = np.load(lab.LOGS / "wtable.npy")
    return _T["t"]


def eq_gain(r) -> float:
    """Match-win chance of the alternative minus the played move, for one paired record (x100 = points of %)."""
    t = table()
    return 100.0 * (lab.equity(t, r["s"] + r["a1"], r["t"] + r["b1"]) - lab.equity(t, r["s"] + r["a0"], r["t"] + r["b0"]))


def band(x, cuts):
    lo = None
    for c in cuts:
        if x < c:
            return f"<{c}" if lo is None else f"{lo}-{c}"
        lo = c
    return f"{lo}+"


S_CUTS = (40, 63, 70, 80, 86, 93)        # 63: s + 7 + 30 >= 100 from here (a caught call ends the match from 63-70 up)
A_CUTS = (0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75)


def _held(view, a, avg):
    kept = mask_points(view.my_hand & ~a.discard_mask)
    return kept + (POINTS_OF[view.discard[-1]] if a.draw == "open" else avg), kept


def probe(view, actions, chosen, state, which=("call", "nobust", "minhold", "cutopen", "cutdeck10", "cutkeep")):
    hero = lab.player(HERO["name"])
    s, t = view.my_score, view.opp_score
    f = state_features(view)
    avg = float(f[77]) * 10.0
    est = float(f[94]) * 40.0
    assaf = float(f[95])
    pts = mask_points(view.my_hand)
    base = {"sb": band(s, S_CUTS), "tb": band(t, S_CUTS), "oc": view.opp_size, "pts": pts, "est": est,
            "assaf": assaf, "ply": view.ply, "opp_true": mask_points(state.hands[1 - view.seat])}
    plays = [a for a in net_actions(actions)[0] if a.kind == "play"]
    out = []
    yan = [a for a in actions if a.kind == "yaniv"]
    if yan and "call" in which:
        if chosen.kind == "yaniv":
            out.append(("call:skip", hero(view, plays, None), {**base, "ab": band(assaf, A_CUTS)}))
        else:
            out.append(("call:call", yan[0], {**base, "ab": band(assaf, A_CUTS)}))
    if chosen.kind != "play":
        return out
    held = {id(a): _held(view, a, avg) for a in plays}
    ch_held, ch_kept = _held(view, chosen, avg)
    if "nobust" in which and s + ch_held > 100:
        ok = [a for a in plays if s + held[id(a)][0] < 100]
        if ok:
            out.append(("nobust", hero(view, ok, None), {**base, "over": s + ch_held - 100}))
    if "minhold" in which and s >= 60:
        low = min(h for h, _ in held.values())
        ok = [a for a in plays if held[id(a)][0] <= low + 1e-9]
        alt = hero(view, ok, None)
        out.append(("minhold", alt, {**base, "ch_bust": s + ch_held > 100, "alt_bust": s + low > 100,
                                     "saved": ch_held - low}))
    for cut in (50, 100):
        if "cutopen" in which:
            ok = [a for a in plays if a.draw == "open" and s + held[id(a)][0] == cut]
            if ok and not (chosen.draw == "open" and s + ch_held == cut):
                out.append((f"cutopen{cut}", hero(view, ok, None), {**base, "thr": band(est, (9, 12, 15, 20))}))
        if "cutdeck10" in which:
            ok = [a for a in plays if a.draw == "stock" and s + held[id(a)][1] == cut - 10]
            if ok and not (s + ch_held == cut and chosen.draw == "open") and not (
                    chosen.draw == "stock" and s + ch_kept == cut - 10):
                out.append((f"cutdeck10_{cut}", hero(view, ok, None), {**base, "thr": band(est, (9, 12, 15, 20))}))
        if "cutkeep" in which and s + pts == cut:
            # closest to staying on the cut: |held after - pts| smallest
            d = min(abs(held[id(a)][0] - pts) for a in plays)
            ok = [a for a in plays if abs(held[id(a)][0] - pts) <= d + 1e-9]
            out.append((f"cutkeep{cut}", hero(view, ok, None), {**base, "d": round(d), "thr": band(est, (9, 12, 15, 20))}))
    return out


def probe_deny(view, actions, chosen, state):
    """The hero calls Yaniv: the alternative is its best throw. Info on how close the opponent's total sits to a
    cut (its score + what I know of its hand): calling gives it its hand, and exactly 50 / 100 is cut by 50."""
    if chosen.kind != "yaniv":
        return []
    hero = lab.player(HERO["name"])
    f = state_features(view)
    t = view.opp_score
    est, known_pts, known_n, opp_min = float(f[94]) * 40.0, round(float(f[83]) * 40.0), round(float(f[84]) * 5.0), float(f[85]) * 40.0
    need = view.opp_size - known_n
    true = mask_points(state.hands[1 - view.seat])
    d = min((t + est - 50, t + est - 100), key=abs)
    plays = [a for a in net_actions(actions)[0] if a.kind == "play"]
    info = {"oc": view.opp_size, "need": need, "known_pts": known_pts, "est": est, "opp_min": opp_min, "d": d,
            "assaf": float(f[95]), "pts": mask_points(view.my_hand), "opp_true": true, "on": t + true in (50, 100),
            "sure": bool(f[86]), "ply": view.ply}
    return [("deny", hero(view, plays, None), info)]


def probe_call(view, actions, chosen, state):
    return probe(view, actions, chosen, state, which=("call",))


def report(records, stats, matches, key, least=40, gain=eq_gain, title=""):
    """Per group: count, per match, equity gain (points of match-win %) each +- se, per match; also points gain."""
    groups = {}
    for r in records:
        groups.setdefault(key(r), []).append((gain(r), r["gain"]))
    lines = [title] if title else []
    for k in sorted(groups, key=str):
        g = groups[k]
        if len(g) < least:
            continue
        e = np.array([x for x, _ in g])
        p = np.array([x for _, x in g])
        lines.append(f"  {str(k):<46} n {len(g):6d} ({len(g) / matches:6.3f}/match)  equity each {e.mean():+7.3f} "
                     f"+-{e.std() / math.sqrt(len(g)):.3f}  per match {e.sum() / matches:+7.4f}   "
                     f"points each {p.mean():+6.2f} +-{p.std() / math.sqrt(len(g)):.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    hero, opp, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    name = sys.argv[5] if len(sys.argv) > 5 else "probe"
    HERO["name"] = hero
    recs, stats = lab.paired(hero, opp, f"contest_fable_match_probe:{name}", n, seed, mode="matches", workers=2)
    path = lab.LOGS / f"match_rec_{name}_{hero.replace(':', '_').replace('.', '_')[:30]}_{opp}_{n}_{seed}.pkl"
    with open(path, "wb") as fh:
        pickle.dump((recs, stats, n), fh)
    print(path, len(recs), stats)

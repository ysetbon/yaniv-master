"""Team opus, topic `match`: playing for the MATCH (scores, the 50/100 cuts, match point), not the round margin.

Tools (tests, nothing here fits or searches weights):
  - win_table(): P(win match | my score, opp score) between equal players, by value iteration over round outcomes of
    hm5 against itself (match_value.win_table). Used only to turn a round's end into match-win units.
  - match_playout(hero, opp, alt, matches, seed): plays whole matches; at each hero decision where the rule `alt`
    proposes another move B, the round is played to its end both ways on the same hidden cards and the record keeps
    dwin = P(win | scores after B) - P(win | scores after A) (table units) and the round-margin difference.
  - alt_* : the probes / rules tested (see results/contest-opus-logs/match.md).
  - add_match_units(kit): the block kept (if any) for contest_opus.build(extras=[...]).

    python -m training.yaniv.contest_opus_match table
    python -m training.yaniv.contest_opus_match probe hm5 alt_flip 400 7100
"""

from __future__ import annotations

import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRATCH = Path("/tmp/claude-0/-home-user-yaniv-master/3c7d4d81-ef83-553b-adc8-6a5382f1abf0/scratchpad/match")
TABLE = SCRATCH / "win_table_hm5.npy"


# ---------------------------------------------------------------- the match-win table
def _outcomes_job(job):
    n, seed = job
    from .contest_opus_playout import player
    from .engine import play_round
    hero = player("hm5")
    rng = random.Random(seed)
    out = []
    for i in range(n):
        st = play_round(hero, hero, rng, [0, 0], i % 2)
        out.append((st.result.added[0], st.result.added[1]))
    return out


def build_table(rounds: int = 20000, seed: int = 4242, processes: int = 2) -> np.ndarray:
    from .match_value import win_table
    with Pool(processes) as pool:
        parts = pool.map(_outcomes_job, [(rounds // processes, seed * 10 + k) for k in range(processes)])
    outs = [o for p in parts for o in p]
    t = win_table(outs)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    np.save(TABLE, t)
    return t


_T = None


def win(mine: int, theirs: int) -> float:
    global _T
    if _T is None:
        _T = np.load(TABLE)
    if mine >= 100 or theirs >= 100:
        return 1.0 if mine < theirs else 0.0 if mine > theirs else 0.5
    return float(_T[mine, theirs])


# ---------------------------------------------------------------- whole matches with paired round play-outs
def _finish_round(st, hero, opp, hs):
    from .contest_opus_playout import finish
    if not st.terminal:
        finish(st, hero, opp, hs)
    return st


def _match_job(job):
    hero_name, opp_name, alt_spec, n, seed = job
    from .contest_opus_playout import _fn, _txt, fork, player
    from .engine import MAX_PLIES, apply_action, legal_actions, make_view, new_round, resolve_showdown, mask_points
    from .features import action_key
    hero, opp = player(hero_name), player(opp_name)
    alt = _fn(alt_spec) if alt_spec else None
    rng = random.Random(seed)
    recs, results, nrounds = [], [], 0
    for m in range(n):
        hs = m % 2
        if m % 2 == 0:
            mseed = rng.getrandbits(64)
        mrng = random.Random(mseed)
        scores, leader, rounds = [0, 0], 0, 0
        while (max(scores) < 100 and rounds < 40) or (scores[0] == scores[1] and rounds < 43 and max(scores) >= 100):
            st = new_round(scores, leader, mrng)
            while not st.terminal:
                if st.ply >= MAX_PLIES:
                    resolve_showdown(st)
                    break
                v = make_view(st, st.turn)
                acts = legal_actions(v)
                if not acts:
                    resolve_showdown(st)
                    break
                if st.turn != hs:
                    apply_action(st, opp(v, acts, None))
                    continue
                a = hero(v, acts, None)
                got = alt(v, acts, a, {"hero": hero, "state": st}) if alt else None
                tag, extra = "", {}
                if isinstance(got, tuple):
                    got, tag, extra = (got + ({},))[:3]
                if got is not None and action_key(got) != action_key(a):
                    fa, fb = fork(st), fork(st)
                    apply_action(fa, a)
                    apply_action(fb, got)
                    _finish_round(fa, hero, opp, hs)
                    _finish_round(fb, hero, opp, hs)
                    wa = win(fa.scores[hs], fa.scores[1 - hs])
                    wb = win(fb.scores[hs], fb.scores[1 - hs])
                    ma = fa.result.added[1 - hs] - fa.result.added[hs]
                    mb = fb.result.added[1 - hs] - fb.result.added[hs]
                    recs.append({"tag": tag, "dwin": wb - wa, "dmargin": mb - ma, "ply": v.ply,
                                 "my": v.my_score, "opp": v.opp_score, "pts": mask_points(v.my_hand),
                                 "a": _txt(a), "b": _txt(got), **extra})
                apply_action(st, a)
            scores = list(st.scores)
            leader = 1 - leader
            rounds += 1
            nrounds += 1
        mine, theirs = scores[hs], scores[1 - hs]
        results.append(1.0 if mine < theirs else 0.0 if mine > theirs else 0.5)
    return recs, results, nrounds


def match_playout(hero, opp, alt, matches=400, seed=7000, processes=2):
    per = [(hero, opp, alt, matches // processes, seed * 100 + k) for k in range(processes)]
    with Pool(processes) as pool:
        parts = pool.map(_match_job, per)
    recs = [r for p in parts for r in p[0]]
    res = [x for p in parts for x in p[1]]
    nr = sum(p[2] for p in parts)
    return recs, res, nr


def summarize(recs, matches, rounds, by="tag") -> str:
    groups: dict = {}
    for r in recs:
        groups.setdefault(r[by], []).append(r)
    lines = [f"| {by} | n | per match | dwin each x100 (+-se) | dwin a match x100 | dmargin each (+-se) |",
             "|---|---|---|---|---|---|"]
    for k in sorted(groups, key=lambda k: str(k)):
        g = groups[k]
        n = len(g)
        w = [r["dwin"] for r in g]
        d = [r["dmargin"] for r in g]
        mw, md = sum(w) / n, sum(d) / n
        sw = math.sqrt(sum((x - mw) ** 2 for x in w) / max(1, n - 1) / n)
        sd = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, n - 1) / n)
        lines.append(f"| {k} | {n} | {n / matches:.3f} | {100 * mw:+.2f} (+-{100 * sw:.2f}) | "
                     f"{100 * mw * n / matches:+.3f} | {md:+.2f} (+-{sd:.2f}) |")
    return "\n".join(lines)


# ---------------------------------------------------------------- probes (rules B against the hero's move A)
def _info(view):
    from .engine import mask_points
    from .features import assaf_chance, opp_min_points, state_features
    f = state_features(view)
    return {"pts": mask_points(view.my_hand), "assaf": assaf_chance(view), "opp_min": opp_min_points(view),
            "opp_est": float(f[5 * 14 + 24]) * 40.0, "my": view.my_score, "opp": view.opp_score}


def _best_throw(view, actions, ctx):
    throws = [x for x in actions if x.kind != "yaniv"]
    return ctx["hero"](view, throws, None) if throws else None


def _yaniv(actions):
    for x in actions:
        if x.kind == "yaniv":
            return x
    return None


def _region(i) -> str:
    """Match situation of a Yaniv decision."""
    my, opp, pts = i["my"], i["opp"], i["pts"]
    if my + pts + 30 > 100:
        mine = "Cdie"                  # a caught call ends the match against me
    elif my + pts + 30 in (50, 100):
        mine = "Ccut"                  # a caught call cuts me (already a reason to call in hm5)
    else:
        mine = "C-"
    if opp + i["opp_est"] > 100.5:
        theirs = "Okill"               # a won call likely ends the match for me
    elif opp >= 80:
        theirs = "Onear"
    else:
        theirs = "O-"
    return f"{mine}/{theirs}"


def _abucket(a: float) -> str:
    return "a0" if a <= 0.0 else "a<.05" if a < 0.05 else "a<.15" if a < 0.15 else "a<.3" if a < 0.3 else "a>=.3"


def alt_flip(view, actions, a, ctx):
    """Probe: flip the Yaniv decision whenever Yaniv is legal (call <-> hero's best throw). Tag: region, assaf."""
    y = _yaniv(actions)
    if y is None:
        return None
    i = _info(view)
    if i["assaf"] <= 0.0:
        return None                    # sure calls: never worth flipping
    if a.kind == "yaniv":
        return _best_throw(view, actions, ctx), f"call->throw {_region(i)} {_abucket(i['assaf'])}"
    return y, f"throw->call {_region(i)} {_abucket(i['assaf'])}"


def _kept(view, act) -> int:
    from .engine import mask_points
    return mask_points(view.my_hand & ~act.discard_mask)


def _mtag(i) -> str:
    my, opp, pts = i["my"], i["opp"], i["pts"]
    m = "Mdie" if my + pts > 100 else "Mnear" if my + pts + 10 > 100 else "M-"
    o = "Ocdie" if opp + 30 > 100 else "O-"
    return f"{m}/{o}"


def alt_greedy(view, actions, a, ctx):
    """Probe: throw to the fewest kept points (hero picks the draw / ties) instead of the hero's throw."""
    if a.kind == "yaniv":
        return None
    throws = [x for x in actions if x.kind != "yaniv"]
    lo = min(_kept(view, x) for x in throws)
    if _kept(view, a) == lo:
        return None
    b = ctx["hero"](view, [x for x in throws if _kept(view, x) == lo], None)
    i = _info(view)
    return b, f"{_mtag(i)} d{min(3, (_kept(view, a) - lo + 2) // 3)}"


def alt_cutopp_truth(view, actions, a, ctx):
    """Upper bound (reads the truth, a measurement only): the hero calls and the opponent's real total would land
    its score exactly on 50/100 -> throw instead."""
    if a.kind != "yaniv":
        return None
    from .engine import mask_points
    st = ctx["state"]
    opp_pts = mask_points(st.hands[1 - st.turn])
    i = _info(view)
    if opp_pts > i["pts"] and view.opp_score + opp_pts in (50, 100):
        return _best_throw(view, actions, ctx), f"cut{view.opp_score + opp_pts}", _cutinfo(view, i, opp_pts)
    return None


def _cutinfo(view, i, opp_pts):
    from .engine import POINTS_OF, cards_from_mask
    known = cards_from_mask(view.opp_known)
    return {"opp_n": view.opp_size, "known_n": len(known), "known_pts": sum(POINTS_OF[c] for c in known),
            "opp_min": i["opp_min"], "opp_est": i["opp_est"], "opp_pts": opp_pts}


def p_exact(view, total: int) -> float:
    """Chance (hidden cards uniform over the unseen pool) that the opponent's hand totals exactly `total`."""
    from .engine import POINTS_OF, cards_from_mask
    from .features import pool_mask
    known = cards_from_mask(view.opp_known)
    room = total - sum(POINTS_OF[c] for c in known)
    need = view.opp_size - len(known)
    if need <= 0:
        return 1.0 if room == 0 else 0.0
    pool = [POINTS_OF[c] for c in cards_from_mask(pool_mask(view))]
    if room < 0 or need > len(pool):
        return 0.0
    ways = [[0] * (room + 1) for _ in range(need + 1)]
    ways[0][0] = 1
    for pt in pool:
        for k in range(need, 0, -1):
            row, below = ways[k], ways[k - 1]
            for s in range(room, pt - 1, -1):
                row[s] += below[s - pt]
    return ways[need][room] / math.comb(len(pool), need)


def _cut_target(view):
    return 50 if view.opp_score < 50 else 100


def alt_cutopp_p(view, actions, a, ctx):
    """Probe: the hero calls; if the chance (from the view) that the opponent's total lands its score exactly on the
    next cut is >= 2%, throw instead. Tag: cut, hidden cards, chance bucket."""
    if a.kind != "yaniv":
        return None
    from .engine import POINTS_OF, cards_from_mask, mask_points
    cut = _cut_target(view)
    p = p_exact(view, cut - view.opp_score)
    if cut - view.opp_score <= mask_points(view.my_hand) or p < 0.02:
        return None
    h = view.opp_size - len(cards_from_mask(view.opp_known))
    pb = "p<.05" if p < 0.05 else "p<.1" if p < 0.1 else "p<.2" if p < 0.2 else "p<.5" if p < 0.5 else "p>=.5"
    return _best_throw(view, actions, ctx), f"cut{cut} h{min(h, 3)} {pb}", {"p": p}


def alt_cutopp_d(view, actions, a, ctx):
    """Probe: the hero calls with the opponent's hidden cards h <= 2; d = cut - opp_score - known points.
    If the opponent's hidden cards could make d (d within 0..10 h) and that total is above my points, throw.
    Tag: cut, h, d (for h = 2, d in steps of 4)."""
    if a.kind != "yaniv":
        return None
    from .engine import POINTS_OF, cards_from_mask, mask_points
    known = cards_from_mask(view.opp_known)
    h = view.opp_size - len(known)
    if h > 2:
        return None
    cut = _cut_target(view)
    d = cut - view.opp_score - sum(POINTS_OF[c] for c in known)
    if d < 0 or d > 10 * h or cut - view.opp_score <= mask_points(view.my_hand):
        return None
    dd = d if h < 2 else 4 * (d // 4)
    return _best_throw(view, actions, ctx), f"h{h} d{dd:02d}", {"cut": cut, "d": d, "h": h,
                                                                 "p": p_exact(view, cut - view.opp_score)}


def _lands(view, act) -> bool:
    """An open-card take that leaves my score + held points exactly on 50 or 100."""
    from .engine import POINTS_OF
    if act.kind == "yaniv" or act.draw != "open":
        return False
    return view.my_score + _kept(view, act) + POINTS_OF[view.discard[-1]] in (50, 100)


def alt_steer(view, actions, a, ctx):
    """Probe of hm5's own cut steering (OPEN_CUT): if the hero's move lands on a cut, B = its best move that does
    not; if it does not but a landing open take exists, B = the best landing take. Tag: direction, cut, threat."""
    if a.kind == "yaniv":
        return None
    throws = [x for x in actions if x.kind != "yaniv"]
    land = [x for x in throws if _lands(view, x)]
    if not land:
        return None
    i = _info(view)
    thr = "t1" if i["opp_est"] <= 9 else "t0" if i["opp_est"] >= 15 else "tm"
    from .engine import POINTS_OF
    cut = view.my_score + _kept(view, land[0]) + POINTS_OF[view.discard[-1]]
    if _lands(view, a):
        rest = [x for x in throws if not _lands(view, x)]
        return ctx["hero"](view, rest, None), f"drop cut{cut} {thr}"
    return ctx["hero"](view, land, None), f"take cut{cut} {thr}"


def alt_cutopp_d3(view, actions, a, ctx):
    """Probe: as alt_cutopp_d for exactly 3 hidden opponent cards (d in steps of 4)."""
    if a.kind != "yaniv":
        return None
    from .engine import POINTS_OF, cards_from_mask, mask_points
    known = cards_from_mask(view.opp_known)
    h = view.opp_size - len(known)
    if h != 3:
        return None
    cut = _cut_target(view)
    d = cut - view.opp_score - sum(POINTS_OF[c] for c in known)
    if d < 0 or d > 30 or cut - view.opp_score <= mask_points(view.my_hand):
        return None
    return _best_throw(view, actions, ctx), f"h3 d{4 * (d // 4):02d}"


def alt_cutopp_h2k(view, actions, a, ctx):
    """Probe: the h = 2, d in 0..11 veto, tagged by how many of the opponent's cards are known (k)."""
    if a.kind != "yaniv":
        return None
    from .engine import POINTS_OF, cards_from_mask, mask_points
    known = cards_from_mask(view.opp_known)
    if view.opp_size - len(known) != 2:
        return None
    cut = _cut_target(view)
    d = cut - view.opp_score - sum(POINTS_OF[c] for c in known)
    if d < 0 or d > 11 or cut - view.opp_score <= mask_points(view.my_hand):
        return None
    return _best_throw(view, actions, ctx), f"h2 k{len(known)} cut{cut}"


# ---------------------------------------------------------------- the block
VETO = 2000.0       # = 2 YBIG: takes a called Yaniv row (+YBIG) down to -YBIG, under every throw
H2_RANGE = None     # (lo, hi) of d for the 2-hidden-card case, or None (set after the probe; see match.md)


def add_match_units(kit, h2_range=None) -> None:
    """Do not call Yaniv when the call would drop the opponent's score exactly on a cut (50 -> 0, 100 -> 50).

    The opponent's total is known up to its h hidden cards: known points K, d = cut - opp_score - K is what the
    hidden cards must make for the landing. Veto the call (Yaniv row only) when, for the next cut (50 or 100):
      h = 0 and d == 0 (the landing is certain), or
      h = 1 and 0 <= d <= 10 (one hidden card can make it; hm5-like opponents steer their hand onto the cut),
      [h = 2 and lo <= d <= hi, if h2_range]
    and the opponent's total (cut - opp_score) is above my points (else the call is caught and the opponent adds 0).
    Inputs: opp_score/120, known_pts/40, opp_cards/5, known_n/5, my_pts/40 - all integers after scaling, so every
    step below is exact (ramps one point wide)."""
    h2 = h2_range if h2_range is not None else H2_RANGE
    hv = {"opp_cards": 5.0, "known_n": -5.0}                       # h = hidden cards, 0..5
    kit.unit(1, "mt_h1", {k: -v for k, v in hv.items()}, bias=1.0, note="relu(1 - h): [h == 0]")
    kit.unit(1, "mt_h2", {k: -v for k, v in hv.items()}, bias=2.0, note="relu(2 - h)")
    is_h = {0: {"mt_h1": 1.0}, 1: {"mt_h2": 1.0, "mt_h1": -2.0}}
    if h2:
        kit.unit(1, "mt_h3", {k: -v for k, v in hv.items()}, bias=3.0, note="relu(3 - h)")
        is_h[2] = {"mt_h3": 1.0, "mt_h2": -2.0, "mt_h1": 1.0}
    for cut in (50, 100):
        d = {"opp_score": -120.0, "known_pts": -40.0}               # d = cut - opp_score - K
        kit.unit(1, f"mt{cut}_dp1", d, bias=cut + 1.0, note=f"relu(d + 1), d = {cut} - opp_score - known points")
        kit.unit(1, f"mt{cut}_d0", d, bias=float(cut), note="relu(d)")
        kit.unit(1, f"mt{cut}_dm1", d, bias=cut - 1.0, note="relu(d - 1)")
        kit.unit(1, f"mt{cut}_hi_a", {k: -v for k, v in d.items()}, bias=11.0 - cut, note="relu(11 - d)")
        kit.unit(1, f"mt{cut}_hi_b", {k: -v for k, v in d.items()}, bias=10.0 - cut, note="relu(10 - d)")
        e = {"opp_score": -120.0, "my_pts": -40.0}                  # e = cut - opp_score - my points
        kit.unit(1, f"mt{cut}_e_a", e, bias=float(cut), note=f"relu(e), e = {cut} - opp_score - my points")
        kit.unit(1, f"mt{cut}_e_b", e, bias=cut - 1.0, note="relu(e - 1): [e >= 1] = e_a - e_b")
        e_ok = {f"mt{cut}_e_a": 1.0, f"mt{cut}_e_b": -1.0}
        d_zero = {f"mt{cut}_dp1": 1.0, f"mt{cut}_d0": -2.0, f"mt{cut}_dm1": 1.0}
        d_in = {f"mt{cut}_dp1": 1.0, f"mt{cut}_d0": -1.0, f"mt{cut}_hi_a": 1.0, f"mt{cut}_hi_b": -1.0}
        cases = [(0, d_zero, 1), (1, d_in, 2)]
        if h2:
            lo, hi = h2
            kit.unit(1, f"mt{cut}_lo2_a", d, bias=cut - lo + 1.0, note=f"relu(d - {lo} + 1)")
            kit.unit(1, f"mt{cut}_lo2_b", d, bias=float(cut - lo), note=f"relu(d - {lo})")
            kit.unit(1, f"mt{cut}_hi2_a", {k: -v for k, v in d.items()}, bias=hi + 1.0 - cut, note=f"relu({hi} + 1 - d)")
            kit.unit(1, f"mt{cut}_hi2_b", {k: -v for k, v in d.items()}, bias=float(hi - cut), note=f"relu({hi} - d)")
            cases.append((2, {f"mt{cut}_lo2_a": 1.0, f"mt{cut}_lo2_b": -1.0,
                              f"mt{cut}_hi2_a": 1.0, f"mt{cut}_hi2_b": -1.0}, 2))
        for h, cond, n_ind in cases:
            w = {"yaniv": 100.0}
            for part in (is_h[h], cond, e_ok):
                for k, v in part.items():
                    w[k] = w.get(k, 0.0) + v
            kit.unit(2, f"mt{cut}_veto_h{h}", w, bias=-(100.0 + n_ind + 1.0),
                     note=f"Yaniv row, {h} hidden opponent cards, and they can land its score on {cut}: 1, else 0")
            kit.out({f"mt{cut}_veto_h{h}": -VETO})


def main(argv):
    cmd = argv[0]
    if cmd == "table":
        t = build_table()
        for a, b in ((0, 0), (50, 0), (0, 50), (80, 80), (90, 60), (60, 90), (95, 95)):
            print(a, b, round(float(t[a, b]), 3))
    elif cmd == "probe":
        opp, alt, n, seed = argv[1], argv[2], int(argv[3]), int(argv[4])
        hero = argv[5] if len(argv) > 5 else "hm5"
        recs, res, nr = match_playout(hero, opp, f"training.yaniv.contest_opus_match:{alt}", n, seed)
        print(f"hero {hero} vs {opp}, {alt}, {len(res)} matches seed {seed}: won {sum(res)} ({sum(res)/len(res):.1%}),"
              f" {nr} rounds")
        print(summarize(recs, len(res), nr))


if __name__ == "__main__":
    main(sys.argv[1:])

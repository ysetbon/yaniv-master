"""Team fable's measuring bench for the real game (5 cards, Yaniv at 7). It only plays and counts: no weight
is fitted or searched here.

A PLAYER is named by a string:
    counter                                   the card counter
    hm5 | trained | hm4                       the shipped nets (5-card hand-written, trained, 4-card hand-written)
    path/to/net.json                          any saved net
    py:contest_fable.build                    a net built by training/yaniv/<module>.<function>() ...
    py:contest_fable.build@{"LIVE":{"5":2}}   ... with keyword arguments (JSON)

    python -m training.yaniv.contest_fable_lab duel HERO OPP [OPP ...] --rounds 20000 --matches 5000 --seed 1
    python -m training.yaniv.contest_fable_lab wtable hm5 --rounds 40000 --out results/contest-fable-logs/wtable.npy

`paired(...)` is the play-both-moves-out instrument: the hero plays rounds (or whole matches) against an
opponent; at each of its decisions a PROBE (a function "module:name") may name alternative moves; the round is
played to the end both ways on the same hidden cards (the hero's own policy afterwards) and one record per
alternative is kept with both outcomes. A probe is

    def probe(view, actions, chosen, state) -> list[(label, alt_action, info_dict)]

(`state` is the true state: use it only to describe, never to choose). Leave YANIV_HAND_SIZE / YANIV_CALL_AT unset.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = {"hm5": "public/model/yaniv-net-5cards-handmade.json", "trained": "public/model/yaniv-net.json",
           "hm4": "public/model/yaniv-net-4cards-yaniv7-handmade.json"}
LOGS = ROOT / "results" / "contest-fable-logs"
_CACHE: dict = {}


# ---------------------------------------------------------------- players
def load(name: str):
    """The net a player string names (None for the counter)."""
    from .network import load_net

    if name == "counter":
        return None
    if name.startswith("py:"):
        target, _, raw = name[3:].partition("@")
        module, _, func = target.rpartition(".")
        built = getattr(importlib.import_module(f"training.yaniv.{module}"), func)(**(json.loads(raw) if raw else {}))
        return built[0] if isinstance(built, tuple) else built
    path = Path(SHIPPED.get(name, name))
    return load_net(path if path.is_absolute() else ROOT / path)


def player(name: str):
    """A chooser(view, actions, rng) for a player string, cached per process."""
    if name not in _CACHE:
        from .heuristic import choose_heuristic
        from .train import net_chooser

        net = load(name)
        _CACHE[name] = choose_heuristic if net is None else net_chooser(net, 0.0)
    return _CACHE[name]


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    p = wins / max(1, n)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


# ---------------------------------------------------------------- duel: rounds and matches, mirrored pairs
def _duel_job(job):
    from .train import match_duel

    kind, hero, opp, n, seed = job
    if kind == "matches":
        return match_duel(player(hero), player(opp), n, seed)
    # rounds, with the per-pair margin so the standard error is the paired one
    from .engine import play_round

    rng = random.Random(seed)
    h, o = player(hero), player(opp)
    out = {"n": 0, "wins": 0, "ties": 0, "mine": 0, "theirs": 0, "calls": 0, "caught": 0, "pair_sum": 0.0,
           "pair_sq": 0.0}
    for pair in range(n // 2):
        deal_seed = rng.getrandbits(64)
        total = 0
        for seat in (0, 1):
            choosers = (h, o) if seat == 0 else (o, h)
            st = play_round(choosers[0], choosers[1], random.Random(deal_seed), [0, 0], pair % 2)
            res = st.result
            mine, theirs = res.added[seat], res.added[1 - seat]
            out["n"] += 1
            out["wins"] += mine < theirs
            out["ties"] += mine == theirs
            out["mine"] += mine
            out["theirs"] += theirs
            if res.caller == seat:
                out["calls"] += 1
                out["caught"] += bool(res.assaf)
            total += theirs - mine
        out["pair_sum"] += total
        out["pair_sq"] += total * total
    return out


def _pool(workers: int):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    return ProcessPoolExecutor(workers)


def duel(hero: str, opp: str, rounds: int, matches: int, seed: int, workers: int = 4) -> dict:
    """Mirrored rounds from 0-0 and mirrored matches of `hero` against `opp`. Margin = opp's points - hero's."""
    out: dict = {"hero": hero, "opp": opp, "seed": seed}
    with _pool(workers) as pool:
        if rounds:
            size = max(2, (rounds // (workers * 4)) // 2 * 2)
            jobs = [("rounds", hero, opp, min(size, rounds - at), seed * 7919 + i)
                    for i, at in enumerate(range(0, rounds, size))]
            parts = list(pool.map(_duel_job, jobs))
            tot = {k: sum(p[k] for p in parts) for k in parts[0]}
            n, pairs = tot["n"], tot["n"] // 2
            mean_pair = tot["pair_sum"] / pairs
            var = max(0.0, tot["pair_sq"] / pairs - mean_pair ** 2)
            out["rounds"] = {"n": n, "margin": (tot["theirs"] - tot["mine"]) / n,
                             "se": math.sqrt(var / pairs) / 2, "won": tot["wins"] / n,
                             "calls": tot["calls"], "caught": tot["caught"] / max(1, tot["calls"])}
        if matches:
            size = max(2, (matches // (workers * 4)) // 2 * 2)
            jobs = [("matches", hero, opp, min(size, matches - at), seed * 104729 + 17 + i)
                    for i, at in enumerate(range(0, matches, size))]
            parts = list(pool.map(_duel_job, jobs))
            wins, losses, ties = (sum(p[k] for p in parts) for k in ("wins", "losses", "ties"))
            n = wins + losses + ties
            lo, hi = wilson(wins + ties / 2, n)
            out["matches"] = {"n": n, "wins": wins, "losses": losses, "ties": ties, "won": (wins + ties / 2) / n,
                              "lo": lo, "hi": hi}
    return out


def duel_line(res: dict) -> str:
    parts = [f"{res['hero']} vs {res['opp']} (seed {res['seed']}):"]
    r, m = res.get("rounds"), res.get("matches")
    if r:
        parts.append(f"margin {r['margin']:+.3f} +-{r['se']:.3f} a round ({r['n']} rounds), rounds won {r['won']:.1%}, "
                     f"calls caught {r['caught']:.1%};")
    if m:
        parts.append(f"matches {m['wins']}-{m['losses']} ({m['won']:.1%}, 95% {m['lo']:.1%}-{m['hi']:.1%})")
    return " ".join(parts)


# ---------------------------------------------------------------- paired play-both-moves-out
def fork(st):
    from .engine import State

    return State(list(st.hands), list(st.known), list(st.stock), list(st.discard), list(st.scores), st.turn, st.ply)


def finish(st, choosers) -> None:
    """Play a round on from any state (engine.play_round's loop)."""
    from .engine import MAX_PLIES, apply_action, legal_actions, make_view, resolve_showdown

    while not st.terminal:
        if st.ply >= MAX_PLIES:
            resolve_showdown(st)
            break
        v = make_view(st, st.turn)
        acts = legal_actions(v)
        if not acts:
            resolve_showdown(st)
            break
        apply_action(st, choosers[st.turn](v, acts, None))
    if not st.terminal:
        resolve_showdown(st)


def std_info(view, state=None) -> dict:
    """A standard description of a position (from the view; the opponent's true total if `state` is given)."""
    from .engine import POINTS_OF, cards_from_mask, mask_points, popcount, rank_of
    from .features import state_features

    f = state_features(view)
    info = {"mc": popcount(view.my_hand), "oc": view.opp_size, "ply": view.ply, "stock": view.stock_size,
            "pts": mask_points(view.my_hand), "hand": sorted(POINTS_OF[c] for c in cards_from_mask(view.my_hand)),
            "hand_ranks": sorted(rank_of(c) for c in cards_from_mask(view.my_hand)),
            "open": POINTS_OF[view.discard[-1]], "open_rank": rank_of(view.discard[-1]),
            "known_n": popcount(view.opp_known), "known_pts": mask_points(view.opp_known),
            "est": float(f[94]) * 40.0, "assaf": float(f[95]), "opp_min": float(f[85]) * 40.0,
            "s": view.my_score, "t": view.opp_score}
    if state is not None:
        info["opp_true"] = mask_points(state.hands[1 - view.seat])
    return info


def describe(action) -> dict:
    from .engine import POINTS_OF, cards_from_mask, mask_points, rank_of

    if action.kind == "yaniv":
        return {"call": True}
    cards = cards_from_mask(action.discard_mask)
    return {"call": False, "n": len(cards), "pts": mask_points(action.discard_mask), "card": POINTS_OF[cards[0]],
            "rank": rank_of(cards[0]), "draw": action.draw}


def _paired_job(job):
    from .engine import MATCH_LIMIT, apply_action, legal_actions, make_view, new_round
    from .features import action_key

    hero_name, opp_name, probe_name, n, seed, mode, after_name = job
    hero, opp = player(hero_name), player(opp_name)
    after = player(after_name) if after_name else hero     # the hero's policy after the forked move
    module, _, func = probe_name.partition(":")
    probe = getattr(importlib.import_module(f"training.yaniv.{module}"), func)
    rng = random.Random(seed)
    records, stats = [], {"rounds": 0, "decisions": 0, "margin": 0.0}

    def one_round(scores, leader, hs):
        st = new_round(scores, leader, rng)
        while not st.terminal and st.ply < 48:
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                break
            if st.turn != hs:
                apply_action(st, opp(v, acts, None))
                continue
            stats["decisions"] += 1
            a = hero(v, acts, None)
            alts = probe(v, acts, a, st)
            if alts:
                cache = {}

                def val(act):
                    k = action_key(act)
                    if k not in cache:
                        f = fork(st)
                        apply_action(f, act)
                        finish(f, (after, opp) if hs == 0 else (opp, after))
                        cache[k] = (f.result.added[hs], f.result.added[1 - hs])
                    return cache[k]

                a0, b0 = val(a)
                for label, alt, info in alts:
                    if action_key(alt) == action_key(a):
                        continue
                    a1, b1 = val(alt)
                    records.append({"label": label, "gain": (b1 - a1) - (b0 - a0), "a0": a0, "b0": b0, "a1": a1,
                                    "b1": b1, "s": v.my_score, "t": v.opp_score, **info})
            apply_action(st, a)
        finish(st, (hero, opp) if hs == 0 else (opp, hero))
        stats["rounds"] += 1
        stats["margin"] += st.result.added[1 - hs] - st.result.added[hs]
        return st

    for i in range(n):
        hs = i % 2
        if mode == "rounds":
            one_round([0, 0], (i // 2) % 2, hs)
        else:                                   # whole matches (the engine's play_match loop)
            scores, leader, played = [0, 0], 0, 0
            while (max(scores) < MATCH_LIMIT and played < 40) or (scores[0] == scores[1] and played < 43):
                scores = one_round(scores, leader, hs).scores
                leader = 1 - leader
                played += 1
    return records, stats


def paired(hero: str, opp: str, probe: str, n: int, seed: int, mode: str = "rounds", workers: int = 4,
           after: str = "") -> tuple[list[dict], dict]:
    """`n` rounds from 0-0 (mode "rounds") or `n` whole matches (mode "matches") of `hero` against `opp`.
    Returns (records, stats); a record's `gain` is the alternative's point margin minus the played move's, and
    a0/b0 (a1/b1) are the points (hero, opponent) added with the played (alternative) move, s/t the scores before.
    `after` names another player whose policy continues the hero's side after a forked move (default: the hero)."""
    size = max(1, n // (workers * 4))
    jobs = [(hero, opp, probe, min(size, n - at), seed * 7919 + i, mode, after)
            for i, at in enumerate(range(0, n, size))]
    with _pool(workers) as pool:
        parts = list(pool.map(_paired_job, jobs))
    records = [r for part, _ in parts for r in part]
    stats = {k: sum(s[k] for _, s in parts) for k in parts[0][1]}
    return records, stats


def summarize(records: list[dict], rounds: int, key=lambda r: r["label"], least: int = 30, gain="gain") -> str:
    """Per group: count, share of rounds, gain per occurrence +- standard error, gain per round."""
    groups: dict = {}
    for r in records:
        groups.setdefault(key(r), []).append(r[gain] if isinstance(gain, str) else gain(r))
    lines = []
    for k in sorted(groups, key=str):
        g = groups[k]
        if len(g) < least:
            continue
        mean = sum(g) / len(g)
        se = math.sqrt(max(0.0, sum(x * x for x in g) / len(g) - mean * mean) / len(g))
        lines.append(f"  {str(k):<60} n {len(g):6d} ({len(g) / rounds:.4f} a round)  each {mean:+7.3f} +-{se:.3f}"
                     f"  a round {sum(g) / rounds:+.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------- match-win table
def _outcome_job(job):
    from .engine import play_round

    a, b, n, seed = job
    rng = random.Random(seed)
    pa, pb = player(a), player(b)
    out = []
    for i in range(n):
        st = play_round(pa, pb, rng, [0, 0], i % 2)
        out.append(tuple(st.result.added))
    return out


def win_table(a: str, b: str, rounds: int, seed: int, workers: int = 4):
    """P[mine, theirs] of winning the match for two equal players whose rounds end like rounds of a against b
    (match_value.win_table: value iteration over the measured round outcomes; nothing is fitted)."""
    from .match_value import win_table as table_of

    size = max(1, rounds // (workers * 4))
    jobs = [(a, b, min(size, rounds - at), seed * 7919 + i) for i, at in enumerate(range(0, rounds, size))]
    with _pool(workers) as pool:
        outcomes = [o for part in pool.map(_outcome_job, jobs) for o in part]
    return table_of(outcomes), outcomes


def equity(table, mine: int, theirs: int) -> float:
    """Chance of winning the match after a round leaves these raw totals (cuts at exactly 50 / 100 applied)."""
    from .engine import MATCH_LIMIT, SCORE_CUTS

    mine, theirs = SCORE_CUTS.get(mine, mine), SCORE_CUTS.get(theirs, theirs)
    if mine >= MATCH_LIMIT or theirs >= MATCH_LIMIT:
        return 1.0 if mine < theirs else 0.0 if mine > theirs else 0.5
    return float(table[mine, theirs])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("duel")
    d.add_argument("hero")
    d.add_argument("opps", nargs="+")
    d.add_argument("--rounds", type=int, default=20000)
    d.add_argument("--matches", type=int, default=4000)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--log", help="append the result lines to this file under results/contest-fable-logs/")
    w = sub.add_parser("wtable")
    w.add_argument("a")
    w.add_argument("--b", default="")
    w.add_argument("--rounds", type=int, default=40000)
    w.add_argument("--seed", type=int, default=5)
    w.add_argument("--workers", type=int, default=4)
    w.add_argument("--out", default=str(LOGS / "wtable.npy"))
    args = parser.parse_args()
    if os.environ.get("YANIV_HAND_SIZE") or os.environ.get("YANIV_CALL_AT"):
        sys.exit("unset YANIV_HAND_SIZE / YANIV_CALL_AT: this bench plays the real game")
    if args.cmd == "duel":
        for opp in args.opps:
            t0 = time.time()
            text = duel_line(duel(args.hero, opp, args.rounds, args.matches, args.seed, args.workers))
            print(text + f"  [{time.time() - t0:.0f}s]", flush=True)
            if args.log:
                with open(LOGS / args.log, "a", encoding="utf-8") as fh:
                    fh.write(text + "\n")
    else:
        import numpy as np

        table, outcomes = win_table(args.a, args.b or args.a, args.rounds, args.seed, args.workers)
        np.save(args.out, table)
        with open(Path(args.out).with_suffix(".outcomes.pkl"), "wb") as fh:
            pickle.dump(outcomes, fh)
        print(f"wrote {args.out}: {len(outcomes)} rounds; P(win) at 0-0 {table[0, 0]:.3f}, at 90-50 {table[90, 50]:.3f}, "
              f"at 50-90 {table[50, 90]:.3f}, at 70-70 {table[70, 70]:.3f}")


if __name__ == "__main__":
    main()

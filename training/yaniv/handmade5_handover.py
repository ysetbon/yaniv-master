"""Handover analysis for the real game: where should the main net hand over to the 4-card net, and what would
the trained net do differently after the handover?

    python -m training.yaniv.handmade5_handover run --main MAIN.json --rounds 8000 --seed 920000 --out OUT.pkl
    python -m training.yaniv.handmade5_handover show OUT.pkl [OUT2.pkl ...]
    python -m training.yaniv.handmade5_handover swap --main MAIN.json --rounds 40000 --seed 922000 --out SWAP.pkl
    python -m training.yaniv.handmade5_handover cuts --matches 2000 MAIN.json MAIN.json:both MAIN.json:either ...

`run` plays hero = (MAIN until both hands hold 4 cards or fewer, then the 4-card hand-written net) against the
card counter, rounds from 0-0. It is a paired measurement, nothing is fitted and no weights are written:

- before the handover (a hand still holds 5 cards): the 4-card net (and the trained net, if MAIN is not it) is
  asked for its move; where it differs from MAIN's, both moves are played to the end of the round on the same
  hidden cards (hero's policy afterwards). Grouped by (my cards, opp cards) this prices an earlier handover:
  "mine" hands the cells (me <= 4, opp 5) to the 4-card net, "opp" the cells (me 5, opp <= 4), "either" both.
- after the handover: the trained net is asked for its move; where it differs from the 4-card net's, both are
  played out the same way. Every such decision is stored with what describes it (cards, ply, stock, the two
  moves) so `show` can group them.
- once a round, at hero's first decision after the handover: the round is also played out with the trained net
  making all of hero's remaining decisions ("whole endgame"), against the round as actually played.
- counts: decisions and rounds by (my cards, opp cards), for the handover frequencies of each rule.

Leave YANIV_HAND_SIZE / YANIV_CALL_AT unset (the real game is the default).
"""

from __future__ import annotations

import argparse
import collections
import math
import pickle
import random
import time
from pathlib import Path

FOUR = "public/model/yaniv-net-4cards-yaniv7-handmade.json"
TRAINED = "public/model/yaniv-net.json"


def describe(view, action) -> dict:
    """What a move is, in a few numbers."""
    from .engine import POINTS_OF, cards_from_mask, mask_points

    if action.kind == "yaniv":
        return {"call": True}
    cards = cards_from_mask(action.discard_mask)
    return {"call": False, "n": len(cards), "pts": mask_points(action.discard_mask),
            "card": POINTS_OF[cards[0]], "joker": cards[0] >= 52, "draw": action.draw}


def run(main_path: str, rounds: int, seed: int, out: str, log=print) -> None:
    from .engine import (MAX_PLIES, POINTS_OF, State, apply_action, cards_from_mask, legal_actions, make_view,
                         mask_points, new_round, popcount, resolve_showdown)
    from .features import action_key, assaf_chance, opp_min_points
    from .heuristic import choose_heuristic
    from .network import load_net
    from .train import net_chooser

    def load(path):
        return net_chooser(load_net(Path(path)), 0.0)

    main, four, trained = load(main_path), load(FOUR), load(TRAINED)
    main_is_trained = Path(main_path).resolve() == Path(TRAINED).resolve()

    def hero(view, actions, rng):
        return (four if max(popcount(view.my_hand), view.opp_size) <= 4 else main)(view, actions, rng)

    def fork(st):
        return State(list(st.hands), list(st.known), list(st.stock), list(st.discard), list(st.scores), st.turn,
                     st.ply)

    def finish(st, chooser, hs):
        while not st.terminal:
            if st.ply >= MAX_PLIES:
                resolve_showdown(st)
                break
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                resolve_showdown(st)
                break
            apply_action(st, (chooser if st.turn == hs else choose_heuristic)(v, acts, None))
        return st.result.added[1 - hs] - st.result.added[hs]

    rng = random.Random(seed)
    cells = collections.Counter()          # decisions by (my cards, opp cards)
    reach = collections.Counter()          # rounds with at least one decision under each rule
    records = []                           # differing decisions
    whole = []                             # per round reaching the handover: trained-for-the-rest minus as played
    total = 0.0
    t0 = time.time()
    for i in range(rounds):
        st = new_round([0, 0], i % 2, rng)
        hs = (i // 2) % 2
        seen = set()
        whole_alt = None
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
                apply_action(st, choose_heuristic(v, acts, None))
                continue
            mc, oc = popcount(v.my_hand), v.opp_size
            cells[(mc, oc)] += 1
            after = max(mc, oc) <= 4
            for rule, yes in (("both", after), ("mine", mc <= 4), ("opp", oc <= 4), ("either", min(mc, oc) <= 4),
                              ("both3", max(mc, oc) <= 3)):
                if yes:
                    seen.add(rule)
            a = hero(v, acts, None)
            if after and whole_alt is None:
                whole_alt = finish(fork(st), trained, hs)
            cache = {}

            def val(act):
                k = action_key(act)
                if k not in cache:
                    f = fork(st)
                    apply_action(f, act)
                    cache[k] = f.result.added[1 - hs] - f.result.added[hs] if f.terminal else finish(f, hero, hs)
                return cache[k]

            alts = [("trained", trained)] if after else \
                [("four", four)] + ([] if main_is_trained else [("trained", trained)])
            for name, alt in alts:
                b = alt(v, acts, None)
                if action_key(b) == action_key(a):
                    continue
                hand = sorted(POINTS_OF[c] for c in cards_from_mask(v.my_hand))
                records.append({
                    "round": i, "after": after, "alt": name, "mc": mc, "oc": oc, "ply": v.ply,
                    "stock": v.stock_size, "hand": hand, "pts": mask_points(v.my_hand),
                    "open": POINTS_OF[v.discard[-1]], "open_joker": v.discard[-1] >= 52,
                    "opp_known": popcount(v.opp_known), "opp_known_pts": mask_points(v.opp_known),
                    "assaf": assaf_chance(v) if a.kind == "yaniv" or b.kind == "yaniv" else None,
                    "opp_min": opp_min_points(v), "va": val(a), "vb": val(b),
                    "a": describe(v, a), "b": describe(v, b), "gain": val(b) - val(a)})
            apply_action(st, a)
        result = st.result.added[1 - hs] - st.result.added[hs]
        total += result
        for rule in seen:
            reach[rule] += 1
        if whole_alt is not None:
            whole.append(whole_alt - result)
        if (i + 1) % 1000 == 0:
            log(f"{i + 1} rounds, {time.time() - t0:.0f}s")
    with open(out, "wb") as fh:
        pickle.dump({"main": main_path, "rounds": rounds, "seed": seed, "margin": total / rounds, "cells": cells,
                     "reach": reach, "records": records, "whole": whole}, fh)
    log(f"wrote {out}: margin {total / rounds:+.2f}, {len(records)} differing decisions")


def swap(main_path: str, rounds: int, seed: int, out: str, log=print) -> None:
    """Every time hero calls Yaniv although the open card is lower than its highest card: play the round out
    both ways, the call, and "throw the highest single, take the open card" (hero's policy afterwards). Stores
    one record a call, with the trained net's own move there. A direct measurement of one alternative."""
    from .engine import (MAX_PLIES, POINTS_OF, State, apply_action, cards_from_mask, legal_actions, make_view,
                         mask_points, new_round, popcount, resolve_showdown)
    from .features import action_key, assaf_chance, opp_min_points
    from .heuristic import choose_heuristic
    from .network import load_net
    from .train import net_chooser

    main, four, trained = (net_chooser(load_net(Path(p)), 0.0) for p in (main_path, FOUR, TRAINED))

    def hero(view, actions, rng):
        return (four if max(popcount(view.my_hand), view.opp_size) <= 4 else main)(view, actions, rng)

    def finish(st, hs):
        while not st.terminal:
            if st.ply >= MAX_PLIES:
                resolve_showdown(st)
                break
            v = make_view(st, st.turn)
            acts = legal_actions(v)
            if not acts:
                resolve_showdown(st)
                break
            apply_action(st, (hero if st.turn == hs else choose_heuristic)(v, acts, None))
        return st.result.added[1 - hs] - st.result.added[hs]

    rng = random.Random(seed)
    records, calls = [], 0
    for i in range(rounds):
        st = new_round([0, 0], i % 2, rng)
        hs = (i // 2) % 2
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
                apply_action(st, choose_heuristic(v, acts, None))
                continue
            a = hero(v, acts, None)
            if a.kind == "yaniv":
                calls += 1
                singles = [x for x in acts if x.kind == "play" and x.draw == "open" and x.discard_mask.bit_count() == 1]
                best = max(singles, key=lambda x: mask_points(x.discard_mask))
                thrown, top = mask_points(best.discard_mask), POINTS_OF[v.discard[-1]]
                if top < thrown:
                    outcomes = []
                    for act in (a, best):
                        f = State(list(st.hands), list(st.known), list(st.stock), list(st.discard), list(st.scores),
                                  st.turn, st.ply)
                        apply_action(f, act)
                        outcomes.append(f.result.added[1 - hs] - f.result.added[hs] if f.terminal else finish(f, hs))
                    t = trained(v, acts, None)
                    records.append({
                        "after": max(popcount(v.my_hand), v.opp_size) <= 4, "mc": popcount(v.my_hand),
                        "oc": v.opp_size, "ply": v.ply, "pts": mask_points(v.my_hand), "open": top,
                        "thrown": thrown, "hand": sorted(POINTS_OF[c] for c in cards_from_mask(v.my_hand)),
                        "assaf": assaf_chance(v), "opp_min": opp_min_points(v), "opp_known": popcount(v.opp_known),
                        "va": outcomes[0], "vb": outcomes[1], "gain": outcomes[1] - outcomes[0],
                        "trained": "call" if t.kind == "yaniv" else "swap" if action_key(t) == action_key(best)
                        else "other"})
            apply_action(st, a)
        if (i + 1) % 5000 == 0:
            log(f"{i + 1} rounds")
    with open(out, "wb") as fh:
        pickle.dump({"main": main_path, "rounds": rounds, "seed": seed, "calls": calls, "records": records}, fh)
    log(f"wrote {out}: {calls} calls, {len(records)} with a lower open card")


def cuts(specs, matches: int, seed: int, log=print) -> None:
    """Matches to 100 against the counter, the same seed for every player: matches won and how often a score
    lands exactly on 50 or 100. A spec is `MAIN.json` (alone) or `MAIN.json:when[:at]` (handover to the 4-card net)."""
    from .engine import SCORE_CUTS, play_round, popcount
    from .heuristic import choose_heuristic
    from .hybrid import switch_chooser
    from .network import load_net
    from .train import net_chooser

    four = net_chooser(load_net(Path(FOUR)), 0.0)
    for spec in specs:
        parts = spec.split(":")
        hero = net_chooser(load_net(Path(parts[0])), 0.0)
        if len(parts) > 1:
            hero = switch_chooser(hero, four, int(parts[2]) if len(parts) > 2 else 4, parts[1])
        rng = random.Random(seed)
        wins = mine = theirs = rounds = five = 0
        t0 = time.time()
        for i in range(matches):
            seat = i % 2
            choosers = (hero, choose_heuristic) if seat == 0 else (choose_heuristic, hero)
            scores, leader, played = [0, 0], 0, 0
            while (max(scores) < 100 and played < 40) or (scores[0] == scores[1] and played < 43):
                state = play_round(choosers[0], choosers[1], rng, scores, leader)
                res = state.result
                for who in (0, 1):
                    if scores[who] + res.added[who] in SCORE_CUTS and res.added[who] > 0:
                        if who == seat:
                            mine += 1
                            five += max(popcount(res.hands[0]), popcount(res.hands[1])) == 5
                        else:
                            theirs += 1
                scores = state.scores
                leader = 1 - leader
                played += 1
            wins += scores[seat] < scores[1 - seat]
            rounds += played
        p = wins / matches
        log(f"{spec}: matches {wins}-{matches - wins} ({p:.1%} +- {math.sqrt(p * (1 - p) / matches):.1%}), "
            f"{rounds / matches:.1f} rounds a match, my cuts {mine / matches:.3f} a match "
            f"(+- {math.sqrt(mine) / matches:.3f}; {five / max(1, mine):.0%} of them in rounds ending with a "
            f"5-card hand), the counter's {theirs / matches:.3f}  [{time.time() - t0:.0f}s]")


def stat(rows, n):
    """Gain a round and its standard error for a list of differing decisions out of n rounds."""
    s = sum(r["gain"] for r in rows)
    s2 = sum(r["gain"] ** 2 for r in rows)
    c = len(rows)
    return f"n {c:5d} ({c / n:.3f} a round)  each {s / max(1, c):+6.2f}  a round {s / n:+.3f} +- {math.sqrt(s2) / n:.3f}"


def table(title, rows, n, key, least=20):
    print(f"\n{title}")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for k in sorted(groups, key=str):
        if len(groups[k]) >= least:
            print(f"  {str(k):<58} {stat(groups[k], n)}")
    print(f"  {'all':<58} {stat(rows, n)}")


def move_kind(r):
    a, b = r["a"], r["b"]
    if a["call"] or b["call"]:
        return "4-card net calls, trained plays on" if a["call"] else "4-card net plays on, trained calls"

    def d(x):
        return ("single" if x["n"] == 1 else "set") + ("+open" if x["draw"] == "open" else "+deck")
    same = (a["n"], a["card"], a["joker"]) == (b["n"], b["card"], b["joker"])
    rel = "same throw" if same else "more pts" if b["pts"] > a["pts"] else "fewer pts" if b["pts"] < a["pts"] \
        else "equal pts"
    return f"{d(a)} -> {d(b)}, {rel}"


def band(x, cuts):
    lo = None
    for c in cuts:
        if x < c:
            return f"<{c}" if lo is None else f"{lo}-{c - 1}"
        lo = c
    return f"{lo}+"


def show(paths) -> None:
    for path in paths:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        n, cells, recs = d["rounds"], d["cells"], d["records"]
        dec = sum(cells.values())
        print(f"\n===== {path}: main {Path(d['main']).name}, {n} rounds seed {d['seed']}, margin {d['margin']:+.2f}, "
              f"{dec} decisions ({dec / n:.2f} a round)")
        rules = {"both": lambda m, o: max(m, o) <= 4, "mine": lambda m, o: m <= 4, "opp": lambda m, o: o <= 4,
                 "either": lambda m, o: min(m, o) <= 4, "both3": lambda m, o: max(m, o) <= 3}
        print("handover frequencies (rounds reaching it / decisions after it):")
        for rule, f in rules.items():
            after = sum(c for (m, o), c in cells.items() if f(m, o))
            print(f"  {rule:<7} {d['reach'][rule] / n:6.1%}  {after / dec:6.1%}")
        w = d["whole"]
        if w:
            mean = sum(w) / len(w)
            sd = math.sqrt(sum((x - mean) ** 2 for x in w) / len(w))
            print(f"whole endgame played by the trained net instead of the 4-card net: {sum(w) / n:+.3f} a round "
                  f"+- {sd * math.sqrt(len(w)) / n:.3f} ({len(w)} rounds reach it, {sum(1 for x in w if x) } differ)")
        before = [r for r in recs if not r["after"]]
        for alt in ("four", "trained"):
            rows = [r for r in before if r["alt"] == alt]
            if rows:
                table(f"before the handover: {alt} net's move instead of the main net's, by (my cards, opp cards)",
                      rows, n, lambda r: (r["mc"], r["oc"]))
                table("  ... grouped as the handover rules group them", rows, n,
                      lambda r: "me<=4 opp 5 (mine)" if r["mc"] <= 4 else "me 5 opp<=4 (opp)" if r["oc"] <= 4
                      else "me 5 opp 5")
        after = [r for r in recs if r["after"]]
        table("after the handover: trained net's move instead of the 4-card net's, by (my cards, opp cards)",
              after, n, lambda r: (r["mc"], r["oc"]))
        table("  by a hand of 4 still in play (both3 would leave these to the main net)", after, n,
              lambda r: "a hand holds 4" if max(r["mc"], r["oc"]) == 4 else "both <= 3")
        table("  by kind of move", after, n, move_kind)
        table("  by ply", after, n, lambda r: "ply " + band(r["ply"], (6, 9, 12, 16)))
        table("  by stock size", after, n, lambda r: "stock " + band(r["stock"], (20, 26, 32, 36)))
        table("  by kind and open card", after, n,
              lambda r: f"{move_kind(r)} | open {'joker' if r['open_joker'] else band(r['open'], (1, 2, 3, 4, 5, 6, 8, 10))}",
              least=30)
        table("  by kind and opp cards", after, n, lambda r: f"{move_kind(r)} | opp {r['oc']}", least=30)
        table("  by kind and my cards", after, n, lambda r: f"{move_kind(r)} | me {r['mc']}", least=30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--main", required=True)
    r.add_argument("--rounds", type=int, default=8000)
    r.add_argument("--seed", type=int, default=920000)
    r.add_argument("--out", required=True)
    s = sub.add_parser("show")
    s.add_argument("files", nargs="+")
    w = sub.add_parser("swap")
    w.add_argument("--main", required=True)
    w.add_argument("--rounds", type=int, default=40000)
    w.add_argument("--seed", type=int, default=922000)
    w.add_argument("--out", required=True)
    c = sub.add_parser("cuts")
    c.add_argument("specs", nargs="+")
    c.add_argument("--matches", type=int, default=2000)
    c.add_argument("--seed", type=int, default=921000)
    args = parser.parse_args()
    if args.cmd == "run":
        run(args.main, args.rounds, args.seed, args.out)
    elif args.cmd == "swap":
        swap(args.main, args.rounds, args.seed, args.out)
    elif args.cmd == "cuts":
        cuts(args.specs, args.matches, args.seed)
    else:
        show(args.files)


if __name__ == "__main__":
    main()

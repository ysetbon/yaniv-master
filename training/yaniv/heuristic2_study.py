"""Studies behind counter2 (results/counter2-logs/): counting what happens, and playing two moves out.

    python -m training.yaniv.heuristic2_study calib  [--against counter] [--rounds 20000]
    python -m training.yaniv.heuristic2_study differ --alt public/model/yaniv-net.json [--against counter]
    python -m training.yaniv.heuristic2_study differ --alt-set CALL_RISK=0.2      # counter2 with one constant changed

`calib` plays counter2 against an opponent and compares what counter2 believed with what was true: the chance
its Yaniv call would be caught, and the chance the opponent calls on its next turn.

`differ` plays counter2 against an opponent; wherever another player (a net, or counter2 with a constant
changed) would have moved differently, the round is played to the end both ways on the same hidden cards, counter2
making every later move. The mean difference says which move was better. This tests stated rules; it does not
fit anything. Add `--hand 4` for the 4-card game. Seeds are 930000 + piece.
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

PIECE = 500


def _copy(s):
    from .engine import State

    return State(hands=list(s.hands), known=list(s.known), stock=list(s.stock), discard=list(s.discard),
                 scores=list(s.scores), turn=s.turn, ply=s.ply)


def _finish(state, choosers) -> None:
    from .engine import MAX_PLIES, apply_action, legal_actions, make_view, resolve_showdown

    while not state.terminal:
        if state.ply >= MAX_PLIES:
            resolve_showdown(state)
            break
        view = make_view(state, state.turn)
        actions = legal_actions(view)
        if not actions:
            resolve_showdown(state)
            break
        apply_action(state, choosers[state.turn](view, actions, None))


def _band(x: float, cuts) -> str:
    low = None
    for cut in cuts:
        if x < cut:
            return f"<{cut}" if low is None else f"{low}-{cut}"
        low = cut
    return f"{low}+"


def _calib_piece(job):
    against, n, seed, overrides = job
    from . import heuristic2 as h2
    from .engine import apply_action, legal_actions, make_view, mask_points, new_round
    from .heuristic2_eval import _apply, _player

    _apply(overrides)
    opp = _player(against)
    rng = random.Random(seed)
    out = defaultdict(lambda: [0, 0.0, 0.0])          # key -> [n, sum of the belief, sum of the truth]
    for i in range(n):
        me = i % 2
        state = new_round([0, 0], (i // 2) % 2, rng)
        pending = None
        before = None
        values = []
        waiting = []                                  # [cdf, its turns seen since] for my earlier throws
        while not state.terminal:
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                break
            if state.turn == me:
                belief = h2.beliefs(view)
                mine = mask_points(view.my_hand)
                theirs = mask_points(state.hands[1 - me])
                if before is not None:
                    for m in (3, 5, 7):
                        cell = out[(f"its total <= {m} at my NEXT turn, by the belief now", _band(before[m], (0.03, 0.1, 0.2, 0.35, 0.5)))]
                        cell[0] += 1
                        cell[1] += before[m]
                        cell[2] += theirs <= m
                before = belief["cdf"]
                values.append((("value by cards (mine, its)", view.my_hand.bit_count(), view.opp_size), belief["value"]))
                values.append((("value by ply", _band(view.ply, (2, 4, 6, 8, 10, 14, 20))), belief["value"]))
                values.append((("value by belief", _band(belief["value"], (-15, -10, -5, 0, 5, 10, 15))), belief["value"]))
                if view.opp_size <= 3:
                    values.append((("value by my points, it holds <= 3", _band(mine, (4, 8, 12, 16, 20, 25, 30))), belief["value"]))
                else:
                    values.append((("value by my points, it holds 4-5", _band(mine, (4, 8, 12, 16, 20, 25, 30))), belief["value"]))
                if mine <= h2.YANIV_LIMIT:
                    risk = belief["risk"]
                    for key in (("risk", _band(risk, (0.001, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5))),
                                ("risk by opp cards", view.opp_size, _band(risk, (0.001, 0.06, 0.15, 0.3))),
                                ("risk by my total", mine, _band(risk, (0.001, 0.06, 0.15, 0.3)))):
                        cell = out[key]
                        cell[0] += 1
                        cell[1] += risk
                        cell[2] += theirs <= mine
                action = h2.choose_counter2(view, actions)
                pending = None if action.kind == "yaniv" else (belief["hazard"], view.opp_size, belief["cdf"])
                waiting.append([belief["cdf"], 0])
            else:
                action = opp(view, actions, None)
                its = mask_points(view.my_hand)
                for item in waiting:
                    item[1] += 1
                    if item[1] in (2, 3):
                        for reach in ((9, 10, 11) if item[1] == 2 else (11, 12)):
                            cell = out[(f"it calls on its turn {item[1]} from now (if the round lasts), by P(its total <= {reach})",
                                        _band(item[0][reach], (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)))]
                            cell[0] += 1
                            cell[1] += item[0][reach]
                            cell[2] += action.kind == "yaniv"
                if its <= h2.YANIV_LIMIT:
                    for key in (("it calls, given its total", its),
                                ("it calls, given its total and my cards", its, view.opp_size),
                                ("it calls, given its total and its cards", its, view.my_hand.bit_count())):
                        cell = out[key]
                        cell[0] += 1
                        cell[2] += action.kind == "yaniv"
                if pending is not None:
                    hazard, size, cdf = pending
                    for reach in (7, 8, 9, 10, 11):
                        cell = out[(f"it calls next, by P(its total <= {reach})",
                                    _band(cdf[reach], (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)))]
                        cell[0] += 1
                        cell[1] += cdf[reach]
                        cell[2] += action.kind == "yaniv"
                    for key in (("hazard", _band(hazard, (0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6))),
                                ("hazard by opp cards", size, _band(hazard, (0.05, 0.2, 0.45)))):
                        cell = out[key]
                        cell[0] += 1
                        cell[1] += hazard
                        cell[2] += action.kind == "yaniv"
                    pending = None
            apply_action(state, action)
        if state.result is not None:
            result = state.result.added[1 - me] - state.result.added[me]
            for key, value in values:
                cell = out[key]
                cell[0] += 1
                cell[1] += value
                cell[2] += result
    return dict(out)


def _key_of(action, view) -> tuple:
    from .heuristic2 import RANK_OF, RANK_PTS

    if action.kind == "yaniv":
        return ("yaniv",)
    low = action.discard_mask & -action.discard_mask
    return (RANK_PTS[RANK_OF[low.bit_length() - 1]], RANK_OF[low.bit_length() - 1], action.discard_mask.bit_count(),
            action.draw)


def _kind(mine: tuple, alt: tuple, view) -> str:
    from .heuristic2 import RANK_OF, RANK_PTS

    if mine[0] == "yaniv":
        return "counter2 calls, the other throws"
    if alt[0] == "yaniv":
        return "the other calls, counter2 throws"
    open_pts = RANK_PTS[RANK_OF[view.discard[-1]]]
    if mine[1:3] == alt[1:3]:
        who = "counter2" if mine[3] == "open" else "the other"
        return f"same throw, {who} takes the open card ({'0-2' if open_pts <= 2 else '3-5' if open_pts <= 5 else '6+'})"
    draw = "" if mine[3] == alt[3] else ", other draw too"
    if mine[2] != alt[2]:
        who = "counter2" if mine[2] > alt[2] else "the other"
        low, high = (mine, alt) if mine[2] > alt[2] else (alt, mine)
        if low[1] == high[1]:
            return f"{who} throws more cards of the same rank" + draw
        return f"{who} throws a set, not a single" + draw
    if mine[0] == alt[0]:
        return "another rank of the same points" + draw
    who = "counter2" if mine[0] > alt[0] else "the other"
    return f"{who} throws the higher {'single' if mine[2] == 1 else 'set'}" + draw


def _differ_piece(job):
    alt_name, alt_set, against, n, seed, overrides, detail = job
    from . import heuristic2 as h2
    from .engine import apply_action, legal_actions, make_view, new_round
    from .heuristic2_eval import _apply, _player

    _apply(overrides)
    opp = _player(against)
    alt_player = _player(alt_name) if alt_name else None

    def alt(view, actions):
        if alt_player is not None:
            return alt_player(view, actions, None)
        saved = {key: getattr(h2, key) for key in alt_set}
        for key, value in alt_set.items():
            setattr(h2, key, value)
        h2.reset_tables()
        try:
            return h2.choose_counter2(view, actions)
        finally:
            for key, value in saved.items():
                setattr(h2, key, value)
            h2.reset_tables()

    rng = random.Random(seed)
    out = defaultdict(lambda: [0, 0.0, 0.0])          # kind -> [n, sum of (other - counter2), sum of squares]
    decisions = 0
    for i in range(n):
        me = i % 2
        state = new_round([0, 0], (i // 2) % 2, rng)
        choosers = [None, None]
        choosers[me] = h2.choose_counter2
        choosers[1 - me] = opp
        while not state.terminal:
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                break
            action = choosers[state.turn](view, actions, None)
            if state.turn == me:
                decisions += 1
                other = alt(view, actions)
                a, b = _key_of(action, view), _key_of(other, view)
                if a != b:
                    results = []
                    for move in (action, other):
                        fork = _copy(state)
                        apply_action(fork, move)
                        _finish(fork, choosers)
                        added = fork.result.added
                        results.append(added[1 - me] - added[me])
                    gain = results[1] - results[0]
                    kind = _kind(a, b, view)
                    keys = [kind, "ALL"]
                    if detail:
                        size = f"opp {view.opp_size} cards"
                        if "calls" in kind:
                            belief = h2.beliefs(view)
                            keys.append(f"{kind} | my total {sum(h2.RANK_PTS[h2.RANK_OF[c]] for c in range(54) if view.my_hand >> c & 1)}"
                                        f" | risk {_band(belief['risk'], (0.03, 0.06, 0.1, 0.15, 0.2, 0.3))}")
                            keys.append(f"{kind} | {size} | my cards {view.my_hand.bit_count()}")
                        elif "set, not a single" in kind:
                            pair, single = (a, b) if a[2] > b[2] else (b, a)
                            keys.append(f"{kind} | set of {pair[0]}s, single {_band(single[0], (3, 6, 10))}")
                            keys.append(f"{kind} | {size}")
                        else:
                            keys.append(f"{kind} | mine {a[0]}x{a[2]} {a[3]}, other {b[0]}x{b[2]} {b[3]}")
                            keys.append(f"{kind} | {size}")
                    for key in keys:
                        cell = out[key]
                        cell[0] += 1
                        cell[1] += gain
                        cell[2] += gain * gain
            apply_action(state, action)
    out["decisions"] = [decisions, 0.0, 0.0]
    out["rounds"] = [n, 0.0, 0.0]
    return dict(out)


def _calls_piece(job):
    against, n, seed, overrides = job
    from . import heuristic2 as h2
    from .engine import apply_action, legal_actions, make_view, new_round
    from .heuristic2_eval import _apply, _player

    _apply(overrides)
    opp = _player(against)
    rng = random.Random(seed)
    out = defaultdict(lambda: [0, 0.0, 0.0])          # cell -> [n, sum of (call - throw), sum of squares]
    for i in range(n):
        me = i % 2
        state = new_round([0, 0], (i // 2) % 2, rng)
        choosers = [None, None]
        choosers[me] = h2.choose_counter2
        choosers[1 - me] = opp
        while not state.terminal:
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                break
            if state.turn == me and actions[-1].kind == "yaniv":
                info = h2.evaluate(view, detail=True)
                if info["risk"] > 0.0 and info["move"] is not None:
                    results = []
                    for move in (actions[-1], h2._action_for(info["move"], actions)):
                        fork = _copy(state)
                        apply_action(fork, move)
                        _finish(fork, choosers)
                        added = fork.result.added
                        results.append(added[1 - me] - added[me])
                    gain = results[0] - results[1]
                    total = sum(h2.RANK_PTS[h2.RANK_OF[c]] for c in range(54) if view.my_hand >> c & 1)
                    edge = info["call"] - info["throw"]
                    risk = _band(info["risk"], (0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.5))
                    for key in (("by believed edge (call - throw)", _band(edge, (-12, -8, -5, -3, -1, 1, 3, 5, 8, 12))),
                                ("by risk", risk),
                                ("by my total and risk", total, risk),
                                ("by its cards and risk", view.opp_size, risk),
                                ("by my cards and risk", view.my_hand.bit_count(), risk)):
                        cell = out[key]
                        cell[0] += 1
                        cell[1] += gain
                        cell[2] += gain * gain
            apply_action(state, choosers[state.turn](view, actions, None))
    return dict(out)


def _merge(parts) -> dict:
    total: dict = {}
    for part in parts:
        for key, cell in part.items():
            have = total.setdefault(key, [0, 0.0, 0.0])
            for i in range(3):
                have[i] += cell[i]
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=("calib", "differ", "calls"))
    parser.add_argument("--against", default="counter")
    parser.add_argument("--alt", default="", help="the other player: counter, or a net file")
    parser.add_argument("--alt-set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--hand", type=int, default=5)
    parser.add_argument("--seed", type=int, default=930000)
    parser.add_argument("--detail", action="store_true", help="differ: also split each kind by the cards involved")
    parser.add_argument("--least", type=int, default=40, help="differ: leave out cells with fewer samples")
    args = parser.parse_args()
    os.environ["YANIV_HAND_SIZE"] = str(args.hand)
    os.environ["YANIV_CALL_AT"] = "7"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    overrides = {k: ast.literal_eval(v) for k, _, v in (item.partition("=") for item in args.set)}
    alt_set = {k: ast.literal_eval(v) for k, _, v in (item.partition("=") for item in args.alt_set)}
    pieces = [(min(PIECE, args.rounds - s), args.seed + i) for i, s in enumerate(range(0, args.rounds, PIECE))]
    with ProcessPoolExecutor(os.cpu_count() or 2) as pool:
        if args.study == "calib":
            total = _merge(pool.map(_calib_piece, [(args.against, n, seed, overrides) for n, seed in pieces]))
            print(f"counter2 against {args.against}, {args.rounds} rounds: belief against truth")
            print("| what | cell | n | believed | true |")
            print("|---|---|---|---|---|")
            for key in sorted(total, key=lambda k: tuple(str(x) for x in k)):
                n, belief, truth = total[key]
                print(f"| {key[0]} | {' / '.join(str(x) for x in key[1:])} | {n} | {belief / n:.3f} | {truth / n:.3f} |")
        elif args.study == "calls":
            total = _merge(pool.map(_calls_piece, [(args.against, n, seed, overrides) for n, seed in pieces]))
            print(f"counter2 against {args.against}, {args.rounds} rounds: every position where Yaniv is legal and "
                  "not sure, played out both ways (call, and counter2's best throw)")
            print("| split | cell | n | call minus throw | ±95% |")
            print("|---|---|---|---|---|")
            for key in sorted(total, key=lambda k: tuple(str(x) for x in k)):
                n, s, sq = total[key]
                if n < args.least:
                    continue
                mean = s / n
                sd = (max(0.0, sq / n - mean * mean) / n) ** 0.5
                print(f"| {key[0]} | {' / '.join(str(x) for x in key[1:])} | {n} | {mean:+.2f} | {1.96 * sd:.2f} |")
        else:
            jobs = [(args.alt, alt_set, args.against, n, seed, overrides, args.detail) for n, seed in pieces]
            total = _merge(pool.map(_differ_piece, jobs))
            rounds = total.pop("rounds")[0]
            decisions = total.pop("decisions")[0]
            name = args.alt or f"counter2 with {alt_set}"
            print(f"counter2 against {args.against}, {rounds} rounds, {decisions} decisions; the other player: {name}")
            print("gain = result with the other's move minus result with counter2's move, both played out by counter2")
            print("| kind of difference | n | a round | mean gain | ±95% | gain a round |")
            print("|---|---|---|---|---|---|")
            for key in sorted(total, key=lambda k: -abs(total[k][1])):
                n, s, sq = total[key]
                if n < args.least:
                    continue
                mean = s / n
                sd = (max(0.0, sq / n - mean * mean) / n) ** 0.5
                print(f"| {key} | {n} | {n / rounds:.3f} | {mean:+.2f} | {1.96 * sd:.2f} | {s / rounds:+.3f} |")


if __name__ == "__main__":
    sys.exit(main())

"""Measure rule-based players and nets against each other, with 95% ranges (results/counter2.md).

    python -m training.yaniv.heuristic2_eval counter2 --against counter
    python -m training.yaniv.heuristic2_eval counter2 counter --against counter public/model/yaniv-net.json
    python -m training.yaniv.heuristic2_eval counter2 --hand 4            # the 4-card game, Yaniv at 7
    python -m training.yaniv.heuristic2_eval counter2 --set CALL_RISK=0.2  # a variant: one constant changed
    python -m training.yaniv.heuristic2_eval counter2 --time              # seconds a move, next to the counter's

A player is `counter`, `counter2`, `greedy`, or a net file. Rounds are mirrored pairs from 0-0 (each deal is
played from both seats); the margin is the opponent's points minus the player's, a round, and its 95% range comes
from the spread of the pair means. Matches are mirrored pairs too; their range is the Wilson interval.

The test is cut into fixed pieces of 500 rounds / 100 matches, each with its own seed (seed * 7919 + piece), so
the deals do not depend on how many processes play them. `--set` changes a module constant of heuristic2.py in
every process, which is how the variants in the log were measured; nothing here fits or searches a constant.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROUND_PIECE = 500
MATCH_PIECE = 100


def wilson(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def _apply(overrides: dict) -> None:
    if overrides:
        from . import heuristic2

        for key, value in overrides.items():
            if not hasattr(heuristic2, key):
                raise KeyError(f"heuristic2 has no constant {key}")
            setattr(heuristic2, key, value)
        heuristic2.reset_tables()


def _player(name: str):
    from .parallel import bot_spec, chooser, net_spec

    if name in ("counter", "counter2", "greedy", "random"):
        return chooser(bot_spec(name))
    from .network import load_net

    return chooser(net_spec(load_net(Path(name))))


def _round_piece(hero: str, villain: str, n: int, seed: int, overrides: dict) -> dict:
    """`n` rounds as mirrored pairs (train.round_duel's deals), keeping each pair's margin for the range."""
    from .engine import play_round

    _apply(overrides)
    a, b = _player(hero), _player(villain)
    rng = random.Random(seed)
    out = {"n": n, "wins": 0, "ties": 0, "mine": 0, "theirs": 0, "calls": 0, "assafs": 0,
           "their_calls": 0, "their_assafs": 0, "pairs": 0, "pair_sum": 0.0, "pair_sq": 0.0}
    for pair in range(n // 2):
        deal_seed = rng.getrandbits(64)
        leader = pair % 2
        margin = 0
        for second in (0, 1):
            deal = random.Random(deal_seed)
            if not second:
                state = play_round(a, b, deal, [0, 0], leader)
                mine, theirs = state.result.added
                seat = 0
            else:
                state = play_round(b, a, deal, [0, 0], leader)
                theirs, mine = state.result.added
                seat = 1
            if state.result.caller == seat:
                out["calls"] += 1
                out["assafs"] += state.result.assaf
            elif state.result.caller is not None:
                out["their_calls"] += 1
                out["their_assafs"] += state.result.assaf
            out["mine"] += mine
            out["theirs"] += theirs
            out["wins"] += mine < theirs
            out["ties"] += mine == theirs
            margin += theirs - mine
        out["pairs"] += 1
        out["pair_sum"] += margin / 2
        out["pair_sq"] += (margin / 2) ** 2
    return out


def _match_piece(hero: str, villain: str, n: int, seed: int, overrides: dict) -> dict:
    from .train import match_duel

    _apply(overrides)
    return match_duel(_player(hero), _player(villain), n, seed)


def measure(pool, hero: str, villain: str, rounds: int, matches: int, seed_rounds: int, seed_matches: int,
            overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    out: dict = {}
    jobs = []
    for i, start in enumerate(range(0, rounds, ROUND_PIECE)):
        jobs.append(pool.submit(_round_piece, hero, villain, min(ROUND_PIECE, rounds - start),
                                seed_rounds * 7919 + i, overrides))
    match_jobs = []
    for i, start in enumerate(range(0, matches, MATCH_PIECE)):
        match_jobs.append(pool.submit(_match_piece, hero, villain, min(MATCH_PIECE, matches - start),
                                      seed_matches * 7919 + i, overrides))
    if jobs:
        parts = [job.result() for job in jobs]
        total = {key: sum(part[key] for part in parts) for key in parts[0]}
        pairs = total["pairs"]
        mean = total["pair_sum"] / pairs
        var = max(0.0, total["pair_sq"] / pairs - mean * mean) * pairs / max(1, pairs - 1)
        total["margin"] = mean
        total["margin_range"] = 1.96 * math.sqrt(var / pairs)
        out["rounds"] = total
    if match_jobs:
        parts = [job.result() for job in match_jobs]
        out["matches"] = {key: sum(part[key] for part in parts) for key in ("wins", "losses", "ties")}
    return out


def line(result: dict) -> str:
    parts = []
    r, m = result.get("rounds"), result.get("matches")
    if r:
        n = r["n"]
        lo, hi = wilson(r["wins"], n)
        parts.append(f"margin {r['margin']:+.2f} ±{r['margin_range']:.2f}, rounds won {r['wins'] / n:.1%} "
                     f"({lo:.1%}-{hi:.1%}), my calls caught {r['assafs']}/{r['calls']} "
                     f"({r['assafs'] / max(1, r['calls']):.1%}), its calls caught "
                     f"{r['their_assafs']}/{r['their_calls']} ({r['their_assafs'] / max(1, r['their_calls']):.1%})")
    if m:
        n = m["wins"] + m["losses"] + m["ties"]
        share = (m["wins"] + 0.5 * m["ties"]) / n
        lo, hi = wilson(m["wins"] + 0.5 * m["ties"], n)
        parts.append(f"matches {m['wins']}-{m['losses']}" + (f" ({m['ties']} tied)" if m["ties"] else "")
                     + f" ({share:.1%}, {lo:.1%}-{hi:.1%})")
    return "; ".join(parts)


def time_moves(names: list[str], rounds: int = 400) -> None:
    """Mean seconds a move of each player, on the same deals against the card counter."""
    from .engine import play_round

    villain = _player("counter")
    for name in names:
        chooser = _player(name)
        box = [0.0, 0]

        def timed(view, actions, rng, chooser=chooser, box=box):
            start = time.perf_counter()
            action = chooser(view, actions, rng)
            box[0] += time.perf_counter() - start
            box[1] += 1
            return action

        rng = random.Random(5)
        for i in range(rounds):
            play_round(timed, villain, rng, [0, 0], i % 2)
        print(f"{name}: {box[0] / box[1] * 1e6:.0f} microseconds a move ({box[1]} moves)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Players against each other, mirrored, with 95% ranges")
    parser.add_argument("players", nargs="+", help="counter, counter2, greedy or a net JSON file")
    parser.add_argument("--against", nargs="+", default=["counter"])
    parser.add_argument("--hand", type=int, default=5, help="cards each (5: the real game; 4: the smaller one)")
    parser.add_argument("--rounds", type=int, default=6000)
    parser.add_argument("--matches", type=int, default=1500)
    parser.add_argument("--seed-rounds", type=int, default=91000)
    parser.add_argument("--seed-matches", type=int, default=91001)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="change a constant of heuristic2.py for this run (a variant)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--time", action="store_true", help="only time the players' moves")
    args = parser.parse_args()
    # The engine reads the rules once, when first imported; the game processes inherit them.
    os.environ["YANIV_HAND_SIZE"] = str(args.hand)
    os.environ["YANIV_CALL_AT"] = "7"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = ast.literal_eval(value)
    _apply(overrides)
    if args.time:
        time_moves(args.players)
        return
    start = time.time()
    with ProcessPoolExecutor(args.workers or os.cpu_count() or 2) as pool:
        for hero in args.players:
            for villain in args.against:
                result = measure(pool, hero, villain, args.rounds, args.matches, args.seed_rounds,
                                 args.seed_matches, overrides)
                print(f"{hero} against {villain}: {line(result)}  [{time.time() - start:.0f}s]", flush=True)


if __name__ == "__main__":
    sys.exit(main())

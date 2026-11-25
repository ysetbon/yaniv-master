"""Measure a net against the card counter: the 4-card game with Yaniv at 7, or the real game (--game 5x7).

    python -m training.yaniv.handmade_eval public/model/yaniv-net-4cards-yaniv7-handmade.json
    python -m training.yaniv.handmade_eval NET.json --rounds 400 --matches 200      # a quick look
    python -m training.yaniv.handmade_eval NET.json --final                          # new games

The default is the gate-replay test the trained 4-card net was measured on:
2,000 rounds with seed 41000 and 800 matches with seed 41001. `--final` plays
the check on new games: 1,000 rounds with seed 31337 and 800 matches with
seed 31338.

`Workers` splits a test into pieces by its process count and seeds each
piece, so the deals depend on that count. The gate replay ran on 13
processes; this splits the same way (`--split-as 13`) whatever number of
processes plays the pieces, so the numbers compare on any machine.

With `--game 5x7` the real game is played (5 cards each, Yaniv at 7), and a
player may also be a switch player (hybrid.py), written

    switch:MAIN.json:ENDGAME.json            # hand over once both hands hold 4 cards or fewer
    switch:MAIN.json:ENDGAME.json:mine       # ... once my hand does (also: opp, either), and ":mine:3" for another size

    python -m training.yaniv.handmade_eval --game 5x7 counter public/model/yaniv-net.json \
        switch:public/model/yaniv-net.json:public/model/yaniv-net-4cards-yaniv7-handmade.json
    python -m training.yaniv.handmade_eval --game 5x7 --dev NET.json     # the bigger test on separate seeds
    python -m training.yaniv.handmade_eval --game 5x7 A.json --against B.json   # head to head
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def measure(pool, spec, opponent, rounds: int, matches: int, seed_rounds: int, seed_matches: int,
            split_as: int) -> dict:
    """Rounds and matches of `spec` against `opponent`, split as a pool of `split_as` processes would."""
    from .parallel import Workers, _matches, _merge_matches, _merge_rounds, _rounds

    out: dict = {}
    like = Workers(split_as)
    if rounds:
        chunks = like._chunks(rounds, 2)
        jobs = [(spec, opponent, size, seed_rounds * 7919 + i) for i, (_, size) in enumerate(chunks)]
        out["rounds"] = _merge_rounds(pool._run(_rounds, jobs, [s for _, s in chunks], rounds, None), rounds)
    if matches:
        chunks = like._chunks(matches, 2)
        jobs = [(spec, opponent, size, seed_matches * 7919 + i) for i, (_, size) in enumerate(chunks)]
        out["matches"] = _merge_matches(pool._run(_matches, jobs, [s for _, s in chunks], matches, None), matches)
    return out


def line(name: str, result: dict) -> str:
    parts = [f"{name}:"]
    r, m = result.get("rounds"), result.get("matches")
    if r:
        n = r["wins"] + r["losses"] + r["ties"]
        parts.append(f"margin {r['villain_avg'] - r['hero_avg']:+.2f} a round, rounds won {r['wins']}/{n} "
                     f"({r['win_rate']:.1%}), Yaniv calls {r['calls']} of which caught {r['assafs']} "
                     f"({r['assaf_rate']:.1%});")
    if m:
        n = m["wins"] + m["losses"] + m["ties"]
        lo, hi = wilson(m["wins"], n)
        parts.append(f"matches {m['wins']}-{m['losses']} ({m['win_rate']:.1%}, 95% range {lo:.1%}-{hi:.1%})")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="A net against the card counter: 4 cards, Yaniv at 7, or the real game")
    parser.add_argument("nets", nargs="+", help="net JSON files, the word 'counter', or switch:MAIN:ENDGAME[:when[:at]]")
    parser.add_argument("--game", choices=("4x7", "5x7"), default="4x7",
                        help="4x7: 4 cards each (default); 5x7: the real game, 5 cards each. Yaniv at 7 in both")
    parser.add_argument("--work", action="store_true",
                        help="the designers' working test: 6,000 rounds (seed 61000), 1,500 matches (seed 61001)")
    parser.add_argument("--dev", action="store_true",
                        help="the bigger test on separate seeds: 12,000 rounds (seed 52000), 3,000 matches (seed 52001)")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--matches", type=int, default=800, help="default 800; the real game's own checks played 400")
    parser.add_argument("--seed-rounds", type=int, default=41000)
    parser.add_argument("--seed-matches", type=int, default=41001)
    parser.add_argument("--final", action="store_true", help="the check on new games: 1,000 rounds, 800 matches")
    parser.add_argument("--against", default="counter", help="'counter' (default) or a net JSON file")
    parser.add_argument("--workers", type=int, default=0, help="processes (default: every core)")
    parser.add_argument("--split-as", type=int, default=0,
                        help="split the test as this many processes would (default: 13 for 4x7 and 11 for 5x7, "
                             "the counts the trained nets' own checks ran on)")
    parser.add_argument("--held-out", action="store_true",
                        help="5x7's one-time final check: 20,000 rounds (seed 77000), 4,000 matches (seed 77001)")
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()
    asked_matches = args.matches
    if args.final:
        args.rounds, args.seed_rounds, args.seed_matches = 1000, 31337, 31338
    args.split_as = args.split_as or (13 if args.game == "4x7" else 11)
    if args.final and args.game == "5x7" and args.matches == 800:
        args.matches = 400  # the trained real-game net's final check played 400
    if args.held_out:
        args.rounds, args.matches, args.seed_rounds, args.seed_matches = 20000, 4000, 77000, 77001
    if args.work:
        args.rounds, args.matches, args.seed_rounds, args.seed_matches = 6000, 1500, 61000, 61001
    if args.dev:
        args.rounds, args.matches, args.seed_rounds, args.seed_matches = 12000, 3000, 52000, 52001
    # The engine reads the rules once, when first imported; the game processes inherit them.
    os.environ["YANIV_HAND_SIZE"] = args.game[0]
    os.environ["YANIV_CALL_AT"] = "7"

    from .network import load_net
    from .parallel import Workers, bot_spec, net_spec, switch_spec

    def spec_of(name: str):
        if name in ("counter", "greedy", "random"):
            return bot_spec(name)
        if name.startswith("switch:"):
            parts = name.split(":")
            when = parts[3] if len(parts) > 3 else "both"
            at = int(parts[4]) if len(parts) > 4 else 4
            return switch_spec(spec_of(parts[1]), spec_of(parts[2]), at, when)
        return net_spec(load_net(Path(name)))

    if asked_matches == 0:
        args.matches = 0  # --matches 0 with a named test: rounds only

    pool = Workers(args.workers or os.cpu_count() or 2)
    opponent = spec_of(args.against)
    results = {}
    t0 = time.time()
    try:
        for name in args.nets:
            result = measure(pool, spec_of(name), opponent, args.rounds, args.matches, args.seed_rounds,
                             args.seed_matches, args.split_as)
            results[name] = result
            print(line(name, result) + f"  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        pool.close()
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

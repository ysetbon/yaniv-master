"""Measure a net against the card counter in the 4-card game with Yaniv at 7.

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
    parser = argparse.ArgumentParser(description="A net against the card counter: 4 cards, Yaniv at 7")
    parser.add_argument("nets", nargs="+", help="net JSON files, or the word 'counter'")
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--matches", type=int, default=800)
    parser.add_argument("--seed-rounds", type=int, default=41000)
    parser.add_argument("--seed-matches", type=int, default=41001)
    parser.add_argument("--final", action="store_true", help="the check on new games: 1,000 rounds, 800 matches")
    parser.add_argument("--against", default="counter", help="'counter' (default) or a net JSON file")
    parser.add_argument("--workers", type=int, default=0, help="processes (default: every core)")
    parser.add_argument("--split-as", type=int, default=13, help="split the test as this many processes would")
    parser.add_argument("--json", help="also write the results to this file")
    args = parser.parse_args()
    if args.final:
        args.rounds, args.seed_rounds, args.seed_matches = 1000, 31337, 31338
    # The engine reads the rules once, when first imported; the game processes inherit them.
    os.environ["YANIV_HAND_SIZE"] = "4"
    os.environ["YANIV_CALL_AT"] = "7"

    from .network import load_net
    from .parallel import Workers, bot_spec, net_spec

    def spec_of(name: str):
        return bot_spec(name) if name in ("counter", "greedy", "random") else net_spec(load_net(Path(name)))

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

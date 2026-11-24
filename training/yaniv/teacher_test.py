"""Is the search teacher stronger than the net it teaches?

A net can only learn what its teacher does better than it. This plays the
net itself and several search teachers built on it against the card counter,
on the same deals and the same match scores, and reports how each does
compared with the plain net, round by round:

    python -m training.yaniv.teacher_test --net public/model/yaniv-net.json

Half the rounds start at 0-0 and half from random match scores, so the test
sees the same mix of situations as training. Two measures:

- points: the counter's points minus the player's, a round;
- match chance: the player's chance of winning the match after the round
  (the table match_value.py builds), in percentage points. This is what the
  search optimises, and what the gate's matches measure.

The teachers: playing every imagined deal to the end ("playouts"), and the
value net ("judge") after 2 and after 4 moves. With --value the judge is a
saved value net (runs/<timestamp>/value-NN.json); otherwise one is trained
here first from the net playing itself.

For a smaller game pass --hand and --yaniv-at, as with main.py.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path


def scored_rounds(hero, villain, n: int, seed: int, table) -> list[tuple[int, int, float]]:
    """(hero's points, villain's points, hero's match chance after the round) for `n` mirrored rounds.

    Each deal is played twice with the seats (hands and scores) swapped, as
    in train.round_duel; odd pairs start from random match scores.
    """
    from .engine import play_round
    from .search import round_value

    rng = random.Random(seed)
    out = []
    deal_seed, scores = 0, [0, 0]
    for i in range(n):
        pair, second = divmod(i, 2)
        if not second:
            deal_seed = rng.getrandbits(64)
            pick = random.Random(deal_seed ^ 0x5C0E)
            scores = [pick.randint(0, 95), pick.randint(0, 95)] if pair % 2 else [0, 0]
        leader = pair // 2 % 2
        if not second:
            state = play_round(hero, villain, random.Random(deal_seed), list(scores), leader)
            seat = 0
        else:
            state = play_round(villain, hero, random.Random(deal_seed), list(scores), leader)
            seat = 1
        added = state.result.added
        out.append((added[seat], added[1 - seat], round_value(state, seat, table)))
    return out


def _scored(a, b, n: int, seed: int, table) -> list:
    from .parallel import chooser

    return scored_rounds(chooser(a), chooser(b), n, seed, table)


def play(workers, spec, n: int, seed: int, table, progress=None) -> list:
    """`scored_rounds` for `spec` against the counter, split over the game processes in a fixed way.

    The split depends only on `n` and the number of processes, so every
    player gets exactly the same deals in the same order.
    """
    from .parallel import bot_spec

    counter = bot_spec("counter")
    if workers.count == 1:
        return _scored(spec, counter, n, seed, table)
    chunks = workers._chunks(n, 2)
    jobs = [(spec, counter, size, seed * 7919 + i, table) for i, (_, size) in enumerate(chunks)]
    parts = workers._run(_scored, jobs, [size for _, size in chunks], n, progress)
    return [row for part in parts for row in part]


def summarize(rows: list, base: list | None) -> dict:
    n = len(rows)
    margin = [v - h for h, v, _ in rows]
    chance = [c for _, _, c in rows]
    out = {"rounds": n, "margin": sum(margin) / n, "won": sum(1 for h, v, _ in rows if h < v) / n,
           "chance": sum(chance) / n}
    if base is not None:
        for key, ours, theirs in (("margin", margin, [v - h for h, v, _ in base]),
                                  ("chance", chance, [c for _, _, c in base])):
            diffs = [a - b for a, b in zip(ours, theirs)]
            mean = sum(diffs) / n
            sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(1, n - 1))
            out[f"{key}_gain"], out[f"{key}_se"] = mean, sd / math.sqrt(n)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Is the search teacher stronger than the net it teaches?")
    parser.add_argument("--net", default="public/model/yaniv-net.json")
    parser.add_argument("--value", help="a saved value net (runs/<timestamp>/value-NN.json); default: train one here")
    parser.add_argument("--rounds", type=int, default=1000, help="rounds each player plays, mirrored pairs (default 1000)")
    parser.add_argument("--teachers", default="playouts:96,judge:2:96,judge:4:192",
                        help="comma-separated: playouts:SAMPLES or judge:DEPTH:SAMPLES")
    parser.add_argument("--hand", type=int)
    parser.add_argument("--yaniv-at", type=int)
    parser.add_argument("--workers", type=int, default=0, help="game processes (default: all cores but two)")
    parser.add_argument("--effort", type=float, help="share of the cores to use, like main.py")
    args = parser.parse_args()
    # The engine reads the rules once, when first imported; the game processes inherit them.
    if args.hand is not None:
        os.environ["YANIV_HAND_SIZE"] = str(args.hand)
    if args.yaniv_at is not None:
        os.environ["YANIV_CALL_AT"] = str(args.yaniv_at)

    from .engine import RULES_NAME
    from .match_value import win_table
    from .network import load_net
    from .parallel import Workers, default_workers, net_spec, search_spec
    from .value import explained, train_value, value_net

    teachers = []
    for item in args.teachers.split(","):
        parts = item.strip().split(":")
        if parts[0] == "playouts" and len(parts) == 2:
            teachers.append((f"playouts, {parts[1]} deals", None, int(parts[1])))
        elif parts[0] == "judge" and len(parts) == 3:
            teachers.append((f"judge after {parts[1]} moves, {parts[2]} deals", int(parts[1]), int(parts[2])))
        else:
            parser.error(f"--teachers: {item!r} is not playouts:SAMPLES or judge:DEPTH:SAMPLES")

    def log(message: str) -> None:
        print(message, flush=True)

    net = load_net(Path(args.net))
    pool = Workers(args.workers or default_workers(args.effort))
    t0 = time.time()
    try:
        log(f"Rules: {RULES_NAME}. Net: {args.net}. {pool.count} game processes, {args.rounds} rounds per player")
        table = win_table(pool.round_outcomes(net, 4000, 4242))
        vnet = None
        if any(depth is not None for _, depth, _ in teachers):
            if args.value:
                vnet = load_net(Path(args.value))
                log(f"Judge: {args.value}")
            else:
                vx, vy = pool.value_data(net, 6000, 5001, table)
                hx, hy = pool.value_data(net, 600, 5002, table)
                vnet = value_net()
                train_value(vnet, vx, vy, epochs=6)
                log(f"Judge: trained here on {len(vy) // 2:,} positions of the net playing itself; it explains "
                    f"{explained(vnet, hx, hy):.0%} of how new rounds ended  [{time.time() - t0:.0f}s]")

        base = play(pool, net_spec(net), args.rounds, 777, table)
        s = summarize(base, None)
        log(f"{'the net alone':>32}: margin {s['margin']:+.2f} a round, rounds won {s['won']:.1%}, "
            f"match chance {s['chance']:.1f}%  [{time.time() - t0:.0f}s]")
        for name, depth, samples in teachers:
            spec = search_spec(net, vnet if depth is not None else None, samples, depth or 2, table=table, seed=99)
            rows = play(pool, spec, args.rounds, 777, table)
            s = summarize(rows, base)
            log(f"{name:>32}: margin {s['margin']:+.2f} ({s['margin_gain']:+.2f} ± {s['margin_se']:.2f} vs the net), "
                f"match chance {s['chance']:.1f}% ({s['chance_gain']:+.2f} ± {s['chance_se']:.2f})  "
                f"[{time.time() - t0:.0f}s]")
        log("A teacher worth learning from beats the net by more than twice its ± on match chance.")
    finally:
        pool.close()


if __name__ == "__main__":
    sys.exit(main())

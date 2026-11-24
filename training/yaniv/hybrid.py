"""Hand over to a smaller game's net once both hands are small.

A hand never grows: every throw is followed by exactly one draw, so a hand
keeps its size after a single card and shrinks after a set. Once both
players hold 4 cards or fewer, the rest of the round is a game of 4 cards,
and a net trained on that game (main.py --hand 4 --yaniv-at 7, with the real
game's Yaniv limit) may play it better than the net trained on the whole game.

This plays the real game with a switch player (the main net until both hands
are small, then the smaller game's net) and with the main net alone, against
the card counter on the same deals, and the two against each other:

    python -m training.yaniv.hybrid --endgame public/model/yaniv-net-4cards-yaniv7.json

Run it under the real game's rules (no YANIV_HAND_SIZE / YANIV_CALL_AT set).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from .engine import HAND_SIZE, STANDARD_RULES, YANIV_LIMIT, View, play_round, popcount


def small_enough(view: View, at: int) -> bool:
    return popcount(view.my_hand) <= at and view.opp_size <= at


def switch_chooser(main, endgame, at: int = 4):
    """`main` chooses until both hands hold `at` cards or fewer; from then on `endgame` does."""

    def choose(view: View, actions, rng):
        return (endgame if small_enough(view, at) else main)(view, actions, rng)

    return choose


def handover_share(main, n_rounds: int, seed: int, at: int = 4) -> dict:
    """How often the switch happens: rounds that reach it, and decisions made after it (main net in both seats)."""
    from .heuristic import choose_heuristic

    rng = random.Random(seed)
    counts = {"decisions": 0, "small": 0, "rounds": 0, "rounds_small": 0}

    def tracked(chooser):
        def choose(view, actions, rng_):
            counts["decisions"] += 1
            if small_enough(view, at):
                counts["small"] += 1
                reached["flag"] = True
            return chooser(view, actions, rng_)
        return choose

    for i in range(n_rounds):
        reached = {"flag": False}
        if i % 2:
            play_round(tracked(main), tracked(choose_heuristic), rng, [0, 0], i // 2 % 2)
        else:
            play_round(tracked(choose_heuristic), tracked(main), rng, [0, 0], i // 2 % 2)
        counts["rounds"] += 1
        counts["rounds_small"] += reached["flag"]
    return {"rounds_reaching": counts["rounds_small"] / max(1, counts["rounds"]),
            "decisions_after": counts["small"] / max(1, counts["decisions"])}


def _wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def compare(main_path: Path, endgame_path: Path, rounds: int, matches: int, workers: int, at: int = 4,
            log=print) -> dict:
    from .network import load_net
    from .parallel import Workers, bot_spec, default_workers, net_spec, switch_spec
    from .train import net_chooser

    if not STANDARD_RULES:
        raise SystemExit(f"Run this under the real game's rules, not {HAND_SIZE} cards with Yaniv at {YANIV_LIMIT}")
    main, endgame = load_net(main_path), load_net(endgame_path)
    trained_on = (json.loads(endgame_path.read_text(encoding="utf-8")).get("meta") or {}).get("rules") or {}
    if trained_on and trained_on.get("yaniv_at") != YANIV_LIMIT:
        log(f"Note: {endgame_path.name} was trained with Yaniv at {trained_on.get('yaniv_at')}; this game calls at "
            f"{YANIV_LIMIT}, so its end of the round is not quite the game it learned")
    share = handover_share(net_chooser(main, 0.0), 400, 77, at)
    log(f"With {main_path.name} playing the counter, {share['rounds_reaching']:.0%} of rounds reach {at} cards each "
        f"or fewer, and {share['decisions_after']:.0%} of decisions come after that point")
    counter = bot_spec("counter")
    alone, hybrid = net_spec(main), switch_spec(net_spec(main), net_spec(endgame), at)
    pool = Workers(workers or default_workers())
    out: dict = {"handover": share}
    t0 = time.time()
    try:
        for name, spec in (("main alone", alone), ("switch", hybrid)):
            r = pool.rounds(spec, counter, rounds, 515151)
            m = pool.matches(spec, counter, matches, 525252)
            lo, hi = _wilson(m["wins"], matches)
            out[name] = {"rounds": r, "matches": m}
            log(f"{name:>10} vs counter: margin {r['villain_avg'] - r['hero_avg']:+.2f} a round, rounds "
                f"{r['win_rate']:.1%}, matches {m['wins']}-{m['losses']} ({m['win_rate']:.1%}, "
                f"95% range {lo:.1%}-{hi:.1%})  [{time.time() - t0:.0f}s]")
        r = pool.rounds(hybrid, alone, rounds, 535353)
        m = pool.matches(hybrid, alone, matches, 545454)
        lo, hi = _wilson(m["wins"], m["wins"] + m["losses"])
        out["head to head"] = {"rounds": r, "matches": m}
        log(f"switch vs main alone: margin {r['villain_avg'] - r['hero_avg']:+.2f} a round for the switch, matches "
            f"{m['wins']}-{m['losses']} ({m['win_rate']:.1%}, 95% range {lo:.1%}-{hi:.1%})  [{time.time() - t0:.0f}s]")
    finally:
        pool.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Does a 4-card net play the end of a real round better?")
    parser.add_argument("--main", default="public/model/yaniv-net.json", help="the real game's net")
    parser.add_argument("--endgame", default="public/model/yaniv-net-4cards-yaniv7.json",
                        help="a net trained with --hand 4 (best with --yaniv-at 7, the real game's limit)")
    parser.add_argument("--at", type=int, default=4, help="hand over once both hands hold this many cards or fewer")
    parser.add_argument("--rounds", type=int, default=4000, help="rounds each player plays, mirrored pairs")
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=0, help="game processes (default: all cores but two)")
    args = parser.parse_args()
    compare(Path(args.main), Path(args.endgame), args.rounds, args.matches, args.workers, args.at)


if __name__ == "__main__":
    main()

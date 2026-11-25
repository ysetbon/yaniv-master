"""Team opus: one or more nets against several opponents in the real game (5 cards, Yaniv at 7).

    python -m training.yaniv.contest_opus_eval NET.json [NET2.json ...] [--opps counter,trained,hm5,hm4]
           [--rounds 4000] [--matches 800] [--seed 300]

Rounds from 0-0 (mirrored pairs, train.round_duel) and matches (mirrored pairs, train.match_duel), split like
handmade_eval.py (11 pieces) so the same seed gives the same deals for every net. Margin is the opponent's points
minus the net's, a round. Rounds use seed S*1000+1, matches S*1000+2 for --seed S. Nothing is fitted: it plays.
Names: counter, trained (yaniv-net.json), hm5 (handmade5), hm4 (4-card hand-written net), opus (the shipped
contest net), or any net JSON path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("nets", nargs="+")
    p.add_argument("--opps", default="counter,trained,hm5")
    p.add_argument("--rounds", type=int, default=4000)
    p.add_argument("--matches", type=int, default=800)
    p.add_argument("--seed", type=int, default=300)
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()
    for k in ("YANIV_HAND_SIZE", "YANIV_CALL_AT"):
        os.environ.pop(k, None)     # the real game (engine defaults)
    from .contest_opus_playout import NAMED
    from .handmade_eval import line, measure
    from .network import load_net
    from .parallel import Workers, bot_spec, net_spec

    def spec(name):
        return bot_spec("counter") if name == "counter" else net_spec(load_net(Path(NAMED.get(name, name))))

    pool = Workers(a.workers or os.cpu_count() or 2)
    t0 = time.time()
    try:
        for opp in a.opps.split(","):
            for net in a.nets:
                r = measure(pool, spec(net), spec(opp), a.rounds, a.matches, a.seed * 1000 + 1, a.seed * 1000 + 2, 11)
                print(f"vs {opp:8s} " + line(Path(net).stem if net.endswith(".json") else net, r)
                      + f"  [{time.time() - t0:.0f}s]", flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    sys.exit(main())

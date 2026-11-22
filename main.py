#!/usr/bin/env python3
"""Train the Yaniv net and watch it live in your browser.

    python main.py                  # full run, GPU if PyTorch can see one
    python main.py --quick          # a two-minute smoke run
    python main.py --device cuda    # fail loudly if the NVIDIA GPU is not usable
    python main.py --legacy         # the first training recipe, for comparison

Each generation is saved under runs/<timestamp>/. The shipped model in
public/model/ is only replaced with --export.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import argparse

from training.yaniv.backend import CHOICES, resolve_device
from training.yaniv.live.server import run_live
from training.yaniv.train import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Yaniv net with a live dashboard.")
    parser.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    parser.add_argument("--device", default="auto", choices=CHOICES,
                        help="auto picks CUDA, then Apple MPS, then NumPy on the CPU")
    parser.add_argument("--legacy", action="store_true",
                        help="the first recipe: fresh wins only, unweighted, 48 test matches, one core")
    parser.add_argument("--generations", type=int, help="self-play passes after the copied net (default 8)")
    parser.add_argument("--games", type=int, help="self-play games per generation (default 800)")
    parser.add_argument("--batch", type=int, help="decisions per gradient step (default 72)")
    parser.add_argument("--workers", type=int,
                        help="CPU processes that play games (default: CPU cores minus two)")
    parser.add_argument("--test-matches", type=int, help="matches against the counter per generation (default 200)")
    parser.add_argument("--ladder-rounds", type=int, default=100,
                        help="rounds each new generation plays against every older one, 0 to skip")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--export", action="store_true",
                        help="also write the best net to public/model/ for the game")
    args = parser.parse_args()

    cfg = TrainConfig.quick() if args.quick else TrainConfig.legacy() if args.legacy else TrainConfig()
    if args.generations is not None:
        cfg.generations = max(0, args.generations)
    if args.games is not None:
        cfg.games_per_generation = max(1, args.games)
    if args.batch is not None:
        cfg.batch = max(1, args.batch)
    if args.workers is not None:
        cfg.workers = max(1, args.workers)
    if args.test_matches is not None:
        cfg.eval_matches = max(2, args.test_matches)
    cfg.ladder_rounds = max(0, args.ladder_rounds if not args.quick else min(args.ladder_rounds, 30))

    device = resolve_device(args.device)
    run_live(cfg, device, args.port, open_browser=not args.no_browser, export=args.export)


if __name__ == "__main__":
    main()

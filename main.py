#!/usr/bin/env python3
"""Train the Yaniv net and watch it live in your browser.

    python main.py                  # genetic algorithm: evolve a population of nets
    python main.py --recipe search  # learn from a search teacher that plays each move out
    python main.py --recipe selfplay  # fine-tune one net on its own wins instead
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
    parser.add_argument("--recipe", choices=("genetic", "search", "selfplay"), default="genetic",
                        help="genetic evolves a population of nets (default); search copies a teacher that plays "
                             "each move out; selfplay fine-tunes one net on its wins")
    parser.add_argument("--search-rounds", type=int, help="search: rounds labelled per pass (default 150)")
    parser.add_argument("--search-samples", type=int, help="search: imagined deals per searched move (default 48)")
    parser.add_argument("--search-confidence", type=float,
                        help="search: standard errors a move must win by to replace the net's own (default 1.5)")
    parser.add_argument("--population", type=int, help="genetic: networks per generation (default 24)")
    parser.add_argument("--fitness-rounds", type=int, help="genetic: rounds each network plays per generation, in mirrored pairs (default 1000)")
    parser.add_argument("--legacy", action="store_true",
                        help="the first recipe: fresh wins only, unweighted, 48 test matches, one core")
    parser.add_argument("--generations", type=int, help="generations after the copied net (default 20 genetic, 8 self-play)")
    parser.add_argument("--games", type=int, help="self-play games per generation (default 800)")
    parser.add_argument("--batch", type=int, help="decisions per gradient step (default 72)")
    parser.add_argument("--workers", type=int,
                        help="CPU processes that play games (default: CPU cores minus two)")
    parser.add_argument("--test-matches", type=int, help="matches against the counter per generation (default 200)")
    parser.add_argument("--finalists", type=int,
                        help="generations the gate replays on shared games when training ends, 0 to skip (default 4)")
    parser.add_argument("--gate-rounds", type=int, help="rounds each finalist plays in the gate replay (default 2000)")
    parser.add_argument("--gate-matches", type=int, help="matches each finalist plays in the gate replay (default 400)")
    parser.add_argument("--ladder-rounds", type=int, default=100,
                        help="rounds each new generation plays against every older one, 0 to skip")
    parser.add_argument("--tables", type=int, default=6, help="exhibition games shown at once on the dashboard (default 6)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("--export", action="store_true",
                        help="also write the best net to public/model/ for the game")
    args = parser.parse_args()

    if args.legacy:
        cfg = TrainConfig.legacy()
    elif args.recipe == "genetic":
        cfg = TrainConfig.genetic(quick=args.quick)
    elif args.recipe == "search":
        cfg = TrainConfig.search(quick=args.quick)
    else:
        cfg = TrainConfig.quick() if args.quick else TrainConfig()
    if args.search_rounds is not None:
        cfg.search_rounds = max(1, args.search_rounds)
    if args.search_samples is not None:
        cfg.search_samples = max(1, args.search_samples)
    if args.search_confidence is not None:
        cfg.search_confidence = max(0.0, args.search_confidence)
    if args.population is not None:
        cfg.population = max(2, args.population)
    if args.fitness_rounds is not None:
        cfg.fitness_rounds = max(2, args.fitness_rounds)
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
    if args.finalists is not None:
        cfg.gate_finalists = max(0, args.finalists)
    if args.gate_rounds is not None:
        cfg.gate_rounds = max(2, args.gate_rounds)
    if args.gate_matches is not None:
        cfg.gate_matches = max(0, args.gate_matches)
    cfg.ladder_rounds = max(0, args.ladder_rounds if not args.quick else min(args.ladder_rounds, 30))

    device = resolve_device(args.device)
    run_live(cfg, device, args.port, open_browser=not args.no_browser, export=args.export,
             tables=max(1, min(12, args.tables)))


if __name__ == "__main__":
    main()

"""Train the Yaniv net: clone a card counter, then improve by self-play.

The shipped checkpoint is whichever player won the most matches against the
card-counting bench. A self-play pass replaces that clone only when it wins
more often.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .engine import (
    MAX_PLIES,
    Action,
    View,
    apply_action,
    legal_actions,
    make_view,
    new_round,
    play_match,
    play_round,
    resolve_showdown,
)
from .features import ACTION_DIM, INPUT_DIM, STATE_DIM, action_matrix, state_features
from .heuristic import choose_greedy, choose_heuristic, choose_random
from .backend import Device, as_numpy, build_net, resolve_device
from .network import MLP, save_net

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "public" / "model" / "yaniv-net.json"
REPORT_PATH = ROOT / "public" / "model" / "training-report.json"


class Dataset:
    def __init__(self, states, actions, offsets, labels):
        self.states = states
        self.actions = actions
        self.offsets = offsets
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def net_chooser(net: MLP, temperature: float):
    def choose(view: View, actions: list[Action], rng) -> Action:
        packed_state = np.repeat(state_features(view)[None, :], len(actions), axis=0)
        packed = np.concatenate([packed_state, action_matrix(view, actions)], axis=1)
        scores = net.forward(packed)[:, 0]
        best = int(np.argmax(scores))
        if temperature <= 1e-6 or rng is None:
            return actions[best]
        shifted = scores / np.float32(temperature)
        shifted = shifted - shifted.max()
        weights = np.exp(np.clip(shifted, -40, 40)).astype(np.float64)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            return actions[best]
        roll = rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += float(weight)
            if roll <= cumulative:
                return actions[index]
        return actions[best]

    return choose


def _tick(progress, done: int, total: int) -> None:
    """Report progress about fifty times per job, plus the last step."""
    if progress is not None and (done == total or done % max(1, total // 50) == 0):
        progress(done, total)


def collect_expert(n_rounds: int, seed: int, progress=None) -> list[tuple[View, int]]:
    rng = random.Random(seed)
    rows: list[tuple[View, int]] = []
    for i in range(n_rounds):
        _tick(progress, i + 1, n_rounds)
        scores = [rng.randint(0, 90), rng.randint(0, 90)] if rng.random() < 0.4 else [0, 0]
        state = new_round(scores, i % 2, rng)
        while not state.terminal:
            if state.ply >= MAX_PLIES:
                resolve_showdown(state)
                break
            view = make_view(state, state.turn)
            actions = legal_actions(view)
            if not actions:
                resolve_showdown(state)
                break
            choice = choose_heuristic(view, actions)
            rows.append((view, actions.index(choice)))
            apply_action(state, choice)
    return rows


def materialize(rows: list[tuple[View, int]]) -> Dataset:
    states = np.empty((len(rows), STATE_DIM), dtype=np.float32)
    blocks: list[np.ndarray] = []
    offsets = np.empty(len(rows) + 1, dtype=np.int32)
    labels = np.empty(len(rows), dtype=np.int16)
    offsets[0] = 0
    for index, (view, label) in enumerate(rows):
        actions = legal_actions(view)
        matrix = action_matrix(view, actions)
        states[index] = state_features(view)
        blocks.append(matrix)
        labels[index] = label
        offsets[index + 1] = offsets[index] + matrix.shape[0]
    packed = np.concatenate(blocks) if blocks else np.zeros((0, ACTION_DIM), np.float32)
    return Dataset(states, packed, offsets, labels)


def concat_datasets(parts: list[Dataset]) -> Dataset:
    states = np.concatenate([part.states for part in parts])
    actions = np.concatenate([part.actions for part in parts])
    labels = np.concatenate([part.labels for part in parts])
    offsets = [0]
    for part in parts:
        base = offsets[-1]
        for end in part.offsets[1:]:
            offsets.append(base + int(end))
    return Dataset(states, actions, np.asarray(offsets, dtype=np.int32), labels)


def train_dataset(net, data: Dataset, lr: float, batch_size: int, rng: random.Random, progress=None) -> tuple[float, float]:
    if not isinstance(net, MLP):
        return net.train_epoch(data, lr, batch_size, rng, progress)
    count = len(data)
    order = list(range(count))
    rng.shuffle(order)
    total_ce = 0.0
    total_acc = 0.0
    seen = 0
    input_dim = STATE_DIM + ACTION_DIM
    batches = max(1, (count + batch_size - 1) // batch_size)
    for batch_index, start in enumerate(range(0, count, batch_size)):
        _tick(progress, batch_index + 1, batches)
        ids = order[start : start + batch_size]
        sizes = [int(data.offsets[i + 1] - data.offsets[i]) for i in ids]
        total = sum(sizes)
        if total == 0:
            continue
        packed = np.empty((total, input_dim), dtype=np.float32)
        cursor = 0
        for example, size in zip(ids, sizes):
            begin = int(data.offsets[example])
            packed[cursor : cursor + size, :STATE_DIM] = data.states[example]
            packed[cursor : cursor + size, STATE_DIM:] = data.actions[begin : begin + size]
            cursor += size
        values = net.forward(packed, cache=True)[:, 0]
        grad = np.zeros_like(values)
        ce_sum = 0.0
        acc_sum = 0.0
        cursor = 0
        for example, size in zip(ids, sizes):
            label = int(data.labels[example])
            sl = values[cursor : cursor + size]
            shifted = sl - sl.max()
            exp = np.exp(np.clip(shifted, -60, 60))
            probs = exp / exp.sum()
            ce_sum += float(-np.log(max(float(probs[label]), 1e-8)))
            acc_sum += 1.0 if int(np.argmax(sl)) == label else 0.0
            piece = probs.copy()
            piece[label] -= 1.0
            grad[cursor : cursor + size] = piece
            cursor += size
        net.zero_grad()
        net.backward_acc(grad[:, None])
        net.step(len(ids), lr)
        total_ce += ce_sum
        total_acc += acc_sum
        seen += len(ids)
    if seen == 0:
        return 0.0, 0.0
    return total_ce / seen, total_acc / seen


def agreement(net, data: Dataset, limit: int = 800) -> float:
    net = as_numpy(net)
    step = max(1, len(data) // limit)
    correct = 0
    checked = 0
    for index in range(0, len(data), step):
        begin = int(data.offsets[index])
        end = int(data.offsets[index + 1])
        size = end - begin
        packed = np.empty((size, STATE_DIM + ACTION_DIM), dtype=np.float32)
        packed[:, :STATE_DIM] = data.states[index]
        packed[:, STATE_DIM:] = data.actions[begin:end]
        scores = net.forward(packed)[:, 0]
        correct += int(np.argmax(scores) == int(data.labels[index]))
        checked += 1
    return correct / max(1, checked)


def collect_self_play(net, n_games: int, seed: int, temperature: float, progress=None) -> list[tuple[View, int]]:
    """Decisions from rounds the net won, against itself and against the bench."""
    rng = random.Random(seed)
    policy = net_chooser(as_numpy(net), temperature)
    rows: list[tuple[View, int]] = []
    for i in range(n_games):
        _tick(progress, i + 1, n_games)
        hero = i % 2
        versus_bench = i % 4 == 0
        traces: list[list[tuple[View, int]]] = [[], []]
        choosers = [policy, policy]
        if versus_bench:
            choosers[1 - hero] = choose_heuristic
        state = new_round([0, 0], i % 2, rng)
        while not state.terminal:
            if state.ply >= MAX_PLIES:
                resolve_showdown(state)
                break
            seat = state.turn
            view = make_view(state, seat)
            actions = legal_actions(view)
            if not actions:
                resolve_showdown(state)
                break
            choice = choosers[seat](view, actions, rng)
            if choosers[seat] is policy:
                traces[seat].append((view, actions.index(choice)))
            apply_action(state, choice)
        added = state.result.added
        for seat in (0, 1):
            if choosers[seat] is not policy:
                continue
            if added[seat] < added[1 - seat]:
                rows.extend(traces[seat])
    return rows


def round_duel(hero, villain, n: int, seed: int, progress=None) -> dict:
    rng = random.Random(seed)
    wins = losses = ties = 0
    hero_points = villain_points = 0
    calls = assafs = 0
    for i in range(n):
        _tick(progress, i + 1, n)
        called = {"flag": False}

        def hero_tracked(view, actions, rng_):
            action = hero(view, actions, rng_)
            if action.kind == "yaniv":
                called["flag"] = True
            return action

        if i % 2 == 0:
            state = play_round(hero_tracked, villain, rng, [0, 0], i % 2)
            mine, theirs = state.result.added
            hero_called = called["flag"] and state.result.caller == 0
        else:
            state = play_round(villain, hero_tracked, rng, [0, 0], i % 2)
            theirs, mine = state.result.added
            hero_called = called["flag"] and state.result.caller == 1
        if hero_called:
            calls += 1
            if state.result.assaf:
                assafs += 1
        hero_points += mine
        villain_points += theirs
        if mine < theirs:
            wins += 1
        elif theirs < mine:
            losses += 1
        else:
            ties += 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / n,
        "hero_avg": hero_points / n,
        "villain_avg": villain_points / n,
        "calls": calls,
        "assafs": assafs,
        "assaf_rate": (assafs / calls) if calls else 0.0,
    }


def match_duel(hero, villain, n: int, seed: int, progress=None) -> dict:
    rng = random.Random(seed)
    wins = losses = ties = 0
    for i in range(n):
        _tick(progress, i + 1, n)
        if i % 2 == 0:
            mine, theirs = play_match(hero, villain, rng)
        else:
            theirs, mine = play_match(villain, hero, rng)
        if mine < theirs:
            wins += 1
        elif theirs < mine:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "win_rate": wins / max(1, n)}


def snapshot(net) -> MLP:
    """A NumPy copy of the current weights, whatever device trains them."""
    return as_numpy(net).clone()


def _cards(mask: int) -> list[int]:
    out = []
    while mask:
        bit = mask & -mask
        out.append(bit.bit_length() - 1)
        mask ^= bit
    return out


def _card(card: int) -> str:
    return "A23456789TJQK"[card % 13] + "cdhs"[card // 13]


def print_sample(log=print) -> None:
    rng = random.Random(1)
    state = new_round([0, 0], 0, rng)
    log("Sample round, card counter versus card counter")
    shown = 0
    while not state.terminal and shown < 8:
        view = make_view(state, state.turn)
        actions = legal_actions(view)
        choice = choose_heuristic(view, actions)
        names = " ".join(_card(c) for c in _cards(view.my_hand))
        if choice.kind == "yaniv":
            played = "calls Yaniv"
        elif choice.draw == "none":
            played = "goes out"
        else:
            thrown = " ".join(_card(c) for c in _cards(choice.discard_mask))
            source = "the deck" if choice.draw == "stock" else _card(view.discard[-1])
            played = f"throws {thrown}, draws {source}"
        log(f"  seat {state.turn} [{names}] {played}")
        apply_action(state, choice)
        shown += 1


@dataclass
class TrainConfig:
    expert_rounds: int = 2400
    holdout_rounds: int = 240
    epochs: int = 6
    generations: int = 4  # self-play passes after the imitation clone
    games_per_generation: int = 800
    eval_rounds: int = 180
    eval_matches: int = 48
    bench_rounds: int = 160
    lrs: list[float] = field(default_factory=lambda: [1.2e-3, 8e-4, 5e-4, 3e-4, 2e-4, 1.2e-4])
    fine_tune_lr: float = 8e-5
    batch: int = 72
    temperature: float = 0.32
    ladder_rounds: int = 0  # rounds per pair of generations; 0 skips the ladder

    @classmethod
    def quick(cls) -> "TrainConfig":
        return cls(
            expert_rounds=220,
            holdout_rounds=40,
            epochs=2,
            generations=1,
            games_per_generation=30,
            eval_rounds=30,
            eval_matches=6,
            bench_rounds=40,
            lrs=[1e-3, 4e-4],
            batch=256,
        )

    def lr_for(self, epoch: int) -> float:
        return self.lrs[min(epoch, len(self.lrs)) - 1]


class Observer:
    """Hooks a run reports to. The default prints to the console like the old script."""

    def log(self, message: str) -> None:
        print(message, flush=True)

    def phase(self, name: str, label: str) -> None:
        pass

    def progress(self, name: str, done: int, total: int) -> None:
        pass

    def history(self, entry: dict) -> None:
        pass

    def generation(self, gen: dict, net: MLP) -> None:
        pass

    def ladder(self, row: int, col: int, record: dict) -> None:
        pass

    def done(self, report: dict, paths: dict) -> None:
        pass


def run_training(
    cfg: TrainConfig,
    observer: Observer | None = None,
    device: Device | None = None,
    model_path: Path | None = MODEL_PATH,
    report_path: Path | None = REPORT_PATH,
    run_dir: Path | None = None,
) -> dict:
    """Clone the card counter, then improve by self-play. Returns the report.

    Every generation is handed to the observer and, with `run_dir`, saved as
    gen-NN.json. Only the gated best is written to `model_path`.
    """
    obs = observer or Observer()
    device = device or resolve_device("numpy")
    log = obs.log

    def job(name: str):
        return lambda done, total: obs.progress(name, done, total)

    started = time.time()
    log(f"Training on {device.label()}")
    if device.note:
        log(device.note)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    print_sample(log)
    obs.phase("bench", "Card counter against the greedy player")
    bench = round_duel(choose_heuristic, choose_greedy, cfg.bench_rounds, 100, job("bench"))
    log(
        f"Bench vs greedy {bench['wins']}-{bench['losses']}-{bench['ties']} "
        f"avg {bench['hero_avg']:.2f}-{bench['villain_avg']:.2f} assaf {bench['assaf_rate']:.2f}"
    )

    obs.phase("expert", f"Card counter plays {cfg.expert_rounds:,} rounds to copy")
    log(f"Collecting {cfg.expert_rounds} expert rounds")
    t0 = time.time()
    expert_rows = collect_expert(cfg.expert_rounds, 2024, job("expert"))
    hold_rows = collect_expert(cfg.holdout_rounds, 909)
    log(f"Decisions {len(expert_rows)} collected in {time.time()-t0:.1f}s")
    t0 = time.time()
    expert = materialize(expert_rows)
    holdout = materialize(hold_rows)
    log(f"Features packed in {time.time()-t0:.1f}s, action rows {expert.actions.shape[0]}")

    net = build_net(device, 11)
    history: list[dict] = []
    rng = random.Random(3)
    for epoch in range(1, cfg.epochs + 1):
        lr = cfg.lr_for(epoch)
        obs.phase("imitation", f"Copying the counter, epoch {epoch} of {cfg.epochs}")
        t0 = time.time()
        ce, acc = train_dataset(net, expert, lr, cfg.batch, rng, job("imitation"))
        hold = agreement(net, holdout)
        log(f"Epoch {epoch}/{cfg.epochs} ce {ce:.3f} acc {acc:.3f} holdout {hold:.3f} in {time.time()-t0:.1f}s")
        entry = {"phase": "imitation", "epoch": epoch, "accuracy": acc, "holdout": hold, "ce": ce,
                 "seconds": round(time.time() - t0, 2)}
        history.append(entry)
        obs.history(entry)

    generations: list[MLP] = []

    def evaluate(gen_index: int, label: str, seed_round: int, seed_match: int):
        obs.phase("eval", f"Gen {gen_index} plays the counter: {cfg.eval_rounds} rounds, {cfg.eval_matches} matches")
        policy = net_chooser(snapshot(net), 0.0)
        rounds = round_duel(policy, choose_heuristic, cfg.eval_rounds, seed_round, job("eval_rounds"))
        matches = match_duel(policy, choose_heuristic, cfg.eval_matches, seed_match, job("eval_matches"))
        return rounds, matches

    def publish_generation(index: int, label: str, rounds: dict, matches: dict, kept: bool, extra: dict) -> None:
        current = snapshot(net)
        generations.append(current)
        if run_dir is not None:
            save_net(current, run_dir / f"gen-{index:02d}.json", {"generation": index, "label": label})
        gen = {"index": index, "label": label, "rounds": rounds, "matches": matches, "kept": kept,
               "elapsed": round(time.time() - started, 1), **extra}
        obs.generation(gen, current)
        if cfg.ladder_rounds > 0 and index > 0:
            obs.phase("ladder", f"Gen {index} plays every earlier generation, {cfg.ladder_rounds} rounds each")
            hero = net_chooser(current, 0.0)
            for older in range(index):
                record = round_duel(hero, net_chooser(generations[older], 0.0), cfg.ladder_rounds,
                                    5000 + 31 * index + older, job("ladder"))
                obs.ladder(index, older, record)
                log(f"  ladder gen {index} vs gen {older}: rounds {record['win_rate']:.3f}")

    best = snapshot(net)
    best_round, best_match = evaluate(0, "Copied the counter", 500, 600)
    best_key = (best_match["win_rate"], best_round["win_rate"], -best_round["hero_avg"])
    best_index = 0
    log(
        f"After imitation rounds {best_round['win_rate']:.3f} ({best_round['hero_avg']:.2f} pts, "
        f"assaf {best_round['assaf_rate']:.2f}) matches {best_match['win_rate']:.3f}"
    )
    entry = {"phase": "eval", "iteration": 0, "rounds": best_round, "matches": best_match}
    history.append(entry)
    obs.history(entry)
    publish_generation(0, "Copied the counter", best_round, best_match, True, {})

    self_play_games = 0
    self_play_decisions = 0
    for iteration in range(1, cfg.generations + 1):
        obs.phase("self_play", f"Gen {iteration}: {cfg.games_per_generation} self-play games")
        t0 = time.time()
        fresh_rows = collect_self_play(net, cfg.games_per_generation, 4000 + iteration, cfg.temperature,
                                       job("self_play"))
        self_play_games += cfg.games_per_generation
        self_play_decisions += len(fresh_rows)
        fresh = materialize(fresh_rows) if fresh_rows else None
        log(f"Self-play {iteration}: {len(fresh_rows)} winning decisions in {time.time()-t0:.1f}s")
        mixed = concat_datasets([expert, fresh]) if fresh is not None and len(fresh) else expert
        obs.phase("fine_tune", f"Gen {iteration}: fine-tune on {len(mixed):,} decisions")
        t0 = time.time()
        ce, acc = train_dataset(net, mixed, cfg.fine_tune_lr, cfg.batch, rng, job("fine_tune"))
        tune_seconds = time.time() - t0
        rounds, matches = evaluate(iteration, f"Self-play pass {iteration}", 700 + iteration, 800 + iteration)
        key = (matches["win_rate"], rounds["win_rate"], -rounds["hero_avg"])
        log(
            f"  fine-tune {tune_seconds:.1f}s ce {ce:.3f} rounds {rounds['win_rate']:.3f} "
            f"assaf {rounds['assaf_rate']:.2f} matches {matches['win_rate']:.3f}"
        )
        entry = {
            "phase": "self_play",
            "iteration": iteration,
            "winning_decisions": len(fresh_rows),
            "rounds": rounds,
            "matches": matches,
        }
        history.append(entry)
        obs.history(entry)
        kept = key > best_key
        if kept:
            best_key = key
            best = snapshot(net)
            best_round = rounds
            best_match = matches
            best_index = iteration
            log("  new best checkpoint")
        publish_generation(iteration, f"Self-play pass {iteration}", rounds, matches, kept,
                           {"winning_decisions": len(fresh_rows), "ce": ce})

    obs.phase("final", "Shipped checkpoint against the greedy and random players")
    greedy = round_duel(net_chooser(best, 0.0), choose_greedy, cfg.eval_rounds, 910, job("final"))
    randoms = round_duel(net_chooser(best, 0.0), choose_random, max(40, cfg.eval_rounds // 2), 911)
    report = {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "input_dim": INPUT_DIM,
        "layers": best.dims,
        "expert_rounds": cfg.expert_rounds,
        "expert_decisions": len(expert),
        "self_play_games": self_play_games,
        "self_play_winning_decisions": self_play_decisions,
        "self_play_iterations": cfg.generations,
        "imitation_epochs": cfg.epochs,
        "seconds": round(time.time() - started, 1),
        "device": device.label(),
        "best_generation": best_index,
        "match_win_rate_vs_benchmark": best_match["win_rate"],
        "match_record_vs_benchmark": best_match,
        "round_win_rate_vs_benchmark": best_round["win_rate"],
        "round_record_vs_benchmark": best_round,
        "round_win_rate_vs_greedy": greedy["win_rate"],
        "round_record_vs_greedy": greedy,
        "round_win_rate_vs_random": randoms["win_rate"],
        "benchmark_vs_greedy": bench,
        "history": history,
        "rules": {"yaniv_max": 5, "assaf_penalty": 30, "match_limit": 100, "deal": 5, "max_plies": MAX_PLIES},
    }
    meta = {
        "match_win_rate_vs_benchmark": best_match["win_rate"],
        "self_play_games": self_play_games,
        "expert_decisions": len(expert),
    }
    paths: dict[str, str] = {}
    if run_dir is not None:
        save_net(best, run_dir / "best.json", meta)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths["run_dir"] = str(run_dir)
    if model_path is not None:
        save_net(best, model_path, meta)
        paths["model"] = str(model_path)
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths["report"] = str(report_path)
    log(
        f"Best vs bench matches {best_match['wins']}-{best_match['losses']}-{best_match['ties']} "
        f"rounds {best_round['win_rate']:.3f} vs greedy {greedy['win_rate']:.3f} "
        f"vs random {randoms['win_rate']:.3f}"
    )
    for where in paths.values():
        log(f"Saved {where}")
    obs.done(report, paths)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="numpy", help="numpy (default), auto, cuda, mps or cpu")
    args = parser.parse_args()
    cfg = TrainConfig.quick() if args.quick else TrainConfig()
    run_training(cfg, device=resolve_device(args.device))


if __name__ == "__main__":
    main()

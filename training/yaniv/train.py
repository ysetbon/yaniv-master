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
from .network import MLP, default_net, save_net

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


def collect_expert(n_rounds: int, seed: int) -> list[tuple[View, int]]:
    rng = random.Random(seed)
    rows: list[tuple[View, int]] = []
    for i in range(n_rounds):
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


def train_dataset(net: MLP, data: Dataset, lr: float, batch_size: int, rng: random.Random) -> tuple[float, float]:
    count = len(data)
    order = list(range(count))
    rng.shuffle(order)
    total_ce = 0.0
    total_acc = 0.0
    seen = 0
    input_dim = STATE_DIM + ACTION_DIM
    for start in range(0, count, batch_size):
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


def agreement(net: MLP, data: Dataset, limit: int = 800) -> float:
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


def collect_self_play(net: MLP, n_games: int, seed: int, temperature: float) -> list[tuple[View, int]]:
    """Decisions from rounds the net won, against itself and against the bench."""
    rng = random.Random(seed)
    policy = net_chooser(net, temperature)
    rows: list[tuple[View, int]] = []
    for i in range(n_games):
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


def round_duel(hero, villain, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    wins = losses = ties = 0
    hero_points = villain_points = 0
    calls = assafs = 0
    for i in range(n):
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


def match_duel(hero, villain, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    wins = losses = ties = 0
    for i in range(n):
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


def snapshot(net: MLP) -> MLP:
    copy = MLP(net.dims, seed=0)
    for i in range(len(net.W)):
        copy.W[i] = net.W[i].copy()
        copy.b[i] = net.b[i].copy()
    return copy


def _cards(mask: int) -> list[int]:
    out = []
    while mask:
        bit = mask & -mask
        out.append(bit.bit_length() - 1)
        mask ^= bit
    return out


def _card(card: int) -> str:
    return "A23456789TJQK"[card % 13] + "cdhs"[card // 13]


def print_sample() -> None:
    rng = random.Random(1)
    state = new_round([0, 0], 0, rng)
    print("Sample round, card counter versus card counter", flush=True)
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
        print(f"  seat {state.turn} [{names}] {played}", flush=True)
        apply_action(state, choice)
        shown += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        expert_rounds, epochs = 220, 2
        self_iters, self_games = 1, 30
        eval_rounds, eval_matches = 30, 6
        lrs = [1e-3, 4e-4]
        batch = 256
    else:
        expert_rounds, epochs = 2400, 6
        self_iters, self_games = 4, 800
        eval_rounds, eval_matches = 180, 48
        lrs = [1.2e-3, 8e-4, 5e-4, 3e-4, 2e-4, 1.2e-4]
        batch = 72

    started = time.time()
    print_sample()
    bench = round_duel(choose_heuristic, choose_greedy, 160 if not args.quick else 40, 100)
    print(
        f"Bench vs greedy {bench['wins']}-{bench['losses']}-{bench['ties']} "
        f"avg {bench['hero_avg']:.2f}-{bench['villain_avg']:.2f} assaf {bench['assaf_rate']:.2f}",
        flush=True,
    )
    print(f"Collecting {expert_rounds} expert rounds", flush=True)
    t0 = time.time()
    expert_rows = collect_expert(expert_rounds, 2024)
    hold_rows = collect_expert(240 if not args.quick else 40, 909)
    print(f"Decisions {len(expert_rows)} collected in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    expert = materialize(expert_rows)
    holdout = materialize(hold_rows)
    print(f"Features packed in {time.time()-t0:.1f}s, action rows {expert.actions.shape[0]}", flush=True)

    net = default_net(11)
    history = []
    rng = random.Random(3)
    for epoch, lr in enumerate(lrs[:epochs], start=1):
        t0 = time.time()
        ce, acc = train_dataset(net, expert, lr, batch, rng)
        hold = agreement(net, holdout)
        print(
            f"Epoch {epoch}/{epochs} ce {ce:.3f} acc {acc:.3f} holdout {hold:.3f} in {time.time()-t0:.1f}s",
            flush=True,
        )
        history.append({"phase": "imitation", "epoch": epoch, "accuracy": acc, "holdout": hold, "ce": ce})

    best = snapshot(net)
    best_round = round_duel(net_chooser(best, 0.0), choose_heuristic, eval_rounds, 500)
    best_match = match_duel(net_chooser(best, 0.0), choose_heuristic, eval_matches, 600)
    best_key = (best_match["win_rate"], best_round["win_rate"], -best_round["hero_avg"])
    print(
        f"After imitation rounds {best_round['win_rate']:.3f} ({best_round['hero_avg']:.2f} pts, "
        f"assaf {best_round['assaf_rate']:.2f}) matches {best_match['win_rate']:.3f}",
        flush=True,
    )
    history.append({"phase": "eval", "iteration": 0, "rounds": best_round, "matches": best_match})

    self_play_games = 0
    self_play_decisions = 0
    for iteration in range(1, self_iters + 1):
        t0 = time.time()
        fresh_rows = collect_self_play(net, self_games, 4000 + iteration, 0.32)
        self_play_games += self_games
        self_play_decisions += len(fresh_rows)
        fresh = materialize(fresh_rows) if fresh_rows else None
        print(
            f"Self-play {iteration}: {len(fresh_rows)} winning decisions in {time.time()-t0:.1f}s",
            flush=True,
        )
        mixed = concat_datasets([expert, fresh]) if fresh is not None and len(fresh) else expert
        t0 = time.time()
        ce, acc = train_dataset(net, mixed, 8e-5, batch, rng)
        rounds = round_duel(net_chooser(net, 0.0), choose_heuristic, eval_rounds, 700 + iteration)
        matches = match_duel(net_chooser(net, 0.0), choose_heuristic, eval_matches, 800 + iteration)
        key = (matches["win_rate"], rounds["win_rate"], -rounds["hero_avg"])
        print(
            f"  fine-tune {time.time()-t0:.1f}s ce {ce:.3f} rounds {rounds['win_rate']:.3f} "
            f"assaf {rounds['assaf_rate']:.2f} matches {matches['win_rate']:.3f}",
            flush=True,
        )
        history.append(
            {
                "phase": "self_play",
                "iteration": iteration,
                "winning_decisions": len(fresh_rows),
                "rounds": rounds,
                "matches": matches,
            }
        )
        if key > best_key:
            best_key = key
            best = snapshot(net)
            best_round = rounds
            best_match = matches
            print("  new best checkpoint", flush=True)

    greedy = round_duel(net_chooser(best, 0.0), choose_greedy, eval_rounds, 910)
    randoms = round_duel(net_chooser(best, 0.0), choose_random, max(40, eval_rounds // 2), 911)
    report = {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "input_dim": INPUT_DIM,
        "layers": best.dims,
        "expert_rounds": expert_rounds,
        "expert_decisions": len(expert),
        "self_play_games": self_play_games,
        "self_play_winning_decisions": self_play_decisions,
        "self_play_iterations": self_iters,
        "imitation_epochs": epochs,
        "seconds": round(time.time() - started, 1),
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
    save_net(
        best,
        MODEL_PATH,
        {
            "match_win_rate_vs_benchmark": best_match["win_rate"],
            "self_play_games": self_play_games,
            "expert_decisions": len(expert),
        },
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Best vs bench matches {best_match['wins']}-{best_match['losses']}-{best_match['ties']} "
        f"rounds {best_round['win_rate']:.3f} vs greedy {greedy['win_rate']:.3f} "
        f"vs random {randoms['win_rate']:.3f}",
        flush=True,
    )
    print(f"Saved {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()

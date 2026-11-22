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
import dataclasses
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
from .network import MLP, load_net, save_net

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "public" / "model" / "yaniv-net.json"
REPORT_PATH = ROOT / "public" / "model" / "training-report.json"


class Dataset:
    """Decisions packed for training. `weights` scales each decision's loss."""

    def __init__(self, states, actions, offsets, labels, weights=None):
        self.states = states
        self.actions = actions
        self.offsets = offsets
        self.labels = labels
        self.weights = np.ones(len(labels), dtype=np.float32) if weights is None else weights

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, states=self.states, actions=self.actions, offsets=self.offsets,
                 labels=self.labels, weights=self.weights)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Dataset":
        with np.load(path) as z:
            return cls(z["states"], z["actions"], z["offsets"], z["labels"], z["weights"])


def subset(data: Dataset, ids) -> Dataset:
    """The decisions at `ids`, with their legal-move rows."""
    ids = np.asarray(ids, dtype=np.int64)
    offsets = data.offsets.astype(np.int64)
    begins = offsets[ids]
    sizes = offsets[ids + 1] - begins
    rows = np.repeat(begins, sizes) + (np.arange(int(sizes.sum())) - np.repeat(np.cumsum(sizes) - sizes, sizes))
    new_offsets = np.zeros(len(ids) + 1, dtype=np.int32)
    new_offsets[1:] = np.cumsum(sizes)
    return Dataset(data.states[ids], data.actions[rows], new_offsets, data.labels[ids], data.weights[ids])


CACHE_DIR = ROOT / "runs" / ".cache"


def _rules_signature() -> str:
    """Changes whenever the rules, features or the card counter change."""
    import hashlib

    digest = hashlib.sha1()
    here = Path(__file__).resolve().parent
    for name in ("engine.py", "features.py", "heuristic.py"):
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:12]


def expert_dataset(n_rounds: int, seed: int, progress=None, log=print, cache: bool = True) -> Dataset:
    """The card counter's decisions, cached on disk because they never change."""
    path = CACHE_DIR / f"expert-{n_rounds}-{seed}-{_rules_signature()}.npz"
    if cache and path.exists():
        try:
            data = Dataset.load(path)
            log(f"Loaded {len(data)} card-counter decisions from {path.name}")
            if progress is not None:
                progress(n_rounds, n_rounds)
            return data
        except (OSError, ValueError, KeyError):
            pass  # a damaged cache file is rebuilt below
    t0 = time.time()
    rows = collect_expert(n_rounds, seed, progress)
    data = materialize(rows)
    log(f"Collected and packed {len(data)} card-counter decisions in {time.time()-t0:.1f}s")
    if cache:
        try:
            data.save(path)
        except OSError:
            pass
    return data


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


def materialize(rows: list) -> Dataset:
    """Rows are (view, label) or (view, label, weight)."""
    states = np.empty((len(rows), STATE_DIM), dtype=np.float32)
    blocks: list[np.ndarray] = []
    offsets = np.empty(len(rows) + 1, dtype=np.int32)
    labels = np.empty(len(rows), dtype=np.int16)
    weights = np.ones(len(rows), dtype=np.float32)
    offsets[0] = 0
    for index, row in enumerate(rows):
        view, label = row[0], row[1]
        if len(row) > 2:
            weights[index] = row[2]
        actions = legal_actions(view)
        matrix = action_matrix(view, actions)
        states[index] = state_features(view)
        blocks.append(matrix)
        labels[index] = label
        offsets[index + 1] = offsets[index] + matrix.shape[0]
    packed = np.concatenate(blocks) if blocks else np.zeros((0, ACTION_DIM), np.float32)
    return Dataset(states, packed, offsets, labels, weights)


def concat_datasets(parts: list[Dataset]) -> Dataset:
    states = np.concatenate([part.states for part in parts])
    actions = np.concatenate([part.actions for part in parts])
    labels = np.concatenate([part.labels for part in parts])
    weights = np.concatenate([part.weights for part in parts])
    pieces = [np.zeros(1, dtype=np.int64)]
    base = 0
    for part in parts:
        pieces.append(part.offsets[1:].astype(np.int64) + base)
        base += int(part.offsets[-1])
    return Dataset(states, actions, np.concatenate(pieces).astype(np.int32), labels, weights)


def train_dataset(net, data: Dataset, lr: float, batch_size: int, rng: random.Random, progress=None) -> tuple[float, float]:
    if not isinstance(net, MLP):
        return net.train_epoch(data, lr, batch_size, rng, progress)
    count = len(data)
    order = list(range(count))
    rng.shuffle(order)
    total_ce = 0.0
    total_acc = 0.0
    total_weight = 0.0
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
        weight_sum = 0.0
        cursor = 0
        for example, size in zip(ids, sizes):
            label = int(data.labels[example])
            weight = float(data.weights[example])
            weight_sum += weight
            sl = values[cursor : cursor + size]
            shifted = sl - sl.max()
            exp = np.exp(np.clip(shifted, -60, 60))
            probs = exp / exp.sum()
            ce_sum += weight * float(-np.log(max(float(probs[label]), 1e-8)))
            acc_sum += 1.0 if int(np.argmax(sl)) == label else 0.0
            piece = probs.copy()
            piece[label] -= 1.0
            grad[cursor : cursor + size] = piece * weight
            cursor += size
        net.zero_grad()
        net.backward_acc(grad[:, None])
        net.step(weight_sum, lr)
        total_ce += ce_sum
        total_acc += acc_sum
        total_weight += weight_sum
        seen += len(ids)
    if seen == 0:
        return 0.0, 0.0
    return total_ce / max(total_weight, 1e-8), total_acc / seen


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


def margin_weight(margin: int) -> float:
    """Loss weight for a won round: 0.5 for a one-point win, 2.0 from a 30-point win up."""
    return 0.5 + min(max(margin, 0), 30) / 20.0


def collect_self_play(net, n_games: int, seed: int, temperature: float, progress=None,
                      weighted: bool = False, offset: int = 0, match_scores: bool = False) -> list:
    """Decisions from rounds the net won, against itself and against the bench.

    With `weighted`, each row carries a loss weight from the points margin.
    With `match_scores`, 40% of rounds start from random match scores, as the
    copied rounds do, so the net keeps seeing positions near the 100 limit.
    `offset` keeps the game pattern (who leads, who faces the bench) when a
    batch of games is split across worker processes.
    """
    rng = random.Random(seed)
    policy = net_chooser(as_numpy(net), temperature)
    rows: list = []
    for done, i in enumerate(range(offset, offset + n_games), start=1):
        _tick(progress, done, n_games)
        hero = i % 2
        versus_bench = i % 4 == 0
        traces: list[list[tuple[View, int]]] = [[], []]
        choosers = [policy, policy]
        if versus_bench:
            choosers[1 - hero] = choose_heuristic
        scores = [rng.randint(0, 90), rng.randint(0, 90)] if match_scores and rng.random() < 0.4 else [0, 0]
        state = new_round(scores, i % 2, rng)
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
                if weighted:
                    weight = margin_weight(added[1 - seat] - added[seat])
                    rows.extend((view, label, weight) for view, label in traces[seat])
                else:
                    rows.extend(traces[seat])
    return rows


def round_duel(hero, villain, n: int, seed: int, progress=None) -> dict:
    """`n` rounds of `hero` against `villain`, played as mirrored pairs.

    Each deal is played twice: once with the hero in seat 0, then with the
    seats swapped, so each player holds both hands and the same player leads
    from the same hand. Card luck cancels within a pair, which makes two
    networks on the same seed far easier to tell apart. Pairs alternate which
    seat leads. An odd `n` ends with one unpaired round.
    """
    rng = random.Random(seed)
    wins = losses = ties = 0
    hero_points = villain_points = 0
    calls = assafs = 0
    deal_seed = 0
    for i in range(n):
        _tick(progress, i + 1, n)
        called = {"flag": False}

        def hero_tracked(view, actions, rng_):
            action = hero(view, actions, rng_)
            if action.kind == "yaniv":
                called["flag"] = True
            return action

        pair, second = divmod(i, 2)
        if not second:
            deal_seed = rng.getrandbits(64)
        deal = random.Random(deal_seed)
        leader = pair % 2
        if not second:
            state = play_round(hero_tracked, villain, deal, [0, 0], leader)
            mine, theirs = state.result.added
            hero_called = called["flag"] and state.result.caller == 0
        else:
            state = play_round(villain, hero_tracked, deal, [0, 0], leader)
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
    """`n` matches, played as mirrored pairs like `round_duel`.

    Both matches of a pair see the same sequence of shuffled decks, with the
    seats swapped, so the hero plays every hand the villain held.
    """
    rng = random.Random(seed)
    wins = losses = ties = 0
    match_seed = 0
    for i in range(n):
        _tick(progress, i + 1, n)
        if i % 2 == 0:
            match_seed = rng.getrandbits(64)
            mine, theirs = play_match(hero, villain, random.Random(match_seed))
        else:
            theirs, mine = play_match(villain, hero, random.Random(match_seed))
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
    if card >= 52:
        return "Jk"
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
        else:
            thrown = " ".join(_card(c) for c in _cards(choice.discard_mask))
            source = "the deck" if choice.draw == "stock" else _card(view.discard[-1])
            played = f"throws {thrown}, draws {source}"
        log(f"  seat {state.turn} [{names}] {played}")
        apply_action(state, choice)
        shown += 1


def gate_key(rounds: dict, matches: dict, metric: str = "rounds") -> tuple:
    """How the gate ranks two test results.

    "rounds": rounds won, then matches won, then fewer points. Rounds come
    first because there are many more of them: 400 mirrored rounds pin a net
    down to about 2 points, 200 matches only to about 3.5.
    "margin": the points margin per round (the counter's points minus the
    net's), then matches, then rounds. Matches are decided by total points, so
    this is what the search recipe optimizes, and it is measured on as many
    rounds.
    """
    if metric == "margin":
        return (rounds["villain_avg"] - rounds["hero_avg"], matches["win_rate"], rounds["win_rate"])
    return (rounds["win_rate"], matches["win_rate"], -rounds["hero_avg"])


@dataclass
class TrainConfig:
    expert_rounds: int = 2400
    holdout_rounds: int = 240
    epochs: int = 6
    generations: int = 8  # self-play passes after the imitation clone
    games_per_generation: int = 800
    eval_rounds: int = 400
    eval_matches: int = 200
    bench_rounds: int = 160
    lrs: list[float] = field(default_factory=lambda: [1.2e-3, 8e-4, 5e-4, 3e-4, 2e-4, 1.2e-4])
    batch: int = 72
    temperature: float = 0.32
    fine_tune_lr: float = 1.5e-4
    fine_tune_epochs: int = 2
    # Self-play wins kept across passes, oldest dropped first. 0 keeps only
    # the current pass, which is how the first version trained.
    replay_capacity: int = 60000
    # Copied decisions mixed in per self-play decision, sliding from start to
    # end over the passes, so the net leans on the counter less as it improves.
    expert_mix_start: float = 2.0
    expert_mix_end: float = 0.5
    weighted: bool = True  # weight won rounds by their points margin
    match_scores: bool = True  # start 40% of self-play rounds from random match scores
    revert_on_reject: bool = False  # restart a pass from the best net when it fails the gate
    # The gate ranks generations by rounds won on their own test (see
    # gate_key). One test per generation is noisy, so when training ends the
    # top `gate_finalists` generations replay one shared, larger set of games
    # and the best of them is the gate's pick. 0 finalists skips the replay.
    gate_metric: str = "rounds"  # "rounds" or "margin", see gate_key
    gate_finalists: int = 4
    gate_rounds: int = 2000
    gate_matches: int = 400
    # Picking the best of several noisy tests still flatters the winner. A
    # final test on new games re-measures the pick and Gen 0; the better one
    # ships, with these honest numbers. 0 skips it.
    final_rounds: int = 1000
    final_matches: int = 400
    ladder_rounds: int = 0  # rounds per pair of generations; 0 skips the ladder
    workers: int = 0  # processes for playing games; 0 picks from the CPU count
    effort: float = 0.0  # share of the logical cores to play games on (0 to 1); 0 means all but two
    cache: bool = True  # keep the card counter's decisions on disk between runs
    # "selfplay" fine-tunes one net on its own wins. "genetic" evolves a
    # population of nets that all start from the copy of the counter.
    recipe: str = "selfplay"
    population: int = 24
    # Rounds per member per generation, the same mirrored deals for everyone.
    # Two mutants differ by under a point of rounds won; at 200 rounds the
    # noise in their difference is about 2 points, at 1000 about 0.9.
    fitness_rounds: int = 1000
    elites: int = 2  # best members copied unchanged into the next generation
    tournament: int = 3
    crossover_rate: float = 0.5
    sigma: float = 0.2  # mutation size, relative to each layer's weight spread (0.2 changes ~3% of moves)
    sigma_decay: float = 0.96  # sigma shrinks by this much every generation
    ga_seed: int = 17
    # "search": the best net plays the counter, a search labels every decision
    # (see search.py), and the net learns the searched moves. Each pass's
    # better net plays the next pass's playouts, so the teacher improves too.
    search_rounds: int = 150  # rounds labelled per pass, about 20 decisions each
    search_samples: int = 48  # imagined deals per searched decision
    search_width: int = 4  # plays searched per decision, best first by the net, plus Yaniv
    # A searched move replaces the net's own only when it wins on the same
    # imagined deals by this many standard errors. Without it the search
    # switches on luck between close moves and loses to the counter.
    search_confidence: float = 1.5
    search_lr: float = 3e-4
    search_epochs: int = 3
    search_capacity: int = 60000  # searched decisions kept across passes, oldest dropped first
    # "latest": each pass's net teaches the next pass, kept by the gate or not.
    # "best": only a pass that passes the gate becomes the teacher. One pass's
    # test is noisy, so "best" tends to keep a lucky pass and stall.
    search_teacher: str = "latest"
    # A saved net (e.g. public/model/yaniv-net.json or a run's best.json) to
    # start from instead of copying the counter. It becomes Gen 0, so the
    # final check only ships a later generation that beats it.
    start_from: str = ""

    @classmethod
    def search(cls, quick: bool = False) -> "TrainConfig":
        if quick:
            return dataclasses.replace(cls.quick(), recipe="search", generations=2, gate_metric="margin",
                                       search_rounds=6, search_samples=8, gate_finalists=2,
                                       expert_mix_start=1.0, expert_mix_end=0.25)
        # 1,000 test rounds a pass instead of 400: nets a fraction of a point
        # apart are told apart about 1.6 times more sharply.
        return cls(recipe="search", generations=8, gate_metric="margin", expert_mix_start=1.0, expert_mix_end=0.25,
                   eval_rounds=1000)

    @classmethod
    def genetic(cls, quick: bool = False) -> "TrainConfig":
        if quick:
            return dataclasses.replace(cls.quick(), recipe="genetic", generations=3, population=6, fitness_rounds=20,
                                       gate_finalists=2)
        return cls(recipe="genetic", generations=20)

    @classmethod
    def quick(cls) -> "TrainConfig":
        return cls(
            expert_rounds=220,
            holdout_rounds=40,
            epochs=2,
            generations=2,
            games_per_generation=40,
            eval_rounds=30,
            eval_matches=6,
            bench_rounds=40,
            lrs=[1e-3, 4e-4],
            batch=256,
            replay_capacity=5000,
            gate_rounds=60,
            gate_matches=8,
            final_rounds=40,
            final_matches=8,
        )

    @classmethod
    def legacy(cls) -> "TrainConfig":
        """The first training recipe: one pass of fresh wins over all copied data, a small test."""
        return cls(
            generations=4,
            eval_rounds=180,
            eval_matches=48,
            fine_tune_lr=8e-5,
            fine_tune_epochs=1,
            replay_capacity=0,
            weighted=False,
            match_scores=False,
            gate_finalists=0,
            final_rounds=0,
            final_matches=0,
            workers=1,
        )

    def lr_for(self, epoch: int) -> float:
        return self.lrs[min(epoch, len(self.lrs)) - 1]

    def expert_mix(self, iteration: int) -> float:
        if self.generations <= 1:
            return self.expert_mix_start
        t = (iteration - 1) / (self.generations - 1)
        return self.expert_mix_start + (self.expert_mix_end - self.expert_mix_start) * t


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

    def population(self, summary: dict, nets: list[MLP]) -> None:
        """Genetic recipe only: one generation's members, best first, with their nets."""
        pass

    def should_stop(self) -> bool:
        """Checked before each self-play pass. True ends training early, keeping the best so far."""
        return False

    def done(self, report: dict, paths: dict) -> None:
        pass


def _restart_from(net, best: MLP, device: Device):
    """A training net carrying the best weights. Adam's moments start fresh."""
    if isinstance(net, MLP):
        return best.clone()
    from .torch_network import TorchMLP

    return TorchMLP(best, net.device)


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
    from .parallel import Workers, bot_spec, default_workers, net_spec

    obs = observer or Observer()
    device = device or resolve_device("numpy")
    log = obs.log
    workers = Workers(cfg.workers or default_workers(cfg.effort or None))

    def job(name: str):
        return lambda done, total: obs.progress(name, done, total)

    started = time.time()
    log(f"Training on {device.label()}, playing games on {workers.count} CPU process{'es' if workers.count > 1 else ''}")
    if device.note:
        log(device.note)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "device": device.label()}, indent=2), encoding="utf-8")

    try:
        print_sample(log)
        counter, greedy = bot_spec("counter"), bot_spec("greedy")
        obs.phase("bench", "Card counter against the greedy player")
        bench = workers.rounds(counter, greedy, cfg.bench_rounds, 100, job("bench"))
        log(
            f"Bench vs greedy {bench['wins']}-{bench['losses']}-{bench['ties']} "
            f"avg {bench['hero_avg']:.2f}-{bench['villain_avg']:.2f} assaf {bench['assaf_rate']:.2f}"
        )

        obs.phase("expert", f"Card counter plays {cfg.expert_rounds:,} rounds to copy")
        log(f"Collecting {cfg.expert_rounds} expert rounds")
        expert = expert_dataset(cfg.expert_rounds, 2024, job("expert"), log, cfg.cache)
        holdout = expert_dataset(cfg.holdout_rounds, 909, None, log, cfg.cache)
        log(f"Expert decisions {len(expert)}, action rows {expert.actions.shape[0]}")

        net = build_net(device, 11)
        history: list[dict] = []
        rng = random.Random(3)
        start_name = Path(cfg.start_from).name if cfg.start_from else ""
        if cfg.start_from:
            start = load_net(Path(cfg.start_from))
            if start.dims[0] != INPUT_DIM:
                raise ValueError(f"{cfg.start_from} takes {start.dims[0]} inputs, but these rules give {INPUT_DIM}: "
                                 "it was trained on other rules")
            net = _restart_from(net, start, device)
            log(f"Starting from {cfg.start_from} ({' -> '.join(map(str, start.dims))}), so the copying step is skipped")
        for epoch in range(1, (0 if cfg.start_from else cfg.epochs) + 1):
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

        def evaluate(gen_index: int, seed_round: int, seed_match: int, candidate: MLP | None = None):
            obs.phase("eval", f"Gen {gen_index} plays the counter: {cfg.eval_rounds} rounds, {cfg.eval_matches} matches")
            spec = net_spec(candidate if candidate is not None else snapshot(net))
            rounds = workers.rounds(spec, counter, cfg.eval_rounds, seed_round, job("eval_rounds"))
            matches = workers.matches(spec, counter, cfg.eval_matches, seed_match, job("eval_matches"))
            return rounds, matches

        def publish_generation(index: int, label: str, rounds: dict, matches: dict, kept: bool, extra: dict,
                               candidate: MLP | None = None) -> None:
            current = candidate.clone() if candidate is not None else snapshot(net)
            generations.append(current)
            if run_dir is not None:
                save_net(current, run_dir / f"gen-{index:02d}.json", {"generation": index, "label": label})
            gen = {"index": index, "label": label, "rounds": rounds, "matches": matches, "kept": kept,
                   "elapsed": round(time.time() - started, 1), **extra}
            obs.generation(gen, current)
            if cfg.ladder_rounds > 0 and index > 0:
                obs.phase("ladder", f"Gen {index} plays every earlier generation, {cfg.ladder_rounds} rounds each")
                spec = net_spec(current)
                for older in range(index):
                    record = workers.rounds(spec, net_spec(generations[older]), cfg.ladder_rounds,
                                            5000 + 31 * index + older, job("ladder"))
                    obs.ladder(index, older, record)
                    log(f"  ladder gen {index} vs gen {older}: rounds {record['win_rate']:.3f}")

        t_gen = time.time()
        best = snapshot(net)
        best_round, best_match = evaluate(0, 500, 600)
        best_key = gate_key(best_round, best_match, cfg.gate_metric)
        tested: dict[int, tuple[dict, dict]] = {0: (best_round, best_match)}
        best_index = 0
        log(
            f"{'Starting net' if cfg.start_from else 'After imitation'}: rounds {best_round['win_rate']:.3f} ({best_round['hero_avg']:.2f} pts, "
            f"assaf {best_round['assaf_rate']:.2f}) matches {best_match['win_rate']:.3f}"
        )
        entry = {"phase": "eval", "iteration": 0, "rounds": best_round, "matches": best_match}
        history.append(entry)
        obs.history(entry)
        publish_generation(0, f"Started from {start_name}" if start_name else "Copied the counter", best_round, best_match, True,
                           {"seconds": round(time.time() - t_gen, 1)})

        self_play_games = 0
        self_play_decisions = 0
        passes_run = 0
        population_history: list[dict] = []
        if cfg.recipe == "genetic":
            from .genetic import breed, fitness, seed_population

            ga_rng = np.random.default_rng(cfg.ga_seed)
            pick = random.Random(cfg.ga_seed)
            sigma = cfg.sigma
            population = seed_population(generations[0], cfg.population, sigma, ga_rng)
            for gen_i in range(1, cfg.generations + 1):
                if obs.should_stop():
                    log(f"Stopping early after generation {gen_i - 1}, as asked")
                    break
                t_gen = time.time()
                obs.phase("evolve", f"Gen {gen_i}: {len(population)} networks play the counter, {cfg.fitness_rounds} rounds each on the same deals")
                records = workers.rounds_many([net_spec(m.net) for m in population], counter, cfg.fitness_rounds,
                                              9000 + gen_i, job("evolve"))
                for member, record in zip(population, records):
                    member.record = record
                    member.fitness = fitness(record)
                ranked = sorted(population, key=lambda m: -m.fitness)
                rates = [m.record["win_rate"] for m in ranked]
                summary = {
                    "gen": gen_i, "sigma": sigma, "rounds": cfg.fitness_rounds,
                    "best": rates[0], "median": float(np.median(rates)), "worst": rates[-1],
                    "members": [m.info() for m in ranked],
                }
                population_history.append(summary)
                self_play_games += len(population) * cfg.fitness_rounds
                obs.population(summary, [m.net for m in ranked])
                champion = ranked[0]
                ops = {op: sum(1 for m in ranked if m.op == op) for op in ("copy", "elite", "cross", "mutant")}
                log(f"Gen {gen_i}: rounds won best {rates[0]:.3f} median {summary['median']:.3f} worst {rates[-1]:.3f}, "
                    f"sigma {sigma:.4f}, champion {champion.id} ({champion.op})")
                rounds, matches = evaluate(gen_i, 700 + gen_i, 800 + gen_i, champion.net)
                key = gate_key(rounds, matches, cfg.gate_metric)
                tested[gen_i] = (rounds, matches)
                log(f"  champion {champion.id} rounds {rounds['win_rate']:.3f} assaf {rounds['assaf_rate']:.2f} "
                    f"matches {matches['win_rate']:.3f}")
                entry = {"phase": "genetic", "iteration": gen_i, "best": rates[0], "median": summary["median"],
                         "worst": rates[-1], "sigma": sigma, "rounds": rounds, "matches": matches}
                history.append(entry)
                obs.history(entry)
                kept = key > best_key
                if kept:
                    best_key = key
                    best = champion.net.clone()
                    best_round, best_match, best_index = rounds, matches, gen_i
                    log("  new best checkpoint")
                publish_generation(gen_i, f"Champion of generation {gen_i}", rounds, matches, kept,
                                   {"member": champion.id, "op": champion.op, "parents": champion.parents,
                                    "fitness_win_rate": rates[0], "median_win_rate": summary["median"],
                                    "worst_win_rate": rates[-1], "sigma": sigma, "ops": ops,
                                    "seconds": round(time.time() - t_gen, 1)}, champion.net)
                passes_run = gen_i
                if gen_i < cfg.generations:
                    population = breed(ranked, cfg.population, cfg.elites, cfg.tournament, cfg.crossover_rate,
                                       sigma * cfg.sigma_decay, gen_i + 1, ga_rng, pick)
                sigma *= cfg.sigma_decay
        elif cfg.recipe == "search":
            searched: list[Dataset] = []
            for iteration in range(1, cfg.generations + 1):
                if obs.should_stop():
                    log(f"Stopping early after generation {iteration - 1}, as asked")
                    break
                t_gen = time.time()
                latest = cfg.search_teacher == "latest"
                teacher, teacher_index = (snapshot(net), iteration - 1) if latest else (best, best_index)
                obs.phase("search", f"Gen {iteration}: searching every decision of {cfg.search_rounds} rounds, "
                                    f"{cfg.search_samples} imagined deals each, with Gen {teacher_index} as the teacher")
                rows = workers.search_label(teacher, cfg.search_rounds, 6000 + iteration, cfg.search_samples,
                                            cfg.search_width, job("search"), cfg.match_scores, cfg.search_confidence)
                self_play_games += cfg.search_rounds
                self_play_decisions += len(rows)
                if rows:
                    searched.append(materialize(rows))
                while len(searched) > 1 and sum(len(d) for d in searched) - len(searched[0]) >= cfg.search_capacity:
                    searched.pop(0)
                taught = concat_datasets(searched) if searched else None
                taught_count = len(taught) if taught is not None else 0
                agree = agreement(teacher, materialize(rows)) if rows else float("nan")
                log(f"Search {iteration}: {len(rows)} decisions in {time.time()-t_gen:.1f}s; "
                    f"the teacher's own net already picks the searched move {agree:.1%} of the time")
                take = min(len(expert), int(round(cfg.expert_mix(iteration) * taught_count)))
                picked = sorted(rng.sample(range(len(expert)), take)) if take < len(expert) else range(len(expert))
                parts = ([subset(expert, list(picked))] if take else []) + ([taught] if taught is not None else [])
                mixed = concat_datasets(parts) if parts else expert

                obs.phase("fine_tune", f"Gen {iteration}: learn {taught_count:,} searched + {take:,} copied decisions")
                t0 = time.time()
                ce = float("nan")
                for _ in range(cfg.search_epochs):
                    ce, acc = train_dataset(net, mixed, cfg.search_lr, cfg.batch, rng, job("fine_tune"))
                tune_seconds = time.time() - t0
                rounds, matches = evaluate(iteration, 700 + iteration, 800 + iteration)
                key = gate_key(rounds, matches, cfg.gate_metric)
                tested[iteration] = (rounds, matches)
                margin = rounds["villain_avg"] - rounds["hero_avg"]
                log(f"  learned in {tune_seconds:.1f}s ce {ce:.3f}: margin {margin:+.2f} pts a round, "
                    f"rounds {rounds['win_rate']:.3f} matches {matches['win_rate']:.3f} assaf {rounds['assaf_rate']:.2f}")
                entry = {"phase": "search", "iteration": iteration, "searched": len(rows), "teacher_agreement": agree,
                         "rounds": rounds, "matches": matches}
                history.append(entry)
                obs.history(entry)
                kept = key > best_key
                if kept:
                    best_key = key
                    best = snapshot(net)
                    best_round, best_match, best_index = rounds, matches, iteration
                    log("  new best checkpoint" + ("" if cfg.search_teacher == "latest" else ", and the next teacher"))
                publish_generation(iteration, f"Search pass {iteration}", rounds, matches, kept,
                                   {"winning_decisions": len(rows), "ce": ce, "copied": take, "replay": taught_count,
                                    "teacher_agreement": agree, "seconds": round(time.time() - t_gen, 1)})
                passes_run = iteration
        else:
            replay: list[Dataset] = []
            for iteration in range(1, cfg.generations + 1):
                if obs.should_stop():
                    log(f"Stopping early after generation {iteration - 1}, as asked")
                    break
                t_gen = time.time()
                obs.phase("self_play", f"Gen {iteration}: {cfg.games_per_generation} self-play games")
                fresh_rows = workers.self_play(snapshot(net), cfg.games_per_generation, 4000 + iteration,
                                               cfg.temperature, cfg.weighted, job("self_play"), cfg.match_scores)
                self_play_games += cfg.games_per_generation
                self_play_decisions += len(fresh_rows)
                fresh = materialize(fresh_rows) if fresh_rows else None
                log(f"Self-play {iteration}: {len(fresh_rows)} winning decisions in {time.time()-t_gen:.1f}s")

                if cfg.replay_capacity > 0:
                    if fresh is not None and len(fresh):
                        replay.append(fresh)
                    while len(replay) > 1 and sum(len(d) for d in replay) - len(replay[0]) >= cfg.replay_capacity:
                        replay.pop(0)
                    won = concat_datasets(replay) if replay else None
                    won_count = len(won) if won is not None else 0
                    take = min(len(expert), int(round(cfg.expert_mix(iteration) * won_count))) if won_count else len(expert)
                    picked = sorted(rng.sample(range(len(expert)), take)) if take < len(expert) else range(len(expert))
                    parts = [subset(expert, list(picked))] + ([won] if won is not None else [])
                    mixed = concat_datasets(parts)
                else:
                    won_count = len(fresh) if fresh is not None else 0
                    take = len(expert)
                    mixed = concat_datasets([expert, fresh]) if fresh is not None and len(fresh) else expert

                obs.phase("fine_tune", f"Gen {iteration}: fine-tune on {take:,} copied + {won_count:,} self-play decisions")
                t0 = time.time()
                ce = float("nan")
                for _ in range(cfg.fine_tune_epochs):
                    ce, acc = train_dataset(net, mixed, cfg.fine_tune_lr, cfg.batch, rng, job("fine_tune"))
                tune_seconds = time.time() - t0
                rounds, matches = evaluate(iteration, 700 + iteration, 800 + iteration)
                key = gate_key(rounds, matches, cfg.gate_metric)
                tested[iteration] = (rounds, matches)
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
                                   {"winning_decisions": len(fresh_rows), "ce": ce, "copied": take,
                                    "replay": won_count, "seconds": round(time.time() - t_gen, 1)})
                passes_run = iteration
                if not kept and cfg.revert_on_reject:
                    net = _restart_from(net, best, device)
                    log(f"  next pass starts again from gen {best_index}")

        # The gate's replay: the top generations by their own tests, Gen 0
        # included if it ranks there, play one shared, larger set of games,
        # and the best of them becomes the pick.
        gate_replay: dict | None = None
        ranked_gens = sorted(tested, key=lambda gi: gate_key(*tested[gi], cfg.gate_metric), reverse=True)
        finalists = ranked_gens[:cfg.gate_finalists]
        if len(finalists) > 1 and cfg.gate_rounds > 0:
            replays: dict[str, dict] = {}
            for gi in finalists:
                obs.phase("final", f"Gate replay for Gen {gi}: {cfg.gate_rounds} rounds and {cfg.gate_matches} matches, "
                                   f"the same games for all {len(finalists)} finalists")
                spec = net_spec(generations[gi])
                gr = workers.rounds(spec, counter, cfg.gate_rounds, 41000, job("final"))
                gm = (workers.matches(spec, counter, cfg.gate_matches, 41001, job("final")) if cfg.gate_matches > 0
                      else {"wins": 0, "losses": 0, "ties": 0, "win_rate": 0.0})
                replays[str(gi)] = {"rounds": gr, "matches": gm, "training_rounds": tested[gi][0],
                                    "training_matches": tested[gi][1]}
                log(f"Gate replay gen {gi}: rounds {gr['win_rate']:.3f} matches {gm['wins']}-{gm['losses']}-{gm['ties']} "
                    f"({gm['win_rate']:.3f}), its own test said rounds {tested[gi][0]['win_rate']:.3f}")
            top = max(finalists, key=lambda gi: gate_key(replays[str(gi)]["rounds"], replays[str(gi)]["matches"],
                                                         cfg.gate_metric))
            if top != best_index:
                log(f"Gate replay picks gen {top} over gen {best_index}")
            else:
                log(f"Gate replay confirms gen {top}")
            best_index = top
            best = generations[top]
            best_round, best_match = replays[str(top)]["rounds"], replays[str(top)]["matches"]
            gate_replay = {"finalists": finalists, "chosen": top, "results": replays,
                           "rounds": cfg.gate_rounds, "matches": cfg.gate_matches}
            entry = {"phase": "gate_replay", **gate_replay}
            history.append(entry)
            obs.history(entry)

        fresh_test: dict[str, dict] = {}
        if cfg.final_matches > 0 and generations:
            candidates = [best_index] if best_index == 0 else [best_index, 0]
            for gi in candidates:
                obs.phase("final", f"Fresh test for Gen {gi}: {cfg.final_rounds} rounds and {cfg.final_matches} matches on new games")
                spec = net_spec(generations[gi])
                fr = workers.rounds(spec, counter, cfg.final_rounds, 31337, job("final"))
                fm = workers.matches(spec, counter, cfg.final_matches, 31338, job("final"))
                fresh_test[str(gi)] = {"rounds": fr, "matches": fm}
                log(f"Fresh test gen {gi}: rounds {fr['win_rate']:.3f} matches {fm['wins']}-{fm['losses']}-{fm['ties']} ({fm['win_rate']:.3f})")
            chosen = best_index
            if best_index != 0:
                pick, copy = fresh_test[str(best_index)], fresh_test["0"]
                if (gate_key(copy["rounds"], copy["matches"], cfg.gate_metric)
                        > gate_key(pick["rounds"], pick["matches"], cfg.gate_metric)):
                    chosen = 0
                    log(f"Gen 0 did better than gen {best_index} on new games, so Gen 0 ships")
                else:
                    log(f"Gen {best_index} held up on new games and ships")
            best_index = chosen
            best = generations[chosen]
            best_round = fresh_test[str(chosen)]["rounds"]
            best_match = fresh_test[str(chosen)]["matches"]
            entry = {"phase": "fresh_test", "chosen": chosen, "results": fresh_test,
                     "rounds": cfg.final_rounds, "matches": cfg.final_matches}
            history.append(entry)
            obs.history(entry)

        obs.phase("final", "Best generation against the greedy and random players")
        best_spec = net_spec(best)
        greedy_rec = workers.rounds(best_spec, greedy, cfg.eval_rounds, 910, job("final"))
        randoms = workers.rounds(best_spec, bot_spec("random"), max(40, cfg.eval_rounds // 2), 911)
    finally:
        workers.close()

    report = {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "input_dim": INPUT_DIM,
        "layers": best.dims,
        "expert_rounds": cfg.expert_rounds,
        "expert_decisions": len(expert),
        "self_play_games": self_play_games,
        "self_play_winning_decisions": self_play_decisions,
        "self_play_iterations": passes_run,
        "imitation_epochs": cfg.epochs,
        "seconds": round(time.time() - started, 1),
        "device": device.label(),
        "best_generation": best_index,
        "gate_replay": gate_replay,
        "fresh_test": fresh_test,
        "match_win_rate_vs_benchmark": best_match["win_rate"],
        "match_record_vs_benchmark": best_match,
        "round_win_rate_vs_benchmark": best_round["win_rate"],
        "round_record_vs_benchmark": best_round,
        "round_win_rate_vs_greedy": greedy_rec["win_rate"],
        "round_record_vs_greedy": greedy_rec,
        "round_win_rate_vs_random": randoms["win_rate"],
        "benchmark_vs_greedy": bench,
        "history": history,
        "config": asdict(cfg),
        "recipe": cfg.recipe,
        "population_history": population_history,
        "rules": {"yaniv_max": 5, "assaf_penalty": 30, "match_limit": 100, "deal": 5, "max_plies": MAX_PLIES},
    }
    meta = {
        "match_win_rate_vs_benchmark": best_match["win_rate"],
        "self_play_games": self_play_games,
        "expert_decisions": len(expert),
        "generation": best_index,
    }
    paths: dict[str, str] = {}
    if run_dir is not None:
        save_net(best, run_dir / "best.json", meta)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        paths["run_dir"] = str(run_dir)
    if model_path is not None:
        export_best(best, report, meta, model_path, report_path)
        paths["model"] = str(model_path)
        if report_path is not None:
            paths["report"] = str(report_path)
    log(
        f"Best is gen {best_index}: vs counter matches {best_match['wins']}-{best_match['losses']}-{best_match['ties']} "
        f"rounds {best_round['win_rate']:.3f}, vs greedy {greedy_rec['win_rate']:.3f}, vs random {randoms['win_rate']:.3f}"
    )
    for where in paths.values():
        log(f"Saved {where}")
    obs.done(report, paths)
    return report


def export_best(best: MLP, report: dict, meta: dict, model_path: Path = MODEL_PATH,
                report_path: Path | None = REPORT_PATH) -> None:
    """Write a net and its report where the game loads them."""
    save_net(best, model_path, meta)
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="the first training recipe, single core")
    parser.add_argument("--device", default="numpy", help="numpy (default), auto, cuda, mps or cpu")
    args = parser.parse_args()
    cfg = TrainConfig.quick() if args.quick else TrainConfig.legacy() if args.legacy else TrainConfig()
    run_training(cfg, device=resolve_device(args.device))


if __name__ == "__main__":
    main()

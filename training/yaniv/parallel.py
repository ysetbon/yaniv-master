"""Play games on several CPU cores.

Self-play and evaluation are pure Python, so they are what limits a run, not
the network. A pool of worker processes splits each batch of games into
chunks. Players travel to the workers as small specs (a bot name, or a
network's weights) because closures cannot be pickled.

With one worker everything runs in this process and matches the old,
single-core results exactly.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .network import MLP

Spec = tuple  # ("counter",) | ("greedy",) | ("random",) | ("net", dims, weights, biases, temperature) | ("switch", ...)


def _parts(net: MLP) -> tuple:
    return list(net.dims), [np.asarray(w) for w in net.W], [np.asarray(b) for b in net.b]


def net_spec(net, temperature: float = 0.0) -> Spec:
    """A plain net as ("net", dims, W, b, temperature); a decider with experts as ("moe", gate, experts, temperature)."""
    if hasattr(net, "experts"):
        return ("moe", _parts(net.gate), [_parts(e) for e in net.experts], temperature)
    return ("net", *_parts(net), temperature)


def bot_spec(name: str) -> Spec:
    return (name,)


def chooser(spec: Spec):
    from .heuristic import choose_greedy, choose_heuristic, choose_random
    from .train import net_chooser

    kind = spec[0]
    if kind == "counter":
        return choose_heuristic
    if kind == "greedy":
        return choose_greedy
    if kind == "random":
        return choose_random
    if kind in ("net", "moe"):
        return net_chooser(_net(spec), spec[-1])
    if kind == "switch":
        from .hybrid import switch_chooser

        return switch_chooser(chooser(spec[1]), chooser(spec[2]), spec[3])
    raise ValueError(f"unknown player {kind}")


def switch_spec(main: Spec, endgame: Spec, at: int = 4) -> Spec:
    """`main` plays until both hands hold `at` cards or fewer, then `endgame` plays (hybrid.py)."""
    return ("switch", main, endgame, at)


def _mlp(dims, weights, biases) -> MLP:
    net = MLP(dims, seed=0)
    net.W = [w.astype(np.float32, copy=False) for w in weights]
    net.b = [b.astype(np.float32, copy=False) for b in biases]
    return net


def _net(spec: Spec):
    if spec[0] == "moe":
        from .moe import MoE

        return MoE(_mlp(*spec[1]), [_mlp(*parts) for parts in spec[2]])
    return _mlp(spec[1], spec[2], spec[3])


# ---- work done inside a worker -------------------------------------------
def _rounds(a: Spec, b: Spec, n: int, seed: int) -> dict:
    from .train import round_duel

    return round_duel(chooser(a), chooser(b), n, seed)


def _matches(a: Spec, b: Spec, n: int, seed: int) -> dict:
    from .train import match_duel

    return match_duel(chooser(a), chooser(b), n, seed)


def _self_play(spec: Spec, n: int, seed: int, temperature: float, weighted: bool, offset: int,
               match_scores: bool) -> list:
    from .train import collect_self_play

    return collect_self_play(_net(spec), n, seed, temperature, weighted=weighted, offset=offset,
                             match_scores=match_scores)


def _teacher_label(spec: Spec, n: int, seed: int) -> list:
    from .train import collect_teacher

    return collect_teacher(_net(spec), n, seed)


def _search_label(spec: Spec, n: int, seed: int, samples: int, width: int, match_scores: bool,
                  confidence: float, table, value_parts=None, depth: int = 2) -> list:
    from .search import collect_search

    value = _mlp(*value_parts) if value_parts is not None else None
    return collect_search(_net(spec), n, seed, samples, width, match_scores=match_scores, confidence=confidence,
                          table=table, value=value, depth=depth)


def _value_data(spec: Spec, n: int, seed: int, table, match_scores: bool) -> tuple:
    from .value import collect_value

    return collect_value(_net(spec), n, seed, table, match_scores)


def _round_outcomes(spec: Spec, n: int, seed: int) -> list:
    from .match_value import round_outcomes

    return round_outcomes(_net(spec), n, seed)


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _init_worker() -> None:
    # Too late to change NumPy's thread count (it is imported by then), but
    # anything the worker starts itself inherits one thread.
    for key in THREAD_VARS:
        os.environ[key] = "1"
    _lower_priority()


def _lower_priority() -> None:
    """Run game workers below normal priority, so the computer stays responsive."""
    try:
        if os.name == "nt":
            import ctypes

            below_normal = 0x00004000
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), below_normal)
        else:
            os.nice(5)
    except Exception:  # priority is a courtesy; never fail a worker over it
        pass


def _merge_rounds(parts: list[dict], n: int) -> dict:
    total = {k: sum(p[k] for p in parts) for k in ("wins", "losses", "ties", "calls", "assafs")}
    hero = sum(p["hero_avg"] * (p["wins"] + p["losses"] + p["ties"]) for p in parts)
    villain = sum(p["villain_avg"] * (p["wins"] + p["losses"] + p["ties"]) for p in parts)
    return {
        **total,
        "win_rate": total["wins"] / max(1, n),
        "hero_avg": hero / max(1, n),
        "villain_avg": villain / max(1, n),
        "assaf_rate": total["assafs"] / total["calls"] if total["calls"] else 0.0,
    }


def _merge_matches(parts: list[dict], n: int) -> dict:
    total = {k: sum(p[k] for p in parts) for k in ("wins", "losses", "ties")}
    return {**total, "win_rate": total["wins"] / max(1, n)}


def default_workers(share: float | None = None) -> int:
    """Game-playing processes. With `share` (0 to 1), that fraction of the logical cores.

    Without it, every core but two, which are left for the dashboard, the
    live table and the OS.
    """
    cores = os.cpu_count() or 2
    if share is not None:
        return max(1, min(cores, round(cores * share)))
    return max(1, min(16, cores - 2))


class Workers:
    def __init__(self, count: int):
        self.count = max(1, int(count))
        self._pool: ProcessPoolExecutor | None = None

    def _get_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # Each worker plays one game at a time, so its BLAS gets one thread.
            # A spawned worker reads these when it first imports NumPy, so they
            # must be set before the pool starts any. With several threads per
            # worker, all the workers' threads fight over the cores, and search
            # labelling ran about 8 times slower than one process alone.
            # NumPy in this process is already loaded and keeps its own setting.
            for key in THREAD_VARS:
                os.environ[key] = "1"
            self._pool = ProcessPoolExecutor(
                self.count, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker
            )
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None

    def _chunks(self, n: int, multiple: int) -> list[tuple[int, int]]:
        """(offset, size) pieces, each a multiple of `multiple` except possibly the last."""
        pieces = self.count * 3
        size = max(multiple, -(-n // pieces))
        size = -(-size // multiple) * multiple
        return [(start, min(size, n - start)) for start in range(0, n, size)]

    def _run(self, fn, jobs: list[tuple], sizes: list[int], total: int, progress):
        pool = self._get_pool()
        futures = {pool.submit(fn, *job): (index, size) for index, (job, size) in enumerate(zip(jobs, sizes))}
        results: list = [None] * len(jobs)
        done = 0
        for future in as_completed(futures):
            index, size = futures[future]
            results[index] = future.result()
            done += size
            if progress is not None:
                progress(done, total)
        return results

    def rounds(self, a: Spec, b: Spec, n: int, seed: int, progress=None) -> dict:
        if self.count == 1:
            from .train import round_duel

            return round_duel(chooser(a), chooser(b), n, seed, progress)
        chunks = self._chunks(n, 2)  # even chunks keep mirrored pairs together
        jobs = [(a, b, size, seed * 7919 + i) for i, (_, size) in enumerate(chunks)]
        parts = self._run(_rounds, jobs, [size for _, size in chunks], n, progress)
        return _merge_rounds(parts, n)

    def rounds_many(self, specs: list[Spec], opponent: Spec, n: int, seed: int, progress=None) -> list[dict]:
        """Each spec plays `n` rounds against `opponent` on the same deals (same seed).

        One job per spec, so each network's weights are sent once.
        """
        if self.count == 1:
            from .train import round_duel

            out = []
            for i, spec in enumerate(specs):
                out.append(round_duel(chooser(spec), chooser(opponent), n, seed))
                if progress is not None:
                    progress((i + 1) * n, len(specs) * n)
            return out
        jobs = [(spec, opponent, n, seed) for spec in specs]
        return self._run(_rounds, jobs, [n] * len(specs), len(specs) * n, progress)

    def matches(self, a: Spec, b: Spec, n: int, seed: int, progress=None) -> dict:
        if self.count == 1:
            from .train import match_duel

            return match_duel(chooser(a), chooser(b), n, seed, progress)
        chunks = self._chunks(n, 2)
        jobs = [(a, b, size, seed * 7919 + i) for i, (_, size) in enumerate(chunks)]
        parts = self._run(_matches, jobs, [size for _, size in chunks], n, progress)
        return _merge_matches(parts, n)

    def teacher_label(self, net: MLP, n: int, seed: int, progress=None) -> list:
        """(view, label) rows: `n` rounds of `net` against the counter, every position labelled with its move."""
        if self.count == 1:
            from .train import collect_teacher

            return collect_teacher(net, n, seed, progress)
        chunks = self._chunks(n, 2)
        jobs = [(net_spec(net), size, seed * 7919 + i) for i, (_, size) in enumerate(chunks)]
        parts = self._run(_teacher_label, jobs, [size for _, size in chunks], n, progress)
        return [row for part in parts for row in part]

    def round_outcomes(self, net, n: int, seed: int) -> list:
        """(net's points, counter's points) for `n` rounds of the net against the card counter."""
        if self.count == 1:
            from .match_value import round_outcomes

            return round_outcomes(net, n, seed)
        chunks = self._chunks(n, 2)
        jobs = [(net_spec(net), size, seed * 7919 + i) for i, (_, size) in enumerate(chunks)]
        parts = self._run(_round_outcomes, jobs, [size for _, size in chunks], n, None)
        return [row for part in parts for row in part]

    def search_label(self, net: MLP, n: int, seed: int, samples: int, width: int, progress=None,
                     match_scores: bool = True, confidence: float = 1.5, table=None, value: MLP | None = None,
                     depth: int = 2) -> list:
        """(view, label) rows: `n` rounds of the net against the counter, every decision searched.

        With a `value` net the search looks `depth` moves ahead and asks it, instead of playing to the end.
        """
        if self.count == 1:
            from .search import collect_search

            return collect_search(net, n, seed, samples, width, progress, match_scores=match_scores,
                                  confidence=confidence, table=table, value=value, depth=depth)
        spec = net_spec(net)
        value_parts = _parts(value) if value is not None else None
        chunks = self._chunks(n, 1)
        jobs = [(spec, size, seed * 7919 + i, samples, width, match_scores, confidence, table, value_parts, depth)
                for i, (_, size) in enumerate(chunks)]
        parts = self._run(_search_label, jobs, [size for _, size in chunks], n, progress)
        return [row for part in parts for row in part]

    def value_data(self, net, n: int, seed: int, table=None, match_scores: bool = True, progress=None) -> tuple:
        """(features, targets) from `n` rounds of the net playing itself, for the value net (value.py)."""
        if self.count == 1:
            from .value import collect_value

            return collect_value(net, n, seed, table, match_scores, progress=progress)
        spec = net_spec(net)
        chunks = self._chunks(n, 1)
        jobs = [(spec, size, seed * 7919 + i, table, match_scores) for i, (_, size) in enumerate(chunks)]
        parts = self._run(_value_data, jobs, [size for _, size in chunks], n, progress)
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    def self_play(self, net: MLP, n: int, seed: int, temperature: float, weighted: bool, progress=None,
                  match_scores: bool = False) -> list:
        if self.count == 1:
            from .train import collect_self_play

            return collect_self_play(net, n, seed, temperature, progress, weighted=weighted, match_scores=match_scores)
        spec = net_spec(net, temperature)
        chunks = self._chunks(n, 4)  # multiples of 4 keep the one-in-four bench games
        jobs = [(spec, size, seed * 7919 + i, temperature, weighted, offset, match_scores)
                for i, (offset, size) in enumerate(chunks)]
        parts = self._run(_self_play, jobs, [size for _, size in chunks], n, progress)
        return [row for part in parts for row in part]

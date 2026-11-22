"""The replay recipe: subsets, weighted loss, the cache and the worker pool."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .backend import as_numpy, build_net, resolve_device
from .network import default_net
from .parallel import Workers, bot_spec, net_spec
from .train import (
    Dataset,
    TrainConfig,
    collect_expert,
    collect_self_play,
    concat_datasets,
    margin_weight,
    materialize,
    subset,
    train_dataset,
)

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except (ImportError, OSError):
    HAVE_TORCH = False


def small_data(seed: int = 77) -> Dataset:
    return materialize(collect_expert(10, seed))


class DatasetOps(unittest.TestCase):
    def test_subset_keeps_each_decision_with_its_moves(self):
        data = small_data()
        ids = [5, 0, 17, 3]
        part = subset(data, ids)
        self.assertEqual(len(part), len(ids))
        for new, old in enumerate(ids):
            a, b = data.offsets[old], data.offsets[old + 1]
            c, d = part.offsets[new], part.offsets[new + 1]
            np.testing.assert_array_equal(part.actions[c:d], data.actions[a:b])
            np.testing.assert_array_equal(part.states[new], data.states[old])
            self.assertEqual(part.labels[new], data.labels[old])

    def test_concat_then_subset_round_trips(self):
        a, b = small_data(1), small_data(2)
        both = concat_datasets([a, b])
        back = subset(both, range(len(a), len(a) + len(b)))
        np.testing.assert_array_equal(back.actions, b.actions)
        np.testing.assert_array_equal(back.offsets, b.offsets)

    def test_cache_file_round_trips(self):
        data = small_data()
        data.weights[:] = np.linspace(0.5, 2.0, len(data), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.npz"
            data.save(path)
            back = Dataset.load(path)
        for name in ("states", "actions", "offsets", "labels", "weights"):
            np.testing.assert_array_equal(getattr(back, name), getattr(data, name))


class Weighting(unittest.TestCase):
    def test_margin_weight_range(self):
        self.assertEqual(margin_weight(0), 0.5)
        self.assertEqual(margin_weight(30), 2.0)
        self.assertEqual(margin_weight(99), 2.0)

    def test_unit_weights_match_unweighted_training(self):
        data = small_data()
        plain, ones = default_net(5), default_net(5)
        train_dataset(plain, data, 1e-3, 16, random.Random(9))
        data.weights = np.ones(len(data), dtype=np.float32)
        train_dataset(ones, data, 1e-3, 16, random.Random(9))
        for wa, wb in zip(plain.W, ones.W):
            np.testing.assert_array_equal(wa, wb)

    def test_zero_weight_rows_do_not_teach(self):
        data = small_data()
        keep = subset(data, range(0, 20))
        mute = subset(data, range(20, len(data)))
        mute.weights[:] = 0.0
        a, b = default_net(5), default_net(5)
        train_dataset(a, keep, 1e-3, 1000, random.Random(1))
        train_dataset(b, concat_datasets([keep, mute]), 1e-3, 1000, random.Random(1))
        for wa, wb in zip(a.W, b.W):
            np.testing.assert_allclose(wa, wb, atol=1e-6)

    @unittest.skipUnless(HAVE_TORCH, "PyTorch is not available")
    def test_weighted_torch_matches_numpy(self):
        data = small_data()
        data.weights = np.random.default_rng(0).uniform(0.5, 2.0, len(data)).astype(np.float32)
        a = build_net(resolve_device("numpy"), 5)
        b = build_net(resolve_device("cpu"), 5)
        ce_a, _ = train_dataset(a, data, 1e-3, 16, random.Random(9))
        ce_b, _ = train_dataset(b, data, 1e-3, 16, random.Random(9))
        self.assertAlmostEqual(ce_a, ce_b, places=3)
        # Compare what the nets do, not raw weights: Adam turns float rounding on
        # weights of rarely-active inputs into lr-sized steps in either direction.
        x = np.concatenate([np.repeat(data.states, np.diff(data.offsets), axis=0), data.actions], 1)
        sa, sb = a.forward(x)[:, 0], as_numpy(b).forward(x)[:, 0]
        self.assertLess(float(np.abs(sa - sb).max()), 0.02 * max(1.0, float(np.ptp(sa))))
        picks = [int(np.argmax(sa[lo:hi]) == np.argmax(sb[lo:hi])) for lo, hi in zip(data.offsets[:-1], data.offsets[1:])]
        self.assertGreaterEqual(sum(picks) / len(picks), 0.99)

    def test_weighted_self_play_rows_carry_weights(self):
        rows = collect_self_play(default_net(3), 8, 1, 0.32, weighted=True)
        self.assertTrue(rows)
        self.assertTrue(all(len(r) == 3 and 0.5 <= r[2] <= 2.0 for r in rows))
        self.assertTrue(all(len(r) == 2 for r in collect_self_play(default_net(3), 8, 1, 0.32)))


class WorkerPool(unittest.TestCase):
    def test_pool_totals_add_up(self):
        workers = Workers(2)
        try:
            rec = workers.rounds(bot_spec("counter"), bot_spec("random"), 20, 4)
            self.assertEqual(rec["wins"] + rec["losses"] + rec["ties"], 20)
            self.assertGreater(rec["win_rate"], 0.6)
            matches = workers.matches(bot_spec("counter"), bot_spec("random"), 4, 4)
            self.assertEqual(matches["wins"] + matches["losses"] + matches["ties"], 4)
            rows = workers.self_play(default_net(3), 12, 2, 0.32, True)
            self.assertTrue(rows and all(len(r) == 3 for r in rows))
            net_rec = workers.rounds(net_spec(default_net(3)), bot_spec("counter"), 6, 1)
            self.assertEqual(net_rec["wins"] + net_rec["losses"] + net_rec["ties"], 6)
        finally:
            workers.close()

    def test_mirrored_pairs_cancel_card_luck(self):
        # Two copies of the same deterministic player play each deal from both
        # seats, so every pair is one win and one loss (or two ties).
        from .heuristic import choose_heuristic
        from .train import match_duel, round_duel

        rounds = round_duel(choose_heuristic, choose_heuristic, 40, 5)
        self.assertEqual(rounds["wins"], rounds["losses"])
        self.assertAlmostEqual(rounds["hero_avg"], rounds["villain_avg"])
        matches = match_duel(choose_heuristic, choose_heuristic, 6, 5)
        self.assertEqual(matches["wins"], matches["losses"])

    def test_mirrored_deals_ignore_the_players_random_draws(self):
        # The random player draws from the shared rng; the deals must not move.
        from .heuristic import choose_heuristic, choose_random
        from .train import round_duel

        a = round_duel(choose_heuristic, choose_random, 20, 8)
        b = round_duel(choose_heuristic, choose_heuristic, 20, 8)
        c = round_duel(choose_heuristic, choose_random, 20, 8)
        self.assertEqual(a, c)
        self.assertNotEqual(a, b)

    def test_one_worker_matches_the_plain_functions(self):
        from .heuristic import choose_heuristic, choose_random
        from .train import round_duel

        direct = round_duel(choose_heuristic, choose_random, 10, 3)
        pooled = Workers(1).rounds(bot_spec("counter"), bot_spec("random"), 10, 3)
        self.assertEqual(direct, pooled)


class Config(unittest.TestCase):
    def test_expert_mix_slides_from_start_to_end(self):
        cfg = TrainConfig(generations=5, expert_mix_start=2.0, expert_mix_end=0.5)
        self.assertAlmostEqual(cfg.expert_mix(1), 2.0)
        self.assertAlmostEqual(cfg.expert_mix(5), 0.5)
        self.assertTrue(cfg.expert_mix(2) > cfg.expert_mix(3) > cfg.expert_mix(4))

    def test_legacy_is_the_first_recipe(self):
        cfg = TrainConfig.legacy()
        self.assertEqual((cfg.replay_capacity, cfg.weighted, cfg.workers, cfg.eval_matches), (0, False, 1, 48))


if __name__ == "__main__":
    unittest.main()

"""handmade5_pack: two nets and a hand-written decider as one model that plays exactly as the switch player."""

from __future__ import annotations

import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .engine import play_round
from .features import INPUT_DIM, STATE_DIM, score_actions, state_features
from .handmade5_pack import GAP, check, decider, pack, same_shape
from .handmade_kit import INPUTS
from .hybrid import WHEN, small_enough, switch_chooser
from .moe import _softmax
from .network import MLP, load_net, save_net
from .train import net_chooser


def _net(dims: list[int], seed: int) -> MLP:
    net = MLP(dims, seed=seed)
    net.W[-1] *= 100.0  # wake up the small last layer
    for b in net.b:
        b += np.random.default_rng(seed).standard_normal(len(b)).astype(np.float32) * 0.1
    return net


def _one_by_one(net: MLP, x: np.ndarray) -> np.ndarray:
    """The browser's forward (src/lib/ai/net.ts): each unit summed input by input in float64, kept as float32."""
    h = x.astype(np.float32)
    last = len(net.W) - 1
    for i, (w, b) in enumerate(zip(net.W, net.b)):
        out = np.empty(w.shape[1], dtype=np.float32)
        for j in range(w.shape[1]):
            total = float(b[j])
            for k in range(w.shape[0]):
                total += float(h[k]) * float(w[k, j])
            out[j] = total if i == last else max(total, 0.0)
        h = out
    return h


class Decider(unittest.TestCase):
    def test_the_weights_are_exactly_one_and_zero_for_every_rule_size_and_pair_of_hands(self):
        rng = np.random.default_rng(1)
        for when in WHEN:
            for at in range(6):
                gate = decider(when, at)
                self.assertEqual(gate.dims, [STATE_DIM, 2, 2])
                for mine in range(6):
                    for theirs in range(6):
                        state = rng.random((1, STATE_DIM), dtype=np.float32) * 4  # the other inputs must not matter
                        state[0, INPUTS.index("my_cards")] = np.float32(mine / 5.0)
                        state[0, INPUTS.index("opp_cards")] = np.float32(theirs / 5.0)
                        a, b = mine <= at, theirs <= at
                        small = int({"both": a and b, "mine": a, "opp": b, "either": a or b}[when])
                        out = gate.forward(state)
                        self.assertEqual(out.dtype, np.float32)
                        self.assertGreaterEqual(float(out[0, small] - out[0, 1 - small]), GAP)
                        weights = _softmax(out)[0]
                        self.assertEqual((float(weights[small]), float(weights[1 - small])), (1.0, 0.0))
                        # The browser: Math.exp on float64.
                        self.assertEqual(math.exp(float(out[0, 1 - small]) - float(out[0, small])), 0.0)

    def test_the_gap_underflows_in_both_number_types(self):
        self.assertEqual(float(np.exp(np.float32(-GAP))), 0.0)
        self.assertEqual(math.exp(-GAP), 0.0)
        self.assertGreater(GAP, 2 * 745.2)  # float64's exp is 0 below about -745.2, float32's below about -104

    def test_it_refuses_an_unknown_rule(self):
        with self.assertRaises(ValueError):
            decider("never")


class SameShape(unittest.TestCase):
    def test_padding_changes_no_bit_when_units_are_summed_one_by_one(self):
        rng = np.random.default_rng(2)
        rows = rng.random((3, INPUT_DIM), dtype=np.float32)
        for first, second in (([INPUT_DIM, 9, 5, 1], [INPUT_DIM, 6, 7, 8, 4, 3, 1]),
                              ([INPUT_DIM, 4, 6, 5, 7, 3, 1], [INPUT_DIM, 12, 1]),
                              ([INPUT_DIM, 8, 8, 1], [INPUT_DIM, 8, 8, 1])):
            a, b = _net(first, 3), _net(second, 4)
            wide_a, wide_b = same_shape(a, b)
            self.assertEqual(wide_a.dims, wide_b.dims)
            self.assertEqual(len(wide_a.W), max(len(a.W), len(b.W)))
            for net, wide in ((a, wide_a), (b, wide_b)):
                for row in rows:
                    self.assertEqual(_one_by_one(net, row).tobytes(), _one_by_one(wide, row).tobytes())
                np.testing.assert_allclose(wide.forward(rows), net.forward(rows), rtol=1e-5, atol=1e-5)

    def test_nets_of_one_shape_are_left_as_they_are(self):
        a, b = _net([INPUT_DIM, 8, 8, 1], 5), _net([INPUT_DIM, 8, 8, 1], 6)
        wide_a, wide_b = same_shape(a, b)
        for net, wide in ((a, wide_a), (b, wide_b)):
            for w, v in zip(net.W + net.b, wide.W + wide.b):
                self.assertEqual(w.tobytes(), v.tobytes())

    def test_only_plain_nets_are_packed(self):
        a = _net([INPUT_DIM, 8, 1], 5)
        with self.assertRaises(ValueError):
            pack(pack(a, a), a)
        with self.assertRaises(ValueError):
            pack(_net([INPUT_DIM - 1, 8, 1], 5), a)


class Packed(unittest.TestCase):
    def test_it_plays_exactly_as_the_switch_player_does(self):
        main, endgame = _net([INPUT_DIM, 16, 12, 1], 7), _net([INPUT_DIM, 10, 14, 6, 1], 8)
        for when, at in (("both", 4), ("mine", 4), ("opp", 3), ("either", 4)):
            result = check(main, endgame, when, at, decisions=400, seed=5)
            self.assertEqual(result["hand_size"], 5)
            self.assertGreater(result["after_handover"], 0)
            self.assertLess(result["after_handover"], result["decisions"])
            for key in ("weights_not_exact", "not_finite", "moves_differ_padded", "rows_differ_padded"):
                self.assertEqual(result[key], 0, f"{key} with {when} at {at}")
            self.assertLess(result["max_score_diff_original"], 1e-3)

    def test_nets_of_one_shape_give_the_switch_players_scores_bit_for_bit(self):
        main, endgame = _net([INPUT_DIM, 16, 12, 1], 9), _net([INPUT_DIM, 16, 12, 1], 10)
        result = check(main, endgame, decisions=150, seed=6)
        for key in ("moves_differ_original", "rows_differ_original", "moves_differ_padded", "rows_differ_padded"):
            self.assertEqual(result[key], 0, key)
        self.assertEqual(result["max_score_diff_original"], 0.0)

    def test_the_other_nets_score_never_leaks_not_even_a_huge_one(self):
        main, endgame = _net([INPUT_DIM, 16, 12, 1], 11), _net([INPUT_DIM, 10, 1], 12)
        loud = endgame.clone()
        loud.b[-1] += 1.0e30  # finite, and enormous
        quiet, noisy = pack(main, endgame), pack(main, loud)
        seen = {"main": 0, "end": 0}

        def choose(view, actions, rng):
            small = small_enough(view, 4)
            seen["end" if small else "main"] += 1
            a, b = score_actions(quiet, view, actions), score_actions(noisy, view, actions)
            self.assertTrue(np.isfinite(b).all())
            if not small:
                self.assertEqual(a.tobytes(), b.tobytes())
            return rng.choice(actions)

        for seed in range(6):
            play_round(choose, choose, random.Random(seed), [0, 0], seed % 2)
        self.assertTrue(seen["main"] and seen["end"])

    def test_the_file_round_trips_and_one_row_alone_scores_the_same(self):
        main, endgame = _net([INPUT_DIM, 16, 12, 1], 13), _net([INPUT_DIM, 10, 14, 6, 1], 14)
        packed = pack(main, endgame)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "packed.json"
            save_net(packed, path, meta={"handmade": True})
            loaded = load_net(path)
        self.assertEqual(loaded.gate.dims, [STATE_DIM, 2, 2])
        self.assertEqual(len(loaded.experts), 2)
        reference = switch_chooser(net_chooser(main, 0.0), net_chooser(endgame, 0.0))
        player = net_chooser(loaded, 0.0)
        rng = random.Random(3)

        def choose(view, actions, rng_):
            self.assertEqual(player(view, actions, None), net_chooser(packed, 0.0)(view, actions, None))
            weights = loaded.weights_for(state_features(view)[None, :])[0]
            self.assertEqual(float(weights[int(small_enough(view, 4))]), 1.0)
            return reference(view, actions, None) if rng.random() < 0.8 else rng.choice(actions)

        for seed in range(4):
            play_round(choose, choose, rng, [0, 0], seed % 2)


if __name__ == "__main__":
    unittest.main()

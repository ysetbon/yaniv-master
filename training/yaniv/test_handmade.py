"""The hand-written net: it builds, has the shape the game loads, and the shipped file is what the code writes."""

import unittest
from pathlib import Path

import numpy as np

from training.yaniv import handmade
from training.yaniv.features import INPUT_DIM
from training.yaniv.handmade_kit import INPUTS, Kit
from training.yaniv.network import load_net


class HandmadeTest(unittest.TestCase):
    def test_kit_matches_its_own_trace(self):
        kit = Kit()
        kit.unit(1, "a", {"pts_kept": -40.0}, 7.0)
        kit.unit(3, "c", {"a": 1.0, "yaniv": 2.0, "sure_yaniv": 1.0}, -1.0)
        kit.out({"c": 1.0, "pts_thrown": 3.0, "a": 0.5}, 0.25)
        net = kit.build()
        x = np.random.default_rng(0).random(len(INPUTS)).astype(np.float32)
        self.assertAlmostEqual(float(net.forward(x[None])[0, 0]), kit.trace(x)["OUT"], places=4)

    def test_shipped_file_is_what_the_code_builds(self):
        net, _ = handmade.build()
        self.assertEqual(net.dims[0], INPUT_DIM)
        self.assertEqual(net.dims[-1], 1)
        if not Path(handmade.MODEL_PATH).exists():
            self.skipTest("the model file has not been written")
        shipped = load_net(Path(handmade.MODEL_PATH))
        self.assertEqual(shipped.dims, net.dims)
        for ours, theirs in zip(net.W + net.b, shipped.W + shipped.b):
            np.testing.assert_array_equal(ours, theirs)


if __name__ == "__main__":
    unittest.main()

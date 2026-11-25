"""The hand-written real-game net: it builds, and the shipped files are what the code writes."""

import unittest

import numpy as np

from training.yaniv import handmade5
from training.yaniv.features import INPUT_DIM
from training.yaniv.network import load_net


def _same(test, ours, theirs):
    test.assertEqual(ours.dims, theirs.dims)
    for a, b in zip(ours.W + ours.b, theirs.W + theirs.b):
        np.testing.assert_array_equal(a, b)


class Handmade5Test(unittest.TestCase):
    def test_shipped_net_is_what_the_code_builds(self):
        net, _ = handmade5.build()
        self.assertEqual((net.dims[0], net.dims[-1]), (INPUT_DIM, 1))
        if not handmade5.MODEL_PATH.exists():
            self.skipTest("the model file has not been written")
        _same(self, net, load_net(handmade5.MODEL_PATH))

    def test_shipped_combination_is_this_net_then_the_4_card_net(self):
        if not (handmade5.COMBINED_PATH.exists() and handmade5.ENDGAME_PATH.exists()):
            self.skipTest("the model files have not been written")
        built, shipped = handmade5.build_combined(), load_net(handmade5.COMBINED_PATH)
        _same(self, built.gate, shipped.gate)
        self.assertEqual(len(shipped.experts), 2)
        for ours, theirs in zip(built.experts, shipped.experts):
            _same(self, ours, theirs)


if __name__ == "__main__":
    unittest.main()

"""Team opus's contest net: it builds, it is one plain net within the size limit, and the shipped file is what
the code writes."""

import unittest

import numpy as np

from training.yaniv import contest_opus
from training.yaniv.features import INPUT_DIM
from training.yaniv.network import MLP, load_net


class ContestOpusTest(unittest.TestCase):
    def test_shape_and_size(self):
        net, _ = contest_opus.build(contest_opus.EXTRAS)
        self.assertIsInstance(net, MLP)
        self.assertEqual((net.dims[0], net.dims[-1]), (INPUT_DIM, 1))
        self.assertLessEqual(sum(net.dims[1:-1]), 1020)

    def test_shipped_net_is_what_the_code_builds(self):
        if not contest_opus.MODEL_PATH.exists():
            self.skipTest("the model file has not been written")
        net, _ = contest_opus.build(contest_opus.EXTRAS)
        shipped = load_net(contest_opus.MODEL_PATH)
        self.assertIsInstance(shipped, MLP)
        self.assertEqual(net.dims, shipped.dims)
        self.assertLessEqual(sum(shipped.dims[1:-1]), 1020)
        for a, b in zip(net.W + net.b, shipped.W + shipped.b):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()

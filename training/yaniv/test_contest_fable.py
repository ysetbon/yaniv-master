"""Team fable's contest net: it builds, it is one plain net within the size limit, and the shipped file is
what the code writes."""

import unittest

import numpy as np

from training.yaniv import contest_fable
from training.yaniv.features import INPUT_DIM
from training.yaniv.network import MLP, load_net


class ContestFableTest(unittest.TestCase):
    def test_one_plain_net_within_the_size_limit(self):
        net, _ = contest_fable.build()
        self.assertIsInstance(net, MLP)
        self.assertEqual((net.dims[0], net.dims[-1]), (INPUT_DIM, 1))
        self.assertLessEqual(sum(net.dims[1:-1]), contest_fable.MAX_HIDDEN)
        self.assertEqual(contest_fable.MAX_HIDDEN, 1020)
        self.assertEqual(len(net.W), len(net.dims) - 1)
        for weight, bias in zip(net.W, net.b):
            self.assertTrue(np.isfinite(weight).all() and np.isfinite(bias).all())

    def test_shipped_net_is_what_the_code_builds(self):
        if not contest_fable.MODEL_PATH.exists():
            self.skipTest("the model file has not been written")
        net, _ = contest_fable.build()
        shipped = load_net(contest_fable.MODEL_PATH)
        self.assertIsInstance(shipped, MLP)
        self.assertEqual(net.dims, shipped.dims)
        for ours, theirs in zip(net.W + net.b, shipped.W + shipped.b):
            np.testing.assert_array_equal(ours, theirs)

    def test_scores_every_legal_move_of_a_dealt_hand(self):
        import random

        from training.yaniv.engine import legal_actions, make_view, new_round
        from training.yaniv.features import score_actions

        net, _ = contest_fable.build()
        state = new_round([0, 0], 0, random.Random(3))
        view = make_view(state, 0)
        actions = legal_actions(view)
        scores = score_actions(net, view, actions)
        self.assertEqual(len(scores), len(actions))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()

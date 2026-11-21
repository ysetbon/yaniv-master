"""The PyTorch trainer must take the same steps as the NumPy one."""

from __future__ import annotations

import random
import unittest

import numpy as np

from .backend import as_numpy, build_net, resolve_device
from .train import collect_expert, materialize, train_dataset

try:
    import torch  # noqa: F401

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "PyTorch is not installed")
class TorchMatchesNumpy(unittest.TestCase):
    def test_one_epoch_lands_on_the_same_weights(self):
        data = materialize(collect_expert(12, 77))
        numpy_net = build_net(resolve_device("numpy"), 5)
        torch_net = build_net(resolve_device("cpu"), 5)
        ce_a, acc_a = train_dataset(numpy_net, data, 1e-3, 16, random.Random(9))
        ce_b, acc_b = train_dataset(torch_net, data, 1e-3, 16, random.Random(9))
        self.assertAlmostEqual(ce_a, ce_b, places=3)
        self.assertAlmostEqual(acc_a, acc_b, places=6)
        after = as_numpy(torch_net)
        for wa, wb in zip(numpy_net.W, after.W):
            np.testing.assert_allclose(wa, wb, atol=1e-5)
        # Adam divides by the root of a tiny second moment when a bias gradient
        # is near zero, so float rounding can move a few biases by ~1e-5. The
        # output bias is skipped: softmax over moves ignores a constant shift,
        # so its true gradient is zero and both versions just follow rounding.
        for ba, bb in zip(numpy_net.b[:-1], after.b[:-1]):
            np.testing.assert_allclose(ba, bb, atol=1e-4)

    def test_forward_matches(self):
        torch_net = build_net(resolve_device("cpu"), 5)
        x = np.random.default_rng(0).standard_normal((7, torch_net.dims[0])).astype(np.float32)
        np.testing.assert_allclose(torch_net.forward(x), as_numpy(torch_net).forward(x), atol=1e-5)


class DeviceChoice(unittest.TestCase):
    def test_numpy_is_always_available(self):
        self.assertEqual(resolve_device("numpy").kind, "numpy")

    def test_auto_never_fails(self):
        self.assertIn(resolve_device("auto").kind, ("cuda", "mps", "numpy"))


if __name__ == "__main__":
    unittest.main()

"""PyTorch twin of `network.MLP` that trains on a GPU.

Only the gradient steps run on the device. Self-play and evaluation score a
handful of moves at a time, which is faster on the CPU, so they use a NumPy
copy of the weights from `to_numpy()`.

The update matches `MLP.step`: gradients are summed over the batch and divided
by the number of decisions, weight decay is added to the weights only, the
global norm is clipped at 5, then Adam with bias correction.
"""

from __future__ import annotations

import numpy as np
import torch

from .features import STATE_DIM
from .network import MLP


class TorchMLP:
    def __init__(self, base: MLP, device: torch.device):
        self.device = device
        self.dims = list(base.dims)
        self.W = [torch.tensor(w, dtype=torch.float32, device=device, requires_grad=True) for w in base.W]
        self.b = [torch.tensor(b, dtype=torch.float32, device=device, requires_grad=True) for b in base.b]
        self.mW = [torch.zeros_like(w) for w in self.W]
        self.vW = [torch.zeros_like(w) for w in self.W]
        self.mb = [torch.zeros_like(b) for b in self.b]
        self.vb = [torch.zeros_like(b) for b in self.b]
        self.t = 0

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        last = len(self.W) - 1
        for i, (weight, bias) in enumerate(zip(self.W, self.b)):
            h = h @ weight + bias
            if i != last:
                h = torch.relu(h)
        return h

    def forward(self, x: np.ndarray, cache: bool = False) -> np.ndarray:
        del cache
        with torch.no_grad():
            out = self._forward(torch.as_tensor(x, dtype=torch.float32, device=self.device))
        return out.cpu().numpy()

    def to_numpy(self) -> MLP:
        net = MLP(self.dims, seed=0)
        net.W = [w.detach().cpu().numpy().astype(np.float32, copy=True) for w in self.W]
        net.b = [b.detach().cpu().numpy().astype(np.float32, copy=True) for b in self.b]
        return net

    def _step(self, grads: list[torch.Tensor], n_examples: float, lr: float,
              weight_decay: float = 1e-6, clip: float = 5.0) -> None:
        count = len(self.W)
        scale = 1.0 / max(1, n_examples)
        with torch.no_grad():
            grads_w = [g * scale for g in grads[:count]]
            grads_b = [g * scale for g in grads[count:]]
            if weight_decay:
                grads_w = [g + weight_decay * w for g, w in zip(grads_w, self.W)]
            norm = torch.sqrt(sum((g * g).sum() for g in grads_w + grads_b))
            shrink = torch.clamp(clip / norm, max=1.0)
            grads_w = [g * shrink for g in grads_w]
            grads_b = [g * shrink for g in grads_b]
            self.t += 1
            b1, b2, eps = 0.9, 0.999, 1e-8
            bc1 = 1 - b1**self.t
            bc2 = 1 - b2**self.t
            for i in range(count):
                self.mW[i].mul_(b1).add_(grads_w[i], alpha=1 - b1)
                self.vW[i].mul_(b2).addcmul_(grads_w[i], grads_w[i], value=1 - b2)
                self.mb[i].mul_(b1).add_(grads_b[i], alpha=1 - b1)
                self.vb[i].mul_(b2).addcmul_(grads_b[i], grads_b[i], value=1 - b2)
                self.W[i].sub_(lr * (self.mW[i] / bc1) / (torch.sqrt(self.vW[i] / bc2) + eps))
                self.b[i].sub_(lr * (self.mb[i] / bc1) / (torch.sqrt(self.vb[i] / bc2) + eps))

    def train_epoch(self, data, lr: float, batch_size: int, rng, progress=None) -> tuple[float, float]:
        """One pass over a Dataset. Same shuffle and loss as `train.train_dataset`."""
        count = len(data)
        order = list(range(count))
        rng.shuffle(order)
        offsets = data.offsets.astype(np.int64)
        total_ce = torch.zeros((), device=self.device)
        total_acc = torch.zeros((), device=self.device)
        total_weight = 0.0
        seen = 0
        batches = max(1, (count + batch_size - 1) // batch_size)
        for batch_index, start in enumerate(range(0, count, batch_size)):
            ids = np.asarray(order[start : start + batch_size], dtype=np.int64)
            begins = offsets[ids]
            sizes = offsets[ids + 1] - begins
            total = int(sizes.sum())
            if total == 0:
                continue
            seg = np.repeat(np.arange(len(ids)), sizes)
            firsts = np.repeat(np.cumsum(sizes) - sizes, sizes)
            col = np.arange(total) - firsts
            rows = np.repeat(begins, sizes) + col
            packed = np.empty((total, data.states.shape[1] + data.actions.shape[1]), dtype=np.float32)
            packed[:, :STATE_DIM] = data.states[ids][seg]
            packed[:, STATE_DIM:] = data.actions[rows]

            x = torch.from_numpy(packed).to(self.device, non_blocking=True)
            seg_t = torch.from_numpy(seg).to(self.device)
            col_t = torch.from_numpy(col).to(self.device)
            labels = torch.from_numpy(data.labels[ids].astype(np.int64)).to(self.device)
            batch_weights = data.weights[ids].astype(np.float32)
            weight_sum = float(batch_weights.sum())
            weights = torch.from_numpy(batch_weights).to(self.device)

            values = self._forward(x)[:, 0]
            dense = values.new_full((len(ids), int(sizes.max())), float("-inf"))
            dense = dense.index_put((seg_t, col_t), values)
            logp = torch.log_softmax(dense, dim=1)
            ce = -logp[torch.arange(len(ids), device=self.device), labels]
            loss = (ce * weights).sum()
            grads = torch.autograd.grad(loss, self.W + self.b)
            self._step(list(grads), weight_sum, lr)
            total_weight += weight_sum

            total_ce += loss.detach()
            total_acc += (dense.detach().argmax(dim=1) == labels).sum()
            seen += len(ids)
            if progress is not None:
                progress(batch_index + 1, batches)
        if seen == 0:
            return 0.0, 0.0
        return float(total_ce.item()) / max(total_weight, 1e-8), float(total_acc.item()) / seen

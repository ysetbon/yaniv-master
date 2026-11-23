"""Small MLP trained in float32 and exported for the browser."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from .features import INPUT_DIM


class MLP:
    def __init__(self, dims: list[int], seed: int = 7):
        rng = np.random.default_rng(seed)
        self.dims = list(dims)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.mW: list[np.ndarray] = []
        self.vW: list[np.ndarray] = []
        self.mb: list[np.ndarray] = []
        self.vb: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / dims[i])
            if i == len(dims) - 2:
                scale *= 0.01
            weight = rng.standard_normal((dims[i], dims[i + 1]), dtype=np.float32) * np.float32(scale)
            bias = np.zeros(dims[i + 1], dtype=np.float32)
            self.W.append(weight)
            self.b.append(bias)
            self.mW.append(np.zeros_like(weight))
            self.vW.append(np.zeros_like(weight))
            self.mb.append(np.zeros_like(bias))
            self.vb.append(np.zeros_like(bias))
        self.t = 0
        self._acts: list[np.ndarray] = []
        self._pres: list[np.ndarray] = []
        self.gW: list[np.ndarray] = [np.zeros_like(w) for w in self.W]
        self.gb: list[np.ndarray] = [np.zeros_like(b) for b in self.b]

    def forward(self, x: np.ndarray, cache: bool = False) -> np.ndarray:
        h = x.astype(np.float32, copy=False)
        acts = [h]
        pres: list[np.ndarray] = []
        last = len(self.W) - 1
        for i, (weight, bias) in enumerate(zip(self.W, self.b)):
            z = h @ weight + bias
            pres.append(z)
            h = z if i == last else np.maximum(z, 0)
            acts.append(h)
        if cache:
            self._acts = acts
            self._pres = pres
        return h

    def clone(self) -> "MLP":
        """A copy of the weights only, for checkpoints and inference."""
        copy = MLP(self.dims, seed=0)
        for i in range(len(self.W)):
            copy.W[i] = self.W[i].copy()
            copy.b[i] = self.b[i].copy()
        return copy

    def zero_grad(self) -> None:
        for grad in self.gW:
            grad.fill(0)
        for grad in self.gb:
            grad.fill(0)

    def backward_acc(self, dy: np.ndarray) -> None:
        """Accumulate dL/dY. Gradients are summed over the batch, not averaged."""
        grad = dy.astype(np.float32, copy=False)
        last = len(self.W) - 1
        for i in range(last, -1, -1):
            if i != last:
                grad = grad * (self._pres[i] > 0)
            incoming = self._acts[i]
            self.gW[i] += incoming.T @ grad
            self.gb[i] += grad.sum(axis=0)
            grad = grad @ self.W[i].T

    def step(self, n_examples: int, lr: float, weight_decay: float = 1e-6, clip: float = 5.0) -> None:
        scale = 1.0 / max(1, n_examples)
        grads_w = [grad * scale for grad in self.gW]
        grads_b = [grad * scale for grad in self.gb]
        if weight_decay:
            for weight, grad in zip(self.W, grads_w):
                grad += weight_decay * weight
        norm_sq = 0.0
        for grad in grads_w:
            norm_sq += float(np.sum(grad * grad))
        for grad in grads_b:
            norm_sq += float(np.sum(grad * grad))
        norm = np.sqrt(norm_sq)
        if norm > clip:
            shrink = np.float32(clip / norm)
            grads_w = [grad * shrink for grad in grads_w]
            grads_b = [grad * shrink for grad in grads_b]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        bc1 = 1 - b1**self.t
        bc2 = 1 - b2**self.t
        lr32 = np.float32(lr)
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * grads_w[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * (grads_w[i] * grads_w[i])
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * grads_b[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * (grads_b[i] * grads_b[i])
            self.W[i] -= lr32 * (self.mW[i] / bc1) / (np.sqrt(self.vW[i] / bc2) + eps)
            self.b[i] -= lr32 * (self.mb[i] / bc1) / (np.sqrt(self.vb[i] / bc2) + eps)


def default_net(seed: int = 7) -> MLP:
    """125 suit-free inputs, two hidden layers of 128: about 33,000 weights.

    In a test copying Gen 2, this played better than 192-wide layers and
    than the older 389-input net, with about a third of their weights.
    """
    return MLP([INPUT_DIM, 128, 128, 1], seed=seed)


def _b64(array: np.ndarray) -> str:
    raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
    return base64.b64encode(raw).decode("ascii")


def _layers(net: MLP) -> list[dict]:
    return [{"in": int(weight.shape[0]), "out": int(weight.shape[1]), "w": _b64(weight), "b": _b64(bias)}
            for weight, bias in zip(net.W, net.b)]


def _mlp(dims: list[int], layers: list[dict]) -> MLP:
    net = MLP(dims, seed=0)
    for layer, spec in zip(range(len(net.W)), layers):
        raw_w = base64.b64decode(spec["w"])
        raw_b = base64.b64decode(spec["b"])
        net.W[layer] = np.frombuffer(raw_w, dtype="<f4").reshape(spec["in"], spec["out"]).copy()
        net.b[layer] = np.frombuffer(raw_b, dtype="<f4").copy()
    return net


def save_net(net, path: Path, meta: dict | None = None) -> None:
    """A plain net as version 1; a decider with expert nets (moe.MoE) as version 2, kind "moe"."""
    payload = net_payload(net)
    if meta:
        payload["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def net_payload(net) -> dict:
    """The JSON the game loads (src/lib/ai/net.ts reads both kinds)."""
    if hasattr(net, "experts"):
        payload = {
            "version": 2,
            "kind": "moe",
            "dims": net.dims,
            "gate": {"dims": net.gate.dims, "layers": _layers(net.gate)},
            "experts": [{"dims": e.dims, "layers": _layers(e)} for e in net.experts],
        }
    else:
        payload = {"version": 1, "dims": net.dims, "layers": _layers(net)}
    return payload


def load_net(path: Path):
    """An MLP, or a moe.MoE for a file saved with a decider and expert nets."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") == "moe":
        from .moe import MoE

        gate = _mlp(payload["gate"]["dims"], payload["gate"]["layers"])
        return MoE(gate, [_mlp(e["dims"], e["layers"]) for e in payload["experts"]])
    return _mlp(payload["dims"], payload["layers"])

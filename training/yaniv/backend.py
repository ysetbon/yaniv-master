"""Pick where the gradient steps run: a CUDA or Apple GPU through PyTorch, or NumPy."""

from __future__ import annotations

from dataclasses import dataclass

from .network import MLP, default_net

CHOICES = ("auto", "cuda", "mps", "cpu", "numpy")


@dataclass
class Device:
    kind: str  # "cuda", "mps", "cpu" (PyTorch on CPU) or "numpy"
    name: str
    note: str = ""

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")

    def label(self) -> str:
        if self.kind == "numpy":
            return "CPU (NumPy)"
        if self.kind == "cpu":
            return "CPU (PyTorch)"
        return f"GPU {self.name} ({self.kind})"


def _torch():
    """PyTorch, or None and the reason it could not be used."""
    try:
        import torch
    except ImportError:
        return None, "missing"
    except OSError as exc:  # installed, but a DLL or shared library failed to load
        return None, f"{type(exc).__name__}: {exc}"
    return torch, ""


BROKEN_HINT = (
    "PyTorch is installed but failed to load:\n  {error}\n"
    "On Windows this usually means the Microsoft Visual C++ Redistributable (x64) is missing or old: "
    "install the latest from https://aka.ms/vs/17/release/vc_redist.x64.exe and open a new terminal. "
    "If that does not fix it, reinstall PyTorch with `python -m pip uninstall -y torch`, "
    "`python -m pip cache purge`, then the install command from https://pytorch.org/get-started/locally/"
)


def resolve_device(choice: str = "auto") -> Device:
    if choice not in CHOICES:
        raise SystemExit(f"--device must be one of {', '.join(CHOICES)}")
    if choice == "numpy":
        return Device("numpy", "NumPy")
    torch, problem = _torch()
    if torch is None and problem != "missing":
        message = BROKEN_HINT.format(error=problem)
        if choice == "auto":
            return Device("numpy", "NumPy", message.replace("\n  ", " ").replace("\n", " ")
                          + " Training runs on the CPU with NumPy until then.")
        raise SystemExit(message)
    if torch is None:
        if choice == "auto":
            return Device(
                "numpy",
                "NumPy",
                "PyTorch is not installed, so training runs on the CPU. "
                "Install a CUDA build of PyTorch to train on an NVIDIA GPU.",
            )
        raise SystemExit(
            f"--device {choice} needs PyTorch. Install it from https://pytorch.org/get-started/locally/ "
            "and pick the build that matches your GPU."
        )
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if choice == "cuda" or (choice == "auto" and cuda_ok):
        if not cuda_ok:
            built = torch.version.cuda or "none (this is a CPU-only PyTorch build)"
            raise SystemExit(
                "--device cuda was requested but PyTorch cannot see a CUDA GPU.\n"
                f"  PyTorch {torch.__version__}, CUDA build: {built}\n"
                "  Check `nvidia-smi`, then reinstall PyTorch with the CUDA build from "
                "https://pytorch.org/get-started/locally/"
            )
        return Device("cuda", torch.cuda.get_device_name(0))
    if choice == "mps" or (choice == "auto" and mps_ok):
        if not mps_ok:
            raise SystemExit("--device mps was requested but this PyTorch has no Apple GPU (MPS) support.")
        return Device("mps", "Apple GPU")
    if choice == "cpu":
        return Device("cpu", "PyTorch CPU")
    return Device(
        "numpy",
        "NumPy",
        f"PyTorch {torch.__version__} is installed but sees no GPU, so training runs on the CPU with NumPy.",
    )


def build_net(device: Device, seed: int):
    """A fresh net. Both backends start from the same NumPy initialisation."""
    base = default_net(seed)
    if device.kind == "numpy":
        return base
    import torch

    from .torch_network import TorchMLP

    return TorchMLP(base, torch.device(device.kind))


def as_numpy(net) -> MLP:
    """Weights as a NumPy MLP for playing games. Shares memory for a NumPy net."""
    return net if isinstance(net, MLP) or hasattr(net, "experts") else net.to_numpy()

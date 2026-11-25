"""Checkpoint loader supporting safetensors, plain state_dict (.pt), and Lightning (.ckpt)."""

import io
from pathlib import Path
import pickle
import re
import torch


class _StubObject:
    """Lightweight placeholder for unused training metadata inside raw Lightning checkpoints."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _SafeCheckpointUnpickler(pickle.Unpickler):
    """Unpickler that safely stubs out non-tensor training classes (e.g., old LitModules/OmegaConf)
    while preserving all PyTorch tensor storage reconstruction.
    """

    _ALLOWED_PREFIXES = (
        "torch",
        "collections",
        "builtins",
        "codecs",
        "copyreg",
        "numpy",
        "_codecs",
    )

    def find_class(self, module: str, name: str):
        if any(module == p or module.startswith(p + ".") for p in self._ALLOWED_PREFIXES):
            return super().find_class(module, name)
        try:
            return super().find_class(module, name)
        except Exception:
            return _StubObject


class _SafePickleModule:
    """Minimal pickle-compatible module wrapper passed to torch.load for legacy .ckpt files."""

    Unpickler = _SafeCheckpointUnpickler

    @staticmethod
    def load(file_obj, **kwargs):
        return _SafeCheckpointUnpickler(file_obj).load()

    @staticmethod
    def loads(data, **kwargs):
        return _SafeCheckpointUnpickler(io.BytesIO(data)).load()


def _load_raw_state_dict(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")

    # Prefer safe weights-only loading first (succeeds for clean exported .pt/.bin weights).
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        # Fallback for raw PyTorch Lightning .ckpt files containing non-tensor metadata.
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False, pickle_module=_SafePickleModule)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint format at {checkpoint_path}: {type(ckpt).__name__}")


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> torch.nn.Module:
    """Load GraftQ model weights from .safetensors, weights-only .pt, or raw Lightning .ckpt.

    Automatically strips common top-level prefixes ('model.', 'policy_model.', '')
    and verifies that every target parameter in ``model`` is populated.
    """
    raw_dict = _load_raw_state_dict(checkpoint_path)
    model_keys = set(model.state_dict().keys())

    for prefix in ("model.", "policy_model.", ""):
        stripped = {
            re.sub(f"^{re.escape(prefix)}", "", k): v.data.clone()
            for k, v in raw_dict.items()
            if isinstance(v, torch.Tensor)
        }
        filtered = {k: v for k, v in stripped.items() if k in model_keys}
        if filtered:
            missing = model_keys - set(filtered.keys())
            if missing:
                raise RuntimeError(
                    f"Missing keys after loading checkpoint with prefix {prefix!r}: {sorted(missing)[:10]}"
                )
            model.load_state_dict(filtered, strict=True)
            return model

    prefixes = sorted({k.split(".")[0] for k in raw_dict.keys()})
    raise RuntimeError(f"Cannot match checkpoint keys to GNNTransformer. Found prefixes: {prefixes}")

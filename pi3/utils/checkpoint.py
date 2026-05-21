from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPTS = {
    "pi3": PROJECT_ROOT / "ckpts" / "Pi3.safetensors",
    "pi3x": PROJECT_ROOT / "ckpts" / "Pi3X.safetensors",
}


def resolve_checkpoint(model_type, ckpt=None):
    if ckpt is not None:
        return Path(ckpt)
    default = DEFAULT_CKPTS[model_type.lower()]
    if default.is_file():
        return default
    return None


def load_checkpoint_state(ckpt):
    ckpt = Path(ckpt)
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(ckpt), device="cpu")
    return torch.load(str(ckpt), map_location="cpu", weights_only=False)

from contextlib import nullcontext
import importlib.util

import torch


def import_musa_if_available():
    if importlib.util.find_spec("torch_musa") is None:
        return False
    try:
        import torch_musa  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "musa")


def musa_is_available():
    return (
        import_musa_if_available()
        and hasattr(torch, "musa")
        and torch.musa.is_available()
    )


def resolve_device(device=None):
    if device is None:
        device = "auto"
    spec = str(device).lower()
    if spec == "auto":
        if musa_is_available():
            return torch.device("musa")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available() and musa_is_available():
        return torch.device("musa" + spec[len("cuda"):])
    if spec.startswith("musa"):
        import_musa_if_available()
    return torch.device(spec)


def accelerator_is_available(device):
    device = resolve_device(device)
    if device.type == "musa":
        return musa_is_available()
    if device.type == "cuda":
        return torch.cuda.is_available()
    return True


def _backend(device):
    device = resolve_device(device)
    if device.type == "musa":
        import_musa_if_available()
        return getattr(torch, "musa", None)
    if device.type == "cuda":
        return torch.cuda
    return None


def get_amp_dtype(device):
    device = resolve_device(device)
    if device.type == "musa":
        backend = _backend(device)
        if backend is not None and getattr(backend, "is_bf16_supported", lambda: False)():
            return torch.bfloat16
        return torch.float16
    if device.type == "cuda":
        major = torch.cuda.get_device_capability(device.index or 0)[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def autocast(device, dtype=None, enabled=True):
    device = resolve_device(device)
    if dtype is None:
        dtype = get_amp_dtype(device)
    if device.type not in {"cuda", "musa"} or dtype is torch.float32:
        return nullcontext()
    return torch.amp.autocast(device.type, dtype=dtype, enabled=enabled)


def disabled_autocast(_device=None):
    device = resolve_device(_device)
    if device.type not in {"cuda", "musa"}:
        return nullcontext()
    return torch.amp.autocast(device.type, enabled=False)


def empty_cache(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    if backend is not None and hasattr(backend, "empty_cache"):
        backend.empty_cache()


def reset_peak_memory_stats(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    if backend is not None and hasattr(backend, "reset_peak_memory_stats"):
        backend.reset_peak_memory_stats(device)


def synchronize(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    if backend is not None and hasattr(backend, "synchronize"):
        backend.synchronize(device)


def max_memory_allocated(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    if backend is not None and hasattr(backend, "max_memory_allocated"):
        return backend.max_memory_allocated(device)
    return 0


def memory_allocated(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    if backend is not None and hasattr(backend, "memory_allocated"):
        return backend.memory_allocated(device)
    return 0


def get_device_name(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    if backend is not None and hasattr(backend, "get_device_name"):
        return backend.get_device_name(device.index or 0)
    return device.type


def get_total_memory(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    if backend is not None and hasattr(backend, "get_device_properties"):
        return backend.get_device_properties(device.index or 0).total_memory
    return 0


def oom_errors(device=None):
    device = resolve_device(device)
    backend = _backend(device)
    errors = [RuntimeError]
    if backend is not None and hasattr(backend, "OutOfMemoryError"):
        errors.insert(0, backend.OutOfMemoryError)
    return tuple(errors)


def manual_seed_all(seed):
    if musa_is_available():
        torch.musa.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

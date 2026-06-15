"""Hardware metadata and device helpers."""

from __future__ import annotations

import platform

import torch


def get_device(disable_cuda: bool = False) -> torch.device:
    if not disable_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hardware_metadata(device: torch.device | str | None = None) -> dict[str, object]:
    meta: dict[str, object] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device) if device is not None else None,
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
    }
    if torch.cuda.is_available():
        meta.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else index
        meta.update(
            {
                "selected_cuda_index": int(index),
                "selected_gpu_name": torch.cuda.get_device_name(index),
            }
        )
    return meta

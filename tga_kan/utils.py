"""Device selection. Default to GPU (cuda) when available, else CPU.

Pass an explicit string ('cpu' / 'cuda' / 'cuda:1') to override.
"""
from __future__ import annotations


def resolve_device(device: str | None = "auto") -> str:
    import torch
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device

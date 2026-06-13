"""Factory: name -> surrogate instance, given env dims."""
from __future__ import annotations

from .baselines import LinearSurrogate, LIMESurrogate
from .tga_kan import TGAKANSurrogate


def make_surrogate(name: str, act_dim: int, obs_dim: int, **overrides):
    name = name.lower()
    if name == "linear":
        return LinearSurrogate()
    if name == "lime":
        return LIMESurrogate(**{k: v for k, v in overrides.items()
                                if k in ("n_anchors", "kernel_scale", "ridge", "seed")})
    if name in ("tga", "tga-kan", "tga_kan"):
        return TGAKANSurrogate(act_dim=act_dim, obs_dim=obs_dim, **overrides)
    raise KeyError(f"unknown surrogate {name!r}")

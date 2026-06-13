"""Environment registry — the ONLY env-aware module in the codebase.

Everything downstream (PPO query, surrogates, eval) consumes the metadata
dict and a thunk that builds a fresh env; nothing else imports gym names.
This is what makes the paper's "one architecture across the ladder" claim
testable: only this file knows about specific environments.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any
import json


@dataclass
class EnvSpec:
    id: str                 # gymnasium id
    obs_dim: int
    act_dim: int
    # action bounds (low, high) as python lists for json-serialisability
    act_low: list
    act_high: list
    # paper §6 expectations, carried for reporting / sanity only
    expected_K: str = ""
    note: str = ""

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "EnvSpec":
        with open(path) as f:
            return EnvSpec(**json.load(f))


# The §6 environment ladder. expected_K is documentation, never used as a constraint.
_LADDER: Dict[str, Dict[str, Any]] = {
    "Pendulum-v1":              dict(expected_K="1 (->0 pairs)",  note="should collapse to flat additive"),
    "MountainCarContinuous-v0": dict(expected_K="2",             note="single switching surface, 1 literal"),
    "LunarLanderContinuous-v3": dict(expected_K="3-4",           note="axis, multi-clause"),
    "BipedalWalker-v3":         dict(expected_K="several",       note="oblique multi-clause gate"),
    "Reacher-v5":               dict(expected_K="1-2",           note="many joint x joint pairs"),
}


def make_env(name: str, render_mode: str | None = None) -> Callable[[], Any]:
    """Return a thunk () -> env. Lazy import so the module loads without gym."""
    if name not in _LADDER:
        raise KeyError(f"{name!r} not in ladder: {list(_LADDER)}")

    def _thunk():
        import gymnasium as gym
        return gym.make(name, render_mode=render_mode)

    return _thunk


def probe_spec(name: str) -> EnvSpec:
    """Build an EnvSpec by briefly instantiating the env to read its spaces."""
    import gymnasium as gym
    import numpy as np

    if name not in _LADDER:
        raise KeyError(f"{name!r} not in ladder: {list(_LADDER)}")
    env = gym.make(name)
    try:
        obs_space, act_space = env.observation_space, env.action_space
        assert hasattr(act_space, "low"), "continuous (Box) action space required"
        meta = _LADDER[name]
        return EnvSpec(
            id=name,
            obs_dim=int(np.prod(obs_space.shape)),
            act_dim=int(np.prod(act_space.shape)),
            act_low=np.asarray(act_space.low, dtype=float).ravel().tolist(),
            act_high=np.asarray(act_space.high, dtype=float).ravel().tolist(),
            expected_K=meta["expected_K"],
            note=meta["note"],
        )
    finally:
        env.close()


def ladder() -> list[str]:
    return list(_LADDER)

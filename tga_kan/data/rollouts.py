"""Data collection: oracle rollouts + DAgger aggregation (paper Eq. 7).

The DAgger loop is surrogate-agnostic: it rolls out whatever policy you give
it (oracle for D0, surrogate thereafter), queries pi* for labels on the
visited states, aggregates, and refits. dagger_iters=0 reduces to pure
pointwise behavioural cloning on d_{pi*}  (the §3 ablation).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Dataset:
    S: np.ndarray  # (N, obs_dim)
    A: np.ndarray  # (N, act_dim)  labels = pi*(S)

    def __len__(self):
        return len(self.S)

    def merge(self, other: "Dataset") -> "Dataset":
        return Dataset(np.vstack([self.S, other.S]), np.vstack([self.A, other.A]))


def _clip(a, low, high):
    return np.clip(a, low, high)


def collect_oracle_rollouts(env_thunk, oracle, n_steps, seed=0, deterministic=True):
    """Roll out pi* itself to sample d_{pi*}; returns Dataset (D0)."""
    env = env_thunk()
    S, A = [], []
    obs, _ = env.reset(seed=seed)
    steps = 0
    while steps < n_steps:
        a = oracle.mean_action(obs)
        S.append(np.asarray(obs, dtype=np.float32))
        A.append(np.asarray(a, dtype=np.float32))
        obs, _, term, trunc, _ = env.step(a)
        steps += 1
        if term or trunc:
            obs, _ = env.reset()
    env.close()
    return Dataset(np.asarray(S), np.asarray(A))


def collect_surrogate_rollouts(env_thunk, surrogate, oracle, n_steps, seed=0):
    """Roll out the SURROGATE (sampling d_{a_hat}), label visited states with pi*.

    Also returns consecutive on-policy state pairs for the boundary loss (Eq. 8).
    """
    env = env_thunk()
    act_low = env.action_space.low
    act_high = env.action_space.high
    S, pairs = [], []
    obs, _ = env.reset(seed=seed)
    prev = None
    steps = 0
    while steps < n_steps:
        a = surrogate.predict(np.asarray(obs, dtype=np.float32)[None, :])[0]
        a = _clip(a, act_low, act_high)
        S.append(np.asarray(obs, dtype=np.float32))
        if prev is not None:
            pairs.append((prev, np.asarray(obs, dtype=np.float32)))
        prev = np.asarray(obs, dtype=np.float32)
        obs, _, term, trunc, _ = env.step(a)
        steps += 1
        if term or trunc:
            obs, _ = env.reset()
            prev = None
    env.close()
    S = np.asarray(S)
    A = oracle.mean_action(S)               # query pi* on the surrogate's own occupancy
    transitions = (np.asarray([p[0] for p in pairs]),
                   np.asarray([p[1] for p in pairs])) if pairs else (
                   np.zeros((0, S.shape[1]), np.float32),
                   np.zeros((0, S.shape[1]), np.float32))
    return Dataset(S, A), transitions


def dagger(
    env_thunk, oracle, surrogate, *,
    n_steps_per_iter=20_000,
    dagger_iters=3,
    seed=0,
    verbose=True,
):
    """Paper Eq. 7. dagger_iters=0 => pure pointwise BC on d_{pi*} (ablation)."""
    data = collect_oracle_rollouts(env_thunk, oracle, n_steps_per_iter, seed=seed)
    transitions = (np.zeros((0, data.S.shape[1]), np.float32),
                   np.zeros((0, data.S.shape[1]), np.float32))

    # initial fit on D0 (no boundary transitions yet)
    surrogate.fit(data.S, data.A, transitions=transitions)
    if verbose:
        print(f"[dagger] iter 0  |D|={len(data)}  (pointwise BC on d_pi*)")

    for t in range(1, dagger_iters + 1):
        new, transitions = collect_surrogate_rollouts(
            env_thunk, surrogate, oracle, n_steps_per_iter, seed=seed + t
        )
        data = data.merge(new)
        surrogate.fit(data.S, data.A, transitions=transitions)
        if verbose:
            print(f"[dagger] iter {t}  |D|={len(data)}  "
                  f"#boundary-pairs={len(transitions[0])}")
    return surrogate, data

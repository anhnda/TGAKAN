"""PPO training + the policy oracle.

The ONLY coupling point between PPO and the surrogate side is `PolicyOracle`,
which exposes pi*(s) -> mean action as a batched, numpy-in/numpy-out callable.
Swap SB3 for any other policy by reimplementing this class.
"""
from __future__ import annotations

import os
import numpy as np


class PolicyOracle:
    """Black-box pi*: S -> R^D, the deterministic mean of the Gaussian policy.

    Wraps an SB3 model (+ optional VecNormalize stats). Callers see only
    numpy arrays and never touch torch or SB3 internals.
    """

    def __init__(self, model, vecnorm=None):
        self.model = model
        self.vecnorm = vecnorm  # VecNormalize (for obs normalisation) or None

    # -- construction -------------------------------------------------------
    @classmethod
    def load(cls, model_path: str, vecnorm_path: str | None = None):
        from stable_baselines3 import PPO
        from utils import resolve_device
        model = PPO.load(model_path, device=resolve_device("auto"))
        vecnorm = None
        if vecnorm_path and os.path.exists(vecnorm_path):
            from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
            # Loaded purely to reuse obs_rms; we only normalise observations.
            vecnorm = VecNormalize.load(
                vecnorm_path, DummyVecEnv([lambda: _ObsStub(model.observation_space)])
            )
            vecnorm.training = False
            vecnorm.norm_reward = False
        return cls(model, vecnorm)

    # -- the oracle ---------------------------------------------------------
    def _norm_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.vecnorm is not None:
            return self.vecnorm.normalize_obs(obs)
        return obs

    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        """obs: (N, obs_dim) -> (N, act_dim) deterministic mean actions."""
        import torch
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        obs_n = self._norm_obs(obs)
        obs_t = self.model.policy.obs_to_tensor(obs_n)[0]
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            mean = dist.distribution.mean.cpu().numpy()
        return mean[0] if single else mean

    # convenience alias so the oracle is itself a callable pi*
    __call__ = mean_action


class _ObsStub:
    """Minimal env exposing only the spaces VecNormalize.load needs."""
    def __init__(self, observation_space):
        import gymnasium as gym
        self.observation_space = observation_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.metadata = {}
        self.render_mode = None

    def reset(self, *a, **k):
        return self.observation_space.sample(), {}

    def step(self, a):
        return self.observation_space.sample(), 0.0, False, False, {}

    def close(self):
        pass


def train_ppo(
    env_name: str,
    out_dir: str,
    total_timesteps: int = 300_000,
    n_envs: int = 4,
    normalize_obs: bool = True,
    seed: int = 0,
    ppo_kwargs: dict | None = None,
):
    """Train PPO on env_name, save model.zip (+ vecnormalize.pkl) and meta.json."""
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, VecNormalize,
    )
    from stable_baselines3.common.env_util import make_vec_env

    from envs.registry import probe_spec
    from utils import resolve_device

    os.makedirs(out_dir, exist_ok=True)
    spec = probe_spec(env_name)
    spec.to_json(os.path.join(out_dir, "meta.json"))

    venv = make_vec_env(env_name, n_envs=n_envs, seed=seed)
    if normalize_obs:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ppo_kwargs = ppo_kwargs or {}
    model = PPO("MlpPolicy", venv, seed=seed, verbose=1,
                device=resolve_device("auto"), **ppo_kwargs)
    model.learn(total_timesteps=total_timesteps)

    model.save(os.path.join(out_dir, "model.zip"))
    if normalize_obs:
        venv.save(os.path.join(out_dir, "vecnormalize.pkl"))
    venv.close()
    return out_dir

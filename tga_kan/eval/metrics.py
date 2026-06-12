"""Evaluation: pointwise action MSE (secondary) + closed-loop return gap (headline).

Return gap |J(pi*) - J(a_hat)| is the paper's headline metric (§6). We estimate
J by rolling each policy in the real env and averaging episodic return.
"""
from __future__ import annotations

import numpy as np


def pointwise_mse(surrogate, S, A):
    pred = surrogate.predict(S)
    return float(((pred - A) ** 2).sum(-1).mean())


def _episodic_return(env_thunk, act_fn, n_episodes, max_steps, seed):
    env = env_thunk()
    low, high = env.action_space.low, env.action_space.high
    returns = []
    for e in range(n_episodes):
        obs, _ = env.reset(seed=seed + e)
        ret, done, t = 0.0, False, 0
        while not done and t < max_steps:
            a = np.clip(act_fn(np.asarray(obs, np.float32)[None, :])[0], low, high)
            obs, r, term, trunc, _ = env.step(a)
            ret += r; done = term or trunc; t += 1
        returns.append(ret)
    env.close()
    return np.array(returns)


def _rollout_stats(env_thunk, act_fn, n_episodes, max_steps, seed):
    """Per-episode return + episode length + terminated-vs-truncated flag.

    `terminated` True means the env reached a terminal state on its own (e.g.
    LunarLander landed or crashed); `truncated` means the time limit hit. Used
    so success can be defined by a return threshold and/or natural termination.
    """
    env = env_thunk()
    low, high = env.action_space.low, env.action_space.high
    returns, lengths, terminated = [], [], []
    for e in range(n_episodes):
        obs, _ = env.reset(seed=seed + e)
        ret, done, t, term = 0.0, False, 0, False
        while not done and t < max_steps:
            a = np.clip(act_fn(np.asarray(obs, np.float32)[None, :])[0], low, high)
            obs, r, term, trunc, _ = env.step(a)
            ret += r; done = term or trunc; t += 1
        returns.append(ret); lengths.append(t); terminated.append(bool(term))
    env.close()
    return np.array(returns), np.array(lengths), np.array(terminated)


def success_rate(env_thunk, act_fn, *, n_episodes=100, max_steps=1000,
                 seed=0, threshold=200.0):
    """Fraction of episodes with return >= threshold.

    Default threshold 200 is the Gymnasium 'solved' bar for LunarLander
    (successful landing). For other envs pass an appropriate threshold.
    """
    R, L, term = _rollout_stats(env_thunk, act_fn, n_episodes, max_steps, seed)
    success = R >= threshold
    return {
        "n_episodes": int(n_episodes),
        "threshold": float(threshold),
        "success_rate": float(success.mean()),
        "n_success": int(success.sum()),
        "return_mean": float(R.mean()), "return_std": float(R.std()),
        "return_min": float(R.min()), "return_max": float(R.max()),
        "ep_len_mean": float(L.mean()),
        "terminated_rate": float(term.mean()),
    }


def return_gap(env_thunk, oracle, surrogate, *,
               n_episodes=20, max_steps=1000, seed=0):
    J_star = _episodic_return(env_thunk, oracle.mean_action, n_episodes, max_steps, seed)
    J_hat = _episodic_return(env_thunk, surrogate.predict, n_episodes, max_steps, seed)
    return {
        "J_star_mean": float(J_star.mean()), "J_star_std": float(J_star.std()),
        "J_hat_mean": float(J_hat.mean()), "J_hat_std": float(J_hat.std()),
        "return_gap": float(abs(J_star.mean() - J_hat.mean())),
    }


def full_eval(env_thunk, oracle, surrogate, S_test, A_test, **kw):
    out = {"pointwise_mse": pointwise_mse(surrogate, S_test, A_test)}
    out.update(return_gap(env_thunk, oracle, surrogate, **kw))
    out["explain"] = surrogate.explain()
    return out

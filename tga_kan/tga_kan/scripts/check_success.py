#!/usr/bin/env python
"""Đo success rate của PPO oracle và (các) surrogate trên cùng env.

Success = episode có return >= threshold. Mặc định 200 cho LunarLander (ngưỡng
'solved' của Gymnasium). Env khác phải tự truyền --threshold.

Ví dụ:
    # chỉ oracle
    python scripts/check_success.py --run runs/lunarlander --episodes 100

    # oracle + TGA-KAN (fit qua DAgger rồi đo)
    python scripts/check_success.py --run runs/lunarlander --surrogate tga \
        --dagger-iters 3 --episodes 100

    # so nhiều surrogate một lượt
    python scripts/check_success.py --run runs/lunarlander \
        --surrogate tga linear lime --dagger-iters 3 --episodes 100
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.registry import EnvSpec, make_env
from ppo.oracle import PolicyOracle
from data.rollouts import dagger
from surrogates.factory import make_surrogate
from eval.metrics import success_rate

# ngưỡng 'solved' mặc định theo env (Gymnasium). None = phải truyền tay.
_DEFAULT_THRESHOLD = {
    "LunarLanderContinuous-v3": 200.0,
    "BipedalWalker-v3": 300.0,
    "Pendulum-v1": None,                  # reward âm, không có 'solved' chuẩn
    "MountainCarContinuous-v0": 90.0,
    "Reacher-v5": None,
}


def _fmt(row):
    return (f"{row['name']:>10s} | "
            f"succ {row['success_rate']*100:5.1f}% ({row['n_success']}/{row['n_episodes']}) | "
            f"ret {row['return_mean']:8.1f} ± {row['return_std']:5.1f} | "
            f"[{row['return_min']:7.1f}, {row['return_max']:7.1f}] | "
            f"len {row['ep_len_mean']:6.1f} | term {row['terminated_rate']*100:4.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir chứa model.zip + meta.json")
    ap.add_argument("--surrogate", nargs="*", default=[],
                    help="0+ surrogate: tga / linear / lime")
    ap.add_argument("--threshold", type=float, default=None,
                    help="ngưỡng success; mặc định suy theo env")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--steps-per-iter", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--axis-only", dest="oblique", action="store_false", default=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = EnvSpec.from_json(os.path.join(args.run, "meta.json"))
    thr = args.threshold
    if thr is None:
        thr = _DEFAULT_THRESHOLD.get(spec.id)
        if thr is None:
            ap.error(f"{spec.id} không có ngưỡng mặc định — truyền --threshold")
    print(f"env={spec.id}  threshold={thr}  episodes={args.episodes}\n")

    vp = os.path.join(args.run, "vecnormalize.pkl")
    oracle = PolicyOracle.load(os.path.join(args.run, "model.zip"),
                               vp if os.path.exists(vp) else None, env_id=spec.id)
    env_thunk = make_env(spec.id)

    rows = []

    # 1) PPO oracle
    s = success_rate(env_thunk, oracle.mean_action,
                     n_episodes=args.episodes, max_steps=args.max_steps,
                     seed=args.eval_seed, threshold=thr)
    s["name"] = "PPO"
    rows.append(s)
    print(_fmt(s))

    # 2) từng surrogate
    for name in args.surrogate:
        overrides = {}
        if name.lower() in ("tga", "tga-kan", "tga_kan"):
            overrides = dict(K=args.K, oblique=args.oblique,
                             epochs=args.epochs, seed=args.seed)
        sur = make_surrogate(name, spec.act_dim, spec.obs_dim, **overrides)
        sur, _ = dagger(env_thunk, oracle, sur,
                        n_steps_per_iter=args.steps_per_iter,
                        dagger_iters=args.dagger_iters, seed=args.seed)
        s = success_rate(env_thunk, sur.predict,
                         n_episodes=args.episodes, max_steps=args.max_steps,
                         seed=args.eval_seed, threshold=thr)
        s["name"] = name
        rows.append(s)
        print(_fmt(s))

    out = args.out or os.path.join(args.run, "success.json")
    with open(out, "w") as f:
        json.dump({"env": spec.id, "threshold": thr, "rows": rows}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

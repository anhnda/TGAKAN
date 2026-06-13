#!/usr/bin/env python
"""Load một surrogate ĐÃ LƯU và đánh giá (KHÔNG fit lại).

Dùng sau khi đã train + --save bằng run_surrogate.py. Đo cả success rate và
return gap vs PPO oracle, không tốn thời gian DAgger lại.

Ví dụ:
    # 1) train + lưu
    python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga \
        --dagger-iters 3 --save runs/lunarlander/tga.pt

    # 2) load + đánh giá
    python scripts/eval_surrogate.py --run runs/lunarlander \
        --load runs/lunarlander/tga.pt --surrogate tga --episodes 100
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.registry import EnvSpec, make_env
from ppo.oracle import PolicyOracle
from data.rollouts import collect_oracle_rollouts
from eval.metrics import full_eval, success_rate


def _load_surrogate(name: str, path: str, device: str):
    name = name.lower()
    if name in ("tga", "tga-kan", "tga_kan"):
        from surrogates.tga_kan import TGAKANSurrogate
        return TGAKANSurrogate.load(path, device=device)
    if name == "linear":
        from surrogates.baselines import LinearSurrogate
        return LinearSurrogate.load(path)
    if name == "lime":
        from surrogates.baselines import LIMESurrogate
        return LIMESurrogate.load(path)
    raise KeyError(f"unknown surrogate {name!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir chứa model.zip + meta.json")
    ap.add_argument("--load", required=True, help="path surrogate đã lưu")
    ap.add_argument("--surrogate", default="tga", help="tga / linear / lime")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--test-steps", type=int, default=20_000)
    ap.add_argument("--eval-seed", type=int, default=10_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spec = EnvSpec.from_json(os.path.join(args.run, "meta.json"))
    thr = args.threshold
    if thr is None:
        thr = {"LunarLanderContinuous-v3": 200.0, "BipedalWalker-v3": 300.0,
               "MountainCarContinuous-v0": 90.0}.get(spec.id)
        if thr is None:
            ap.error(f"{spec.id} không có ngưỡng mặc định — truyền --threshold")

    vp = os.path.join(args.run, "vecnormalize.pkl")
    oracle = PolicyOracle.load(os.path.join(args.run, "model.zip"),
                               vp if os.path.exists(vp) else None, env_id=spec.id)
    env_thunk = make_env(spec.id)

    surrogate = _load_surrogate(args.surrogate, args.load, args.device)
    print(f"loaded {args.surrogate} <- {args.load}\n")

    # success rate cả PPO lẫn surrogate
    ppo_s = success_rate(env_thunk, oracle.mean_action,
                         n_episodes=args.episodes, max_steps=args.max_steps,
                         seed=args.eval_seed, threshold=thr)
    sur_s = success_rate(env_thunk, surrogate.predict,
                         n_episodes=args.episodes, max_steps=args.max_steps,
                         seed=args.eval_seed, threshold=thr)

    # pointwise MSE + return gap (trên test rollouts của oracle)
    test = collect_oracle_rollouts(env_thunk, oracle, args.test_steps, seed=999)
    full = full_eval(env_thunk, oracle, surrogate, test.S, test.A,
                     n_episodes=args.episodes, max_steps=args.max_steps,
                     seed=args.eval_seed)

    print(f"env={spec.id}  threshold={thr}  episodes={args.episodes}")
    print(f"  PPO       success={ppo_s['success_rate']*100:5.1f}%  "
          f"ret={ppo_s['return_mean']:.1f}±{ppo_s['return_std']:.1f}")
    print(f"  {args.surrogate:9s} success={sur_s['success_rate']*100:5.1f}%  "
          f"ret={sur_s['return_mean']:.1f}±{sur_s['return_std']:.1f}")
    print(f"  pointwise MSE={full['pointwise_mse']:.5f}  "
          f"return_gap={full['return_gap']:.1f}")

    out = args.out or os.path.join(args.run, f"eval_{args.surrogate}.json")
    def _safe(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return str(o)
    with open(out, "w") as f:
        json.dump({"env": spec.id, "threshold": thr,
                   "ppo_success": ppo_s, "surrogate_success": sur_s,
                   "full_eval": full}, f, indent=2, default=_safe)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

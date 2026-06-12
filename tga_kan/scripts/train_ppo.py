#!/usr/bin/env python
"""Train a PPO oracle on one ladder environment.

Example:
    python scripts/train_ppo.py --env Pendulum-v1 --steps 300000 --out runs/pendulum
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo.oracle import train_ppo
from envs.registry import ladder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=ladder())
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train_ppo(args.env, args.out,
              total_timesteps=args.steps, n_envs=args.n_envs,
              normalize_obs=not args.no_normalize, seed=args.seed)
    print(f"saved -> {args.out}/model.zip")


if __name__ == "__main__":
    main()

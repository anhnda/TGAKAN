#!/usr/bin/env python
"""Fit a surrogate to a trained PPO oracle and evaluate it.

Examples:
    # paper-faithful TGA-KAN with DAgger
    python scripts/run_surrogate.py --run runs/pendulum --surrogate tga --dagger-iters 3

    # ablation: pure pointwise BC (no trajectory constraint)
    python scripts/run_surrogate.py --run runs/pendulum --surrogate tga --dagger-iters 0

    # baselines
    python scripts/run_surrogate.py --run runs/pendulum --surrogate linear
    python scripts/run_surrogate.py --run runs/pendulum --surrogate lime
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from envs.registry import EnvSpec, make_env
from ppo.oracle import PolicyOracle
from data.rollouts import dagger, collect_oracle_rollouts
from surrogates.factory import make_surrogate
from eval.metrics import full_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with model.zip + meta.json")
    ap.add_argument("--surrogate", default="tga")
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--steps-per-iter", type=int, default=20_000)
    ap.add_argument("--test-steps", type=int, default=20_000)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--oblique", action="store_true", default=True)
    ap.add_argument("--axis-only", dest="oblique", action="store_false")
    # regularizer weights (default None -> use TGAKANSurrogate defaults)
    ap.add_argument("--lam-tr", dest="lam_tr", type=float, default=None,
                    help="boundary/trajectory weight (Eq. 8)")
    ap.add_argument("--lam-g", dest="lam_g", type=float, default=None,
                    help="MDL gate weight (drives K down, §3)")
    ap.add_argument("--lam-2", dest="lam_2", type=float, default=None,
                    help="group-lasso weight on second-order surfaces")
    ap.add_argument("--lam-c", dest="lam_c", type=float, default=None,
                    help="ANOVA purge weight (Eq. 5)")
    ap.add_argument("--n-basis", dest="n_basis", type=int, default=None,
                    help="spline basis count per 1-D term")
    ap.add_argument("--max-pairs", dest="max_pairs", type=int, default=None,
                    help="second-order interaction pairs to RESERVE. Default 0 "
                         "(no interactions; first-order/linear consequent only). "
                         "Set >0 to opt in to N candidate pairs; pass -1 for all "
                         "C(n,2). Interactions are off by default because the "
                         "group-lasso under-prunes them on simple policies.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None,
                    help="lưu surrogate đã fit ra path này (.pt cho tga, .pkl cho khác)")
    args = ap.parse_args()

    spec = EnvSpec.from_json(os.path.join(args.run, "meta.json"))
    vp = os.path.join(args.run, "vecnormalize.pkl")
    oracle = PolicyOracle.load(os.path.join(args.run, "model.zip"),
                               vp if os.path.exists(vp) else None,
                               env_id=spec.id)
    env_thunk = make_env(spec.id)

    overrides = {}
    if args.surrogate.lower() in ("tga", "tga-kan", "tga_kan"):
        overrides = dict(K=args.K, oblique=args.oblique, epochs=args.epochs,
                         seed=args.seed)
        # only pass regularizer overrides that were explicitly set, so the
        # surrogate's own defaults stand otherwise
        for name in ("lam_tr", "lam_g", "lam_2", "lam_c", "n_basis", "max_pairs"):
            val = getattr(args, name)
            if val is not None:
                # --max-pairs -1 is the sentinel for "all C(n,2) pairs"
                if name == "max_pairs" and val < 0:
                    val = None
                overrides[name] = val
    surrogate = make_surrogate(args.surrogate, spec.act_dim, spec.obs_dim, **overrides)

    surrogate, _ = dagger(env_thunk, oracle, surrogate,
                          n_steps_per_iter=args.steps_per_iter,
                          dagger_iters=args.dagger_iters, seed=args.seed)

    if args.save:
        surrogate.save(args.save)
        print(f"surrogate saved -> {args.save}")

    test = collect_oracle_rollouts(env_thunk, oracle, args.test_steps, seed=999)
    metrics = full_eval(env_thunk, oracle, surrogate, test.S, test.A,
                        n_episodes=20, max_steps=1000, seed=123)

    # numpy -> json-safe
    def _safe(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return str(o)
    print(json.dumps(metrics, indent=2, default=_safe))

    out = args.out or os.path.join(args.run, f"metrics_{args.surrogate}.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, default=_safe)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
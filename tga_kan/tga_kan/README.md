# TGA-KAN: Gated Additive Surrogates for Continuous-Control Policies

Reference implementation of the draft paper. One architecture, no per-env code changes.

## Install
    pip install -r requirements.txt   # installs torch, sb3, gymnasium[box2d,mujoco]

Device split: **PPO train/inference is forced to CPU** (MLP policy is faster on
CPU; GPU transfer overhead dominates — SB3 issue #1245). The **TGA-KAN surrogate
defaults to GPU** when available (batched spline/gate math benefits), else CPU.
Override the surrogate with device='cpu' in TGAKANSurrogate(...).

## 1. Train a PPO oracle
    python scripts/train_ppo.py --env LunarLanderContinuous-v3 --steps 1000000 --out runs/lunarlander

## 2. Fit + evaluate a surrogate
    # TGA-KAN, paper-faithful (DAgger + boundary loss)
    python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga --dagger-iters 3
    # ablation: pure pointwise BC, no trajectory constraint (Eq.7 off)
    python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga --dagger-iters 0
    # axis-aligned gate instead of oblique
    python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga --axis-only
    # baselines
    python scripts/run_surrogate.py --run runs/lunarlander --surrogate linear
    python scripts/run_surrogate.py --run runs/lunarlander --surrogate lime

## Layout
    envs/registry.py      only env-aware module (the §6 ladder)
    ppo/oracle.py         train_ppo + PolicyOracle (pi* mean-action query)
    data/rollouts.py      oracle/surrogate rollouts + DAgger loop (Eq.7)
    surrogates/
      base.py             BaseSurrogate ABC
      baselines.py        Linear, LIME (locally-weighted)
      spline.py           B-spline basis (KAN terms)
      gate.py             soft-DNF gate (Eq.2-4), MDL, Gumbel selectors
      tga_kan.py          Eq.1 model: SHARED gate + per-output heads; Eq.5/6/8
      factory.py          name -> surrogate
    eval/metrics.py       pointwise MSE + closed-loop return gap (headline)

## Multi-output design
Gate g_k(s) is SHARED across all action coords (regimes = policy-level modes).
Intercepts, first-order splines, second-order set S_{d,k} are per-output.
=> one joint fit over the full action vector, shared-trunk / multi-head. Never
D independent fits. Self-sizing via MDL (gate->K=1), group-lasso (surfaces->0),
DAgger self-deactivation.

## Notes
- PPO forced to CPU (faster for MLP policy); TGA-KAN surrogate uses GPU by
  default, set device='cpu' in TGAKANSurrogate(...) to force CPU.
  LunarLanderContinuous-v3 needs gymnasium[box2d] (Box2D).
- max_pairs caps the 2nd-order candidate reserve for high-dim obs (Reacher).
- LIME is global/rollout-able (kernel-blended local linears), distinct from Linear.

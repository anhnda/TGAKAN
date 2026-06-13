#!/usr/bin/env python
"""Extract human-readable rules from a trained TGA-KAN surrogate.

Reads a saved checkpoint (state_dict + cfg + standardisation stats) and the run's
meta.json, then materialises the three layers of the Eq. (1) decomposition:

  Layer 1  gate regimes  -> soft-DNF clauses as half-space rules (the partition)
  Layer 2  first-order   -> psi^k_{d,i}(s_i) effect curves        (always present)
  Layer 3  second-order  -> psi^k_{d,ij}(s_i,s_j) interaction surfaces (lasso-gated)

Everything is read from the checkpoint alone; NO gym/env and NO refit. Thresholds
are reported in PHYSICAL state units (un-standardised with mu_/sd_), because the
model lives in standardised coordinates and raw taus are meaningless to a reader.

Outputs:
  <out>/rules.md           text report (regimes + per-curve summaries)
  <out>/curves/*.csv       one CSV per (regime,output,feature) first-order curve
  <out>/surfaces/*.csv     one CSV per surviving (regime,output,pair) surface
  <out>/plots/*.png        plots, if --plots and matplotlib present

Usage:
  python scripts/extract_rules.py --run runs/pendulum \
      --load runs/pendulum/tga.pt --out runs/pendulum/rules [--plots]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from surrogates.tga_kan import TGAKANSurrogate


# ---------------------------------------------------------------------------
# obs-component labels: only this dict is env-aware; everything else falls back
# to s[i]. Add rows here as you extend the ladder.
# ---------------------------------------------------------------------------
OBS_LABELS = {
    "Pendulum-v1": ["cos_theta", "sin_theta", "theta_dot"],
    "MountainCarContinuous-v0": ["pos", "vel"],
    "LunarLanderContinuous-v3": ["x", "y", "vx", "vy", "angle", "ang_vel",
                                 "leg_L", "leg_R"],
}
ACT_LABELS = {
    "Pendulum-v1": ["torque"],
    "MountainCarContinuous-v0": ["force"],
    "LunarLanderContinuous-v3": ["main_engine", "lateral_engine"],
}


def _labels(spec_id, n, table, prefix):
    base = table.get(spec_id)
    if base is not None and len(base) == n:
        return list(base)
    return [f"{prefix}[{i}]" for i in range(n)]


def _eval_spline_curve(basis, coef, grid_std):
    """coef:(n_basis,) over the 1-D basis -> curve values on grid_std:(G,)."""
    with torch.no_grad():
        B = basis(torch.tensor(grid_std, dtype=torch.float32))   # (G, n_basis)
        return (B @ coef).cpu().numpy()                           # (G,)


def _eval_spline_surface(basis, coef2, gi_std, gj_std):
    """coef2:(n_basis,n_basis) -> surface on grid gi x gj (standardised)."""
    with torch.no_grad():
        Bi = basis(torch.tensor(gi_std, dtype=torch.float32))     # (G, nb)
        Bj = basis(torch.tensor(gj_std, dtype=torch.float32))     # (G, nb)
        # surface[a,b] = Bi[a] . coef2 . Bj[b]
        return (Bi @ coef2 @ Bj.T).cpu().numpy()                  # (G, G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with meta.json")
    ap.add_argument("--load", required=True, help="saved surrogate .pt")
    ap.add_argument("--out", default=None, help="output dir (default <run>/rules)")
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine; extraction is tiny")
    ap.add_argument("--grid", type=int, default=200, help="curve resolution")
    ap.add_argument("--surf-grid", type=int, default=40, help="surface resolution")
    ap.add_argument("--surface-thresh", type=float, default=1e-2,
                    help="keep a pairwise surface iff ||c2|| / max||c2|| exceeds this")
    ap.add_argument("--mass-thresh", type=float, default=1e-2,
                    help="regime counts as active iff mean gate mass exceeds this")
    ap.add_argument("--sample-states", type=int, default=4096,
                    help="states drawn from N(mu_,sd_) to estimate realized gate mass")
    ap.add_argument("--support-pct", type=float, default=2.0,
                    help="percentile clip for support-aware effect ranking "
                         "(2.0 -> use the 2nd..98th pct of states each regime owns)")
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    out = args.out or os.path.join(args.run, "rules")
    os.makedirs(os.path.join(out, "curves"), exist_ok=True)
    os.makedirs(os.path.join(out, "surfaces"), exist_ok=True)
    if args.plots:
        os.makedirs(os.path.join(out, "plots"), exist_ok=True)

    # -- load spec (for labels/units) and the surrogate ---------------------
    with open(os.path.join(args.run, "meta.json")) as f:
        spec = json.load(f)
    spec_id = spec.get("id", "?")

    surr = TGAKANSurrogate.load(args.load, device=args.device)
    model = surr.model
    model.eval()
    gate = model.gate
    D, n, K = model.D, model.n, model.K

    mu = np.asarray(surr.mu_, np.float64) if surr.mu_ is not None else np.zeros(n)
    sd = np.asarray(surr.sd_, np.float64) if surr.sd_ is not None else np.ones(n)

    obs_lbl = _labels(spec_id, n, OBS_LABELS, "s")
    act_lbl = _labels(spec_id, D, ACT_LABELS, "a")

    def unstd(i, x_std):                 # standardised value on feature i -> physical
        return mu[i] + sd[i] * x_std

    # -- estimate realized gate mass on sampled states ----------------------
    # We have no stored states in the checkpoint, so sample from the fitted input
    # distribution N(mu_,sd_) in physical units, restandardise, push through the
    # hard gate. This matches how active_clause_count() defines "active".
    rng = np.random.default_rng(0)
    S_phys = rng.normal(mu, sd, size=(args.sample_states, n)).astype(np.float32)
    S_std = ((S_phys - mu) / sd).astype(np.float32)
    s_t = torch.tensor(S_std, device=args.device)
    with torch.no_grad():
        g_hard, aux = gate(s_t, hard=True)
        mass = g_hard.mean(0).cpu().numpy()                       # (K,)
        w = gate._w().detach().cpu().numpy()                      # (M, n)
        tau = gate.tau.detach().cpu().numpy()                     # (M,)
        z_hard = (gate.z_logits.detach().cpu().numpy() > 0)       # (K, M)
        clause_on = (torch.sigmoid(gate.clause_logit).detach().cpu().numpy() > 0.5)

    active = [k for k in range(K) if mass[k] > args.mass_thresh]
    alpha = float(gate.alpha.item())

    # -- begin report -------------------------------------------------------
    lines = []
    def w_(s=""): lines.append(s)

    w_(f"# TGA-KAN extracted rules — {spec_id}")
    w_("")
    w_(f"- checkpoint: `{args.load}`")
    w_(f"- nominal K={K}, literals M={gate.M}, "
       f"gate={'oblique' if gate.oblique else 'axis-aligned'}, "
       f"alpha(final)={alpha:.2f}")
    w_(f"- obs_dim={n}, act_dim={D}, candidate pairs={len(model.pairs)}")
    w_(f"- active regimes (mass>{args.mass_thresh}): "
       f"**{len(active)}** of {K}  (masses: "
       + ", ".join(f"{m:.3f}" for m in mass) + ")")
    w_("")
    if len(active) <= 1:
        w_("> Gate collapsed to a single regime: there is **no switching rule**. "
           "The surrogate is a flat additive KAN; rule extraction reduces to the "
           "first-order effect curves below (Layer 2). This is the self-sizing "
           "claim landing on the simplest rung.")
        w_("")

    # ----- Layer 1: regime rules ------------------------------------------
    w_("## Layer 1 — regime rules (soft-DNF gate)")
    w_("")
    if len(active) <= 1:
        w_("_Single regime; no clause to state._")
        w_("")
    else:
        w_("Each regime is the soft-AND of its selected literals; a literal fires "
           "when the half-space `w·s > tau` is entered. Thresholds shown in "
           "physical units.")
        w_("")
        for k in active:
            w_(f"### Regime {k}  (mass={mass[k]:.3f}, "
               f"clause_on={bool(clause_on[k])})")
            lits = np.where(z_hard[k])[0]
            if len(lits) == 0:
                w_("- (no active literals — this clause is the default/else region)")
            for m in lits:
                wm, tm = w[m], tau[m]
                if not gate.oblique:
                    # near one-hot: report as a single-feature threshold
                    i = int(np.argmax(np.abs(wm)))
                    wi = wm[i]
                    # w·s_std > tau  <=>  s_std_i > tau/wi  (flip if wi<0)
                    thr_std = tm / wi
                    thr_phys = unstd(i, thr_std)
                    sense = ">" if wi > 0 else "<"
                    w_(f"- literal {m}: `{obs_lbl[i]} {sense} {thr_phys:.4g}`")
                else:
                    # oblique: report the dense half-space, weights in std coords
                    terms = " + ".join(
                        f"{wm[i]:+.3g}*({obs_lbl[i]}-{mu[i]:.3g})/{sd[i]:.3g}"
                        for i in range(n) if abs(wm[i]) > 1e-3)
                    w_(f"- literal {m}: `{terms} > {tm:.4g}`  (standardised half-space)")
            w_("")

    # ----- Layer 2: first-order effect curves -----------------------------
    w_("## Layer 2 — first-order effect curves  psi^k_{d,i}(s_i)")
    w_("")
    w_("One CSV per (regime, output, feature) in `curves/`. Summary below reports "
       "the curve's value range (peak-to-peak) as an effect-strength proxy.")
    w_("")
    # Fix 1 (sentinel-0): the degree-0 indicator in the B-spline basis is
    # half-open `x < k[1:]`, so the right boundary x=hi falls outside every
    # interval and the partition-of-unity collapses there -> the curve's LAST
    # sample is a hard ~0.0 regardless of the coefficients. That spurious row
    # corrupts every ptp. Drop the trailing grid point: evaluate on the open
    # interval [lo, hi) so no fabricated endpoint reaches the CSV or the ptp.
    grid_std = np.linspace(-3.0, 3.0, args.grid + 1).astype(np.float32)[:-1]

    # Fix 2 (support-aware ranking): rank effects on each regime's ACTUAL data
    # support, not the full [-3,3] design range. A spline can swing wildly in
    # sparse extrapolation regions it never sees on-policy (the regime-2 `x`
    # edge-cliff); those swings are boundary artifacts, not control laws.
    # We approximate per-regime support from the sampled states routed to each
    # regime under the hard gate, in standardised coords, and clip the curve to
    # [p_lo, p_hi] (default 2-98th pct) before computing ptp.
    g_assign = g_hard.argmax(-1).cpu().numpy()                    # (N,) regime id
    S_std_np = S_std                                              # (N, n) standardised
    support_std = {}                                             # k -> (n,2) [lo,hi]
    for k in active:
        sel = S_std_np[g_assign == k]
        if sel.shape[0] >= 8:
            lo_q = np.percentile(sel, args.support_pct, axis=0)
            hi_q = np.percentile(sel, 100.0 - args.support_pct, axis=0)
        else:
            # too few routed states to trust: fall back to full range
            lo_q = np.full(n, -3.0, np.float32)
            hi_q = np.full(n, 3.0, np.float32)
        support_std[k] = np.stack([lo_q, hi_q], axis=1).astype(np.float32)

    for k in active:
        expert = model.experts[k]
        basis = expert.basis
        c1 = expert.c1.detach().cpu()                              # (D, n, nb)
        supp = support_std[k]                                      # (n, 2)
        for d in range(D):
            ranges = []
            for i in range(n):
                curve = _eval_spline_curve(basis, c1[d, i], grid_std)
                # support-aware ptp: only consider the curve where this regime
                # actually has data on feature i.
                in_supp = (grid_std >= supp[i, 0]) & (grid_std <= supp[i, 1])
                if in_supp.any():
                    cvals = curve[in_supp]
                    ptp = float(cvals.max() - cvals.min())
                else:
                    ptp = 0.0
                ranges.append((i, ptp))
                # write CSV in physical x-units (full grid; a `support` column
                # flags which rows are inside this regime's data support)
                fn = os.path.join(out, "curves",
                                  f"reg{k}_{act_lbl[d]}_{obs_lbl[i]}.csv")
                with open(fn, "w", newline="") as f:
                    cw = csv.writer(f)
                    cw.writerow([obs_lbl[i], f"psi_{act_lbl[d]}", "in_support"])
                    for xs, yv, ins in zip(grid_std, curve, in_supp):
                        cw.writerow([f"{unstd(i, xs):.6g}", f"{yv:.6g}",
                                     int(bool(ins))])
            ranges.sort(key=lambda t: -t[1])
            top = ", ".join(f"{obs_lbl[i]} (ptp={p:.3g})" for i, p in ranges[:3])
            w_(f"- regime {k}, output `{act_lbl[d]}`: intercept "
               f"beta={float(expert.beta[d]):+.4g}; strongest effects "
               f"(support-aware): {top}")
    w_("")

    # ----- Layer 3: surviving interaction surfaces ------------------------
    w_("## Layer 3 — second-order surfaces  psi^k_{d,ij}(s_i,s_j)")
    w_("")
    # global max norm for relative thresholding
    all_norms = []
    for k in active:
        e = model.experts[k]
        if e.c2 is not None:
            all_norms.append(e.c2.detach().pow(2).sum(dim=(-1, -2)).sqrt().cpu().numpy())
    max_norm = float(np.max([a.max() for a in all_norms])) if all_norms else 0.0
    if len(model.pairs) == 0:
        w_("_Second-order terms are disabled (max_pairs=0): this is a "
           "first-order / linear-consequent model by construction. The emitted "
           "action is `beta + sum_i psi_i(s_i)` per regime — no interactions._")
        w_("")
        max_norm = 0.0
    elif max_norm == 0.0:
        w_("_All surfaces are exactly zero (pruned by the group-lasso)._")
        w_("")
    else:
        kept_any = False
        # same open-interval fix as the first-order curves (drop x=hi sentinel)
        sg = np.linspace(-3.0, 3.0, args.surf_grid + 1).astype(np.float32)[:-1]
        for k in active:
            e = model.experts[k]
            if e.c2 is None:
                continue
            norms = e.c2.detach().pow(2).sum(dim=(-1, -2)).sqrt().cpu().numpy()  # (D,P)
            basis = e.basis
            for d in range(D):
                for p, (i, j) in enumerate(e.pairs):
                    rel = norms[d, p] / max_norm
                    if rel < args.surface_thresh:
                        continue
                    kept_any = True
                    surf = _eval_spline_surface(basis, e.c2[d, p].detach().cpu(), sg, sg)
                    fn = os.path.join(out, "surfaces",
                                      f"reg{k}_{act_lbl[d]}_{obs_lbl[i]}_x_{obs_lbl[j]}.csv")
                    with open(fn, "w", newline="") as f:
                        cw = csv.writer(f)
                        cw.writerow([f"{obs_lbl[i]}\\{obs_lbl[j]}"]
                                    + [f"{unstd(j, v):.4g}" for v in sg])
                        for a, xs in enumerate(sg):
                            cw.writerow([f"{unstd(i, xs):.4g}"]
                                        + [f"{surf[a, b]:.4g}" for b in range(len(sg))])
                    w_(f"- regime {k}, output `{act_lbl[d]}`: "
                       f"`{obs_lbl[i]} x {obs_lbl[j]}` "
                       f"(rel-norm={rel:.3f}, ptp={surf.max()-surf.min():.3g})")
        if not kept_any:
            w_(f"_All surfaces below relative threshold {args.surface_thresh}: "
               "no genuine pairwise interaction survived the group-lasso._")
        w_("")

    # ----- optional plots --------------------------------------------------
    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for k in active:
                expert = model.experts[k]
                basis = expert.basis
                c1 = expert.c1.detach().cpu()
                for d in range(D):
                    fig, ax = plt.subplots(figsize=(7, 4))
                    for i in range(n):
                        curve = _eval_spline_curve(basis, c1[d, i], grid_std)
                        ax.plot([unstd(i, x) for x in grid_std], curve, label=obs_lbl[i])
                    ax.set_title(f"regime {k} — effect curves for {act_lbl[d]}")
                    ax.set_xlabel("state component (physical units)")
                    ax.set_ylabel(f"psi -> {act_lbl[d]}")
                    ax.legend(fontsize=8)
                    fig.tight_layout()
                    fig.savefig(os.path.join(out, "plots",
                                f"reg{k}_{act_lbl[d]}_curves.png"), dpi=120)
                    plt.close(fig)
            w_("_Plots written to `plots/`._")
        except Exception as ex:        # noqa: BLE001
            w_(f"_Plotting skipped: {ex}_")

    report = os.path.join(out, "rules.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[extract] active regimes: {len(active)}/{K}  masses={np.round(mass,3)}")
    print(f"[extract] wrote {report}")
    print(f"[extract] curves -> {os.path.join(out,'curves')}")
    print(f"[extract] surfaces -> {os.path.join(out,'surfaces')}")


if __name__ == "__main__":
    main()

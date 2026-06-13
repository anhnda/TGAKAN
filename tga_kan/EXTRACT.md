# EXTRACT.md — reading rules out of a TGA-KAN surrogate

A "rule" in TGA-KAN is not one object. It is the two-level decomposition of
Eq. (1): the **gate** gives logical region rules (the soft-DNF part), and within
each surviving region the **KAN expert** gives effect curves and interaction
surfaces. Extraction therefore has three layers, and which ones are non-trivial
depends entirely on what the self-sizing regularizers left switched on.

`scripts/extract_rules.py` reads a saved checkpoint and `<run>/meta.json` and
emits all three layers. It needs **no gym/env and never refits** — everything is
read from the stored `state_dict`, `cfg`, and standardisation stats (`mu_`,
`sd_`). CPU is fine.

```
python scripts/extract_rules.py \
    --run runs/pendulum \
    --load runs/pendulum/tga.pt \
    --out  runs/pendulum/rules \
    [--plots]
```

Outputs under `--out`:

| path             | contents                                                       |
|------------------|----------------------------------------------------------------|
| `rules.md`       | regimes as half-space rules + per-curve / per-surface summary  |
| `curves/*.csv`   | one first-order effect curve per (regime, output, feature)     |
| `surfaces/*.csv` | one surviving interaction surface per (regime, output, pair)   |
| `plots/*.png`    | effect-curve plots per (regime, output), only with `--plots`   |

---

## The one thing that makes thresholds meaningful: un-standardisation

The model lives entirely in standardised coordinates `s_std = (s − mu_)/sd_`
(see `TGAKANSurrogate._standardize`). Splines assume the input lands in
`[lo, hi] = [-3, 3]`, and every gate threshold `tau_m` is a threshold on
`s_std`, not on physical state. A raw `tau` is meaningless to a domain reader.
The script un-standardises before writing anything: a threshold on standardised
feature `i` becomes `mu_[i] + sd_[i] · (tau/w_i)` in physical units, and curve /
surface CSV axes are written in physical units too. If you ever read the model
by hand, do this conversion first.

---

## Layer 1 — regime rules from the gate (the only *logical* rules)

The gate (`SoftDNFGate`) is a relaxed DNF: literals → soft-AND clauses → soft-OR
over clauses, Eqs. (2)–(4). Extraction:

1. **Count what survived.** A regime is "active" iff its realized hard-gate mass
   exceeds a threshold (default `1e-2`), matching `gate.active_clause_count`.
   The checkpoint stores no states, so the script samples from the fitted input
   law `N(mu_, sd_)`, restandardises, and pushes through `hard=True` to estimate
   per-regime mass. If only one regime is active there is **no switching rule**
   and Layer 1 is empty — report the collapsed `K` as the finding.
2. **Read each surviving clause.** Active literals for clause `k` are
   `z_logits[k] > 0` (the hard selector). Each active literal `m` is the
   half-space `w_m · s_std > tau_m`.
   - **Axis-aligned** (`oblique=False`): `_w()` is near one-hot, so the literal
     collapses to a single feature: `s_i ⋛ threshold`. Clean and directly
     readable. Prefer this when auditability is the whole point.
   - **Oblique** (`oblique=True`, the default): `w_m` is dense — a diagonal
     half-space. Still a linear rule, but the script reports the full weight
     vector rather than pretending it is one feature. Don't call an oblique
     literal a single-feature rule.
3. A regime is the AND of its literals; the gate is the OR over regimes. That is
   your readable partition.

`clause_logit` is the explicit per-regime off-switch; `clause_on` in the report
is `sigmoid(clause_logit) > 0.5`. Note a regime can still hold mass with a
negative logit (softmax always sums to 1), which is why activity is judged by
realized mass, not by the off-switch alone — exactly the distinction made in
`active_clause_count`.

---

## Layer 2 — first-order effect curves (always present)

For expert `k`, output `d`, feature `i`:

```
psi^k_{d,i}(s_i) = sum_b  c1[d, i, b] · B_b(s_i)
```

The script grids `s_std` over `[-3, 3]`, runs it through the expert's
`SplineBasis` to get `B` of shape `(grid, n_basis)`, computes `B @ c1[d, i]`,
and writes the curve against the physical axis `mu_[i] + sd_[i]·s_std`. Each
`(d, i)` curve is one auditable 1-D effect — how action `d` depends on state
component `i` inside regime `k`. The report ranks features per output by
peak-to-peak range as an effect-strength proxy, and lists each expert's
intercept `beta_{d,k}`.

These exist in every regime regardless of sizing. On a collapsed (`K=1`) model
they **are** the whole rule set.

---

## Layer 3 — interaction surfaces (only the ones that survived the lasso)

For a surviving pair `(i, j)`:

```
psi^k_{d,ij}(s_i, s_j) = sum_{a,b}  c2[d, p, a, b] · B_a(s_i) · B_b(s_j)
```

The group-lasso (`KANExpert.group_lasso`, the `lambda_2` term) zeroes whole
surfaces, so **filter by surviving norm first**. The script computes
`||c2[d,p]||` per (output, pair), normalises by the global max, and keeps only
pairs above a relative threshold (default `1e-2`); the rest are off and are not
reported as interactions. Because of the ANOVA purge (Eq. 5), surviving surfaces
carry *pure* interaction — main-effect mass has been removed — so a near-flat
surface means "no genuine coupling" and a structured one is a real `s_i × s_j`
interaction not already explained by the curves. Each kept surface is written as
a 2-D CSV grid in physical units.

---

## Caveats to carry into any writeup

- **Hardening jumps.** `predict()` uses `hard_gate=True`, but the extracted
  curves are continuous. At a hard boundary the realized action can jump, so for
  `K > 1` also report the boundary set and check `boundary_loss` magnitude
  before claiming the curves are globally faithful. With `K = 1` this is moot.
- **Oblique ≠ auditable.** Oblique literals are linear but not axis-readable;
  calling them "rules" is fair only if you report the weight vector. For
  presentation-grade single-feature rules, retrain with `oblique=False`.
- **Representation, not optimisation.** Proposition 1 guarantees a configuration
  *exists*; the non-convex annealed gate does not guarantee training found it.
  The extracted regime count / sparsity is empirical evidence, not a theorem.
- **Sampled mass.** Active-regime mass is estimated from `N(mu_, sd_)` samples,
  not stored on-policy states (the checkpoint holds none). For a partition that
  hugs the policy's occupancy tightly, pass real rollout states if you have them
  and prefer those numbers; the Gaussian estimate is a reasonable default.

---

## Expected result on the Pendulum checkpoint

The reported run collapsed to `K → 1` with `#pairwise → 0` (MSE 0.00996,
return_gap 2.8). Extraction should therefore show: **one active regime, no gate
rule, three first-order torque curves** (over `cos_theta`, `sin_theta`,
`theta_dot`), and **zero surviving surfaces**. The scientific content is the
*absence* of structure — `K=4`, oblique gate, and 64 candidate pairs were
available and switched off, which is the self-sizing claim landing on the
simplest rung of the §6 ladder.

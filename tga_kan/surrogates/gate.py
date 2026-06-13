"""Soft-DNF gate, paper Eqs. (2)-(4).

literal  l_m(s) = sigma( alpha (w_m^T phi(s) - tau_m) )         (2)
clause   c_k(s) = prod_m l_m(s)^{z_km}                          (3)   (soft AND)
gate     g_k(s) = softmax_k( gamma log c_k(s) )                (4)   (soft OR)

- alpha annealed soft->hard during training (set_alpha).
- axis-aligned: w_m = e_i (one feature per literal), directly readable.
  oblique: w_m dense, learns diagonal boundaries.
- z_km in {0,1} relaxed via Gumbel-sigmoid (concrete) -> learnable clause structure.
- MDL penalty on active clauses + literal count drives gate -> K=1 (§3).

The gate is SHARED across all output coordinates: regimes are policy-level
behavioural modes, not per-action. This is the multi-output design decision.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDNFGate(nn.Module):
    def __init__(self, obs_dim, K=4, n_literals=8, oblique=True,
                 gamma=1.0, alpha0=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.obs_dim = obs_dim
        self.K = K
        self.M = n_literals
        self.oblique = oblique
        self.gamma = gamma
        self.register_buffer("alpha", torch.tensor(float(alpha0)))

        if oblique:
            self.w = nn.Parameter(torch.randn(self.M, obs_dim, generator=g) * 0.5)
        else:
            # axis-aligned: each literal reads one feature; learn which via a
            # softmax selector over features, plus a sign/scale.
            self.feat_logits = nn.Parameter(torch.randn(self.M, obs_dim, generator=g) * 0.1)
            self.w_scale = nn.Parameter(torch.ones(self.M))
        self.tau = nn.Parameter(torch.randn(self.M, generator=g) * 0.5)

        # clause-literal selector logits z_km (relaxed Bernoulli / concrete)
        self.z_logits = nn.Parameter(torch.randn(K, self.M, generator=g) * 0.1 + 0.5)
        self.z_temp = 0.5

        # per-clause activation logit: an explicit off-switch for each regime.
        # logsigmoid(clause_logit) biases log_c toward -inf, removing a clause
        # from the softmax. Without this, an empty clause has log_c=0 (the max
        # achievable, since log l_m <= 0) and would DOMINATE the gate, so MDL
        # could never collapse K. Init positive so all clauses start "on".
        self.clause_logit = nn.Parameter(torch.full((K,), 2.0))

    # -- annealing ----------------------------------------------------------
    def set_alpha(self, alpha: float):
        self.alpha.fill_(float(alpha))

    # -- literal directions -------------------------------------------------
    def _w(self):
        if self.oblique:
            return self.w                                  # (M, obs_dim)
        sel = F.softmax(self.feat_logits, dim=-1)          # near one-hot
        return sel * self.w_scale[:, None]

    # -- selector sampling --------------------------------------------------
    def _z(self, hard=False):
        if hard:
            return (self.z_logits > 0).float()
        if self.training:
            u = torch.rand_like(self.z_logits).clamp(1e-6, 1 - 1e-6)
            logistic = torch.log(u) - torch.log(1 - u)
            return torch.sigmoid((self.z_logits + logistic) / self.z_temp)
        return torch.sigmoid(self.z_logits / self.z_temp)

    # -- forward ------------------------------------------------------------
    def forward(self, s, hard=False):
        """s:(N, obs_dim) -> g:(N, K), and aux dict for penalties/inspection."""
        w = self._w()                                      # (M, obs_dim)
        pre = self.alpha * (s @ w.T - self.tau)            # (N, M)
        lit = torch.sigmoid(pre).clamp(1e-6, 1 - 1e-6)     # (N, M)  literals
        z = self._z(hard=hard)                             # (K, M)
        # log clause: sum_m z_km log l_m
        log_c = z[None] * torch.log(lit)[:, None, :]       # (N, K, M)
        log_c = log_c.sum(-1)                              # (N, K)
        # off-switch: dead clauses (clause_logit -> -inf) get log_c -> -inf and
        # drop out of the softmax. logsigmoid(.) in (-inf, 0].
        log_c = log_c + F.logsigmoid(self.clause_logit)[None]
        g = F.softmax(self.gamma * log_c, dim=-1)          # (N, K)
        aux = {"literals": lit, "z": z, "log_c": log_c}
        return g, aux

    # -- penalties ----------------------------------------------------------
    def mdl_penalty(self, s=None):
        """MDL cost on the number of regimes that carry data mass.

        Penalizing sum(sigmoid(clause_logit)) alone pushes every clause down
        UNIFORMLY, which shifts the softmax by a constant and changes nothing —
        mass stays spread and K never collapses (it just looked like "0" once
        all logits went negative). Instead we penalize the realized per-regime
        mass p_k = mean_s g_k(s).

        Crucially the FIRST regime is free: a surrogate always needs >=1 regime,
        so charging it makes the penalty fight fidelity even on a 1-regime
        policy. We charge only the EXCESS beyond one. A 2nd regime survives iff
        the fidelity drop it buys exceeds lam_g * (its marginal MDL cost); this
        is what lets the same lam_g keep K=1 on Pendulum yet K=2 on MountainCar,
        instead of crushing every env to 1.
        """
        a = torch.sigmoid(self.clause_logit)              # (K,)
        z = torch.sigmoid(self.z_logits / self.z_temp)    # (K, M)
        literal_count = (a[:, None] * z).sum()
        if s is None:
            return torch.relu(a.sum() - 1.0) + 0.1 * literal_count
        g, _ = self.forward(s, hard=False)                # (N, K), differentiable
        p = g.mean(0)                                      # (K,) regime mass
        soft_active = torch.sigmoid(50.0 * (p - 0.02)).sum()
        excess = torch.relu(soft_active - 1.0)            # first regime free
        return excess + 0.1 * literal_count

    @torch.no_grad()
    def active_clause_count(self, s=None, thresh=1e-2):
        """Number of regimes that actually carry gate mass over the data.

        Counted purely from realized gate mass (deterministic, hard=True). The
        off-switch already removes a dead clause from the softmax, so a regime
        that's truly off contributes ~0 mass and isn't counted. We do NOT also
        require sigmoid(clause_logit)>0.5: the softmax always sums to 1, so the
        surviving regime keeps its mass even when every clause_logit is negative
        — gating the count on the off-switch too would spuriously report 0.
        """
        if s is None:
            return int((torch.sigmoid(self.clause_logit) > 0.5).sum().item())
        was_training = self.training
        self.eval()
        g, _ = self.forward(s, hard=True)                  # no Gumbel noise
        if was_training:
            self.train()
        return int((g.mean(0) > thresh).sum().item())

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
    def mdl_penalty(self):
        """MDL-style: number of active clauses + their active literal count.

        Penalty is gated by per-clause activation a_k = sigmoid(clause_logit),
        which is the actual off-switch in forward(). Driving a_k -> 0 removes a
        clause from the softmax AND zeroes its literal cost, so unused regimes
        collapse and K -> 1 when no switching is needed.
        """
        a = torch.sigmoid(self.clause_logit)              # (K,) per-clause "on"
        z = torch.sigmoid(self.z_logits / self.z_temp)    # (K, M)
        active_clauses = a.sum()
        literal_count = (a[:, None] * z).sum()            # literals in live clauses
        return active_clauses + 0.1 * literal_count

    @torch.no_grad()
    def active_clause_count(self, s=None, thresh=1e-2):
        """Number of regimes that actually carry gate mass.

        If states `s` are given, count regimes whose mean gate weight exceeds
        `thresh` (the data-driven, self-sizing notion of K). Otherwise fall back
        to the structural count of switched-on clauses.
        """
        if s is not None:
            g, _ = self.forward(s, hard=False)
            return int((g.mean(0) > thresh).sum().item())
        return int((torch.sigmoid(self.clause_logit) > 0.5).sum().item())

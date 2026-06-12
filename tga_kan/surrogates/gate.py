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
        g = F.softmax(self.gamma * log_c, dim=-1)          # (N, K)
        aux = {"literals": lit, "z": z, "log_c": log_c}
        return g, aux

    # -- penalties ----------------------------------------------------------
    def mdl_penalty(self):
        """MDL-style: number of active clauses + total active literal count.

        Uses soft selector mass so it's differentiable; drives unused clauses
        and literals to zero, collapsing K -> 1 when no switching is needed.
        """
        z = torch.sigmoid(self.z_logits / self.z_temp)     # (K, M)
        literal_count = z.sum()                            # total literals used
        clause_activity = z.sum(-1)                        # (K,) literals per clause
        active_clauses = torch.sigmoid(4.0 * (clause_activity - 0.5)).sum()
        return active_clauses + 0.1 * literal_count

    @torch.no_grad()
    def active_clause_count(self, thresh=0.5):
        z = (torch.sigmoid(self.z_logits / self.z_temp) > thresh).float()
        return int((z.sum(-1) > 0).sum().item())

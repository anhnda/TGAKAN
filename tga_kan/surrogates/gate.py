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

    # -- warm start ---------------------------------------------------------
    @torch.no_grad()
    def warm_start(self, s, seed=0):
        """Fix 3 (init): seed literals/clauses from k-means on the state buffer.

        Random literal directions rarely separate the on-policy state cloud, so
        at init most clauses evaluate near-identically and the gate is degenerate
        before training. We run a light k-means (K clusters) on the standardised
        states and orient each clause toward its cluster: literal m gets a
        direction toward centroid m and a threshold at the centroid's projection,
        and clause k is wired to its own literal. This gives K genuinely distinct
        regimes to start from; training then merges/prunes via MDL rather than
        having to first discover any structure at all.
        """
        s = s.detach()
        N = s.shape[0]
        if N < self.K:
            return
        gcpu = torch.Generator(device="cpu").manual_seed(seed)
        # init centroids = K random distinct states
        idx = torch.randperm(N, generator=gcpu)[: self.K]
        C = s[idx].clone()                                 # (K, obs_dim)
        for _ in range(10):                                # Lloyd iterations
            d2 = torch.cdist(s, C)                          # (N, K)
            assign = d2.argmin(-1)                          # (N,)
            for k in range(self.K):
                m = assign == k
                if m.any():
                    C[k] = s[m].mean(0)
        # one literal per clause (M >= K assumed for the readable axis-aligned
        # case; if M<K we wrap around). Direction = unit vector to centroid,
        # threshold = projection of the centroid (so the half-space opens at it).
        for m in range(self.M):
            k = m % self.K
            v = C[k]
            nv = v.norm().clamp(min=1e-6)
            if self.oblique:
                self.w[m].copy_(v / nv)
            else:
                # axis-aligned: point the selector at the centroid's dominant dim
                self.feat_logits[m].zero_()
                dom = int(v.abs().argmax().item())
                self.feat_logits[m, dom] = 3.0
                self.w_scale[m].fill_(float(torch.sign(v[dom]).item()) or 1.0)
            self.tau[m].fill_(float((v / nv) @ v) if self.oblique
                              else float(v[int(v.abs().argmax())]))
        # wire clause k -> literal (k mod M) on, others off
        self.z_logits.fill_(-2.0)
        for k in range(self.K):
            self.z_logits[k, k % self.M] = 2.0
        self.clause_logit.fill_(2.0)

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
        # Fix 3 (gate-init collapse): sum_m z log l_m is the log of a PRODUCT of
        # sigmoids, so a clause with more active literals has a more negative
        # log_c purely because it multiplies more (<=1) factors — independent of
        # whether those literals fire. At init that makes the softmax near
        # one-hot toward the SHORTEST clause, and MDL then prunes the rest before
        # training can use them (the observed 4->2 collapse at lam_g=0). Divide
        # by the active-literal count so log_c is the MEAN log-literal (geometric
        # mean of the literals): clause score no longer depends on clause length,
        # only on how well its literals are satisfied.
        zcount = z.sum(-1).clamp(min=1.0)                  # (K,) active literals
        log_c = log_c / zcount[None]                       # (N, K)
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

        The FIRST regime is free (we subtract the largest mass), because a
        surrogate always needs >=1 regime. We charge the EXCESS mass carried by
        all other regimes with a smooth, NON-SATURATING cost (see Fix 4 below):
        a 2nd regime survives iff the fidelity drop it buys exceeds lam_g times
        its marginal mass. This is what lets one CV-selected lam_g keep K=1 on
        Pendulum yet K=2 on MountainCar, instead of crushing every env to 1.
        """
        a = torch.sigmoid(self.clause_logit)              # (K,)
        z = torch.sigmoid(self.z_logits / self.z_temp)    # (K, M)
        literal_count = (a[:, None] * z).sum()
        if s is None:
            return torch.relu(a.sum() - 1.0) + 0.1 * literal_count
        g, _ = self.forward(s, hard=False)                # (N, K), differentiable
        p = g.mean(0)                                      # (K,) regime mass
        # Fix 4 (MDL saturation): the old sigmoid(50*(p-0.02)) saturates to ~1
        # for ANY regime above ~2% mass, so its gradient w.r.t. p is ~0 on every
        # healthy regime — it can count regimes but cannot trade a real regime's
        # fidelity against its mass, which is what the paper claims it does.
        # Replace with a smooth, non-saturating cost on the mass carried by every
        # regime EXCEPT the largest (the first regime is always free): sum of all
        # masses minus the top one. Gradient is constant in p (=1 per extra
        # regime, scaled by mass), so a 2nd regime survives iff its fidelity gain
        # exceeds lam_g * its mass — a genuine fidelity/complexity trade.
        top = p.max()
        excess_mass = (p.sum() - top)                     # mass beyond top regime
        return excess_mass + 0.1 * literal_count

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

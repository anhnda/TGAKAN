"""TGA-KAN model, paper Eq. (1).

  a_hat_d(s) = sum_k g_k(s) [ beta_{d,k}
                              + sum_i psi^k_{d,i}(s_i)
                              + sum_{(i,j) in S_{d,k}} psi^k_{d,ij}(s_i,s_j) ]

MULTI-OUTPUT DESIGN (the key structural decision):
  - the gate g_k(s) is SHARED across all D output coordinates (one partition of
    state space into behavioural regimes);
  - everything else is PER-OUTPUT: intercepts beta_{d,k}, first-order splines,
    and the selected second-order set S_{d,k}.
  This is a shared-trunk / multi-head network: gate = trunk, KAN experts = heads.
  The whole thing is ONE joint optimisation over the full action vector
  (fidelity loss ||a_hat(s) - pi*(s)||^2 is over all D coords at once), never
  D independent fits. Outputs are coupled through the shared gate and the joint
  loss, but each gets its own readable effect curves.

Second-order terms carry pure interaction only (ANOVA purge, Eq. 5), and are
gated off by a group-lasso (Eq. 6, lambda_2 term) so S_{d,k} is an outcome of
fitting. Boundary smoothness (Eq. 8) penalises action jumps across regime
boundaries along on-policy transitions.
"""
from __future__ import annotations

import itertools
import numpy as np
import torch
import torch.nn as nn

from .base import BaseSurrogate
from .spline import SplineBasis
from .gate import SoftDNFGate


class KANExpert(nn.Module):
    """One regime's expert for ALL output coords (vectorised over d).

    first-order:  coefficients (D, obs_dim, n_basis)
    second-order: coefficients (D, n_pairs, n_basis, n_basis) over candidate pairs
    intercept:    (D,)
    A per-(d, pair) group-lasso gate selects which surfaces survive -> S_{d,k}.
    """

    def __init__(self, act_dim, obs_dim, pairs, n_basis=12,
                 lo=-3.0, hi=3.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.D = act_dim
        self.n = obs_dim
        self.pairs = pairs                         # list of (i,j) candidate couplings
        self.P = len(pairs)
        self.basis = SplineBasis(n_basis=n_basis, lo=lo, hi=hi)
        nb = self.basis.n_basis

        self.beta = nn.Parameter(torch.zeros(self.D))
        self.c1 = nn.Parameter(torch.randn(self.D, self.n, nb, generator=g) * 0.01)
        if self.P:
            self.c2 = nn.Parameter(torch.randn(self.D, self.P, nb, nb, generator=g) * 0.01)
        else:
            self.register_parameter("c2", None)

    def _phi(self, s):
        # s:(N,n) -> (N, n, nb)
        N, n = s.shape
        return self.basis(s.reshape(-1)).reshape(N, n, -1)

    def first_order(self, phi):
        # phi:(N,n,nb), c1:(D,n,nb) -> (N,D,n) per-feature effects
        return torch.einsum("ng b,d n b->n d g".replace(" ", ""), phi, self.c1) \
            if False else torch.einsum("nib,dib->ndi", phi, self.c1)

    def second_order(self, phi):
        if self.c2 is None:
            return None
        # for each pair (i,j): outer(phi_i, phi_j) contracted with c2[d,p]
        outs = []
        for p, (i, j) in enumerate(self.pairs):
            # (N,nb,nb) einsum with (D,nb,nb) -> (N,D)
            outer = torch.einsum("na,nb->nab", phi[:, i, :], phi[:, j, :])
            outs.append(torch.einsum("nab,dab->nd", outer, self.c2[:, p]))
        return torch.stack(outs, dim=-1)           # (N, D, P)

    def forward(self, s):
        phi = self._phi(s)
        fo = self.first_order(phi)                 # (N,D,n)
        so = self.second_order(phi)                # (N,D,P) or None
        contrib = self.beta[None, :] + fo.sum(-1)  # (N,D)
        if so is not None:
            contrib = contrib + so.sum(-1)
        return contrib, fo, so

    def group_lasso(self):
        """ell_{2,1} over each (d,pair) surface -> zeroes whole surfaces."""
        if self.c2 is None:
            return torch.zeros((), device=self.beta.device)
        # norm over (nb,nb) per (D,P), then sum -> group sparsity over surfaces
        return torch.sqrt((self.c2 ** 2).sum(dim=(-1, -2)) + 1e-12).sum()


class TGAKANModule(nn.Module):
    def __init__(self, act_dim, obs_dim, K=4, n_literals=8, oblique=True,
                 n_basis=12, max_pairs=None, lo=-3.0, hi=3.0, seed=0):
        super().__init__()
        self.D, self.n, self.K = act_dim, obs_dim, K
        all_pairs = list(itertools.combinations(range(obs_dim), 2))
        if max_pairs is not None and len(all_pairs) > max_pairs:
            all_pairs = all_pairs[:max_pairs]      # cap candidate reserve for big obs
        self.pairs = all_pairs
        self.gate = SoftDNFGate(obs_dim, K=K, n_literals=n_literals,
                                oblique=oblique, seed=seed)
        self.experts = nn.ModuleList([
            KANExpert(act_dim, obs_dim, self.pairs, n_basis=n_basis,
                      lo=lo, hi=hi, seed=seed + k)
            for k in range(K)
        ])

    def forward(self, s, hard_gate=False):
        g, aux = self.gate(s, hard=hard_gate)      # (N,K) SHARED gate
        ahat = torch.zeros(s.shape[0], self.D, device=s.device)
        fos, sos = [], []
        for k, expert in enumerate(self.experts):
            ck, fo, so = expert(s)                 # (N,D)
            ahat = ahat + g[:, k:k + 1] * ck
            fos.append(fo); sos.append(so)
        return ahat, {"g": g, "fo": fos, "so": sos, **aux}

    # -- regularizers -------------------------------------------------------
    def anova_penalty(self, s):
        """Eq. (5): purge first-order mass from bivariate surfaces.

        Penalise the (empirical) conditional means of each surface:
        E[psi_ij]=0, E[psi_ij|s_i]=0, E[psi_ij|s_j]=0. We approximate the
        marginal-integral with the batch mean and the conditionals with a
        light binning, summed over experts/outputs/pairs.
        """
        pen = torch.zeros((), device=s.device)
        for expert in self.experts:
            phi = expert._phi(s)
            so = expert.second_order(phi)          # (N,D,P) or None
            if so is None:
                continue
            pen = pen + (so.mean(0) ** 2).sum()    # E[psi]=0
            # conditional means via coarse bins on s_i and s_j
            for p, (i, j) in enumerate(expert.pairs):
                for col in (i, j):
                    bins = torch.bucketize(s[:, col].contiguous(),
                                           torch.linspace(-3, 3, 6, device=s.device))
                    for b in bins.unique():
                        m = bins == b
                        if m.sum() > 1:
                            pen = pen + (so[m, :, p].mean(0) ** 2).sum()
        return pen

    def group_lasso(self):
        return sum(e.group_lasso() for e in self.experts)

    def mdl(self):
        return self.gate.mdl_penalty()

    def boundary_loss(self, s, s_next):
        """Eq. (8): ||a_hat(s)-a_hat(s')||^2 over pairs that CROSS a regime.

        reg(.) = argmax gate. Indicator on regime change, evaluated on
        on-policy consecutive pairs (passed in from rollouts).
        """
        if s.shape[0] == 0:
            return torch.zeros((), device=next(self.parameters()).device)
        a1, aux1 = self.forward(s)
        a2, aux2 = self.forward(s_next)
        r1 = aux1["g"].argmax(-1)
        r2 = aux2["g"].argmax(-1)
        crossed = (r1 != r2).float()
        jump = ((a1 - a2) ** 2).sum(-1)
        denom = crossed.sum().clamp(min=1.0)
        return (crossed * jump).sum() / denom


class TGAKANSurrogate(BaseSurrogate):
    """sklearn-style wrapper: fit/predict/explain + the §3 self-sizing losses.

    Standardises observations internally (splines assume a stable input range).
    """

    def __init__(self, act_dim, obs_dim, *, K=4, n_literals=8, oblique=True,
                 n_basis=12, max_pairs=64, epochs=300, lr=3e-3, batch=4096,
                 lam_tr=1.0, lam_g=1e-3, lam_2=1e-3, lam_c=1e-2,
                 alpha_schedule=(1.0, 12.0), device="cuda", seed=0, verbose=True):
        self.cfg = dict(K=K, n_literals=n_literals, oblique=oblique,
                        n_basis=n_basis, max_pairs=max_pairs)
        self.act_dim, self.obs_dim = act_dim, obs_dim
        self.epochs, self.lr, self.batch = epochs, lr, batch
        self.lam_tr, self.lam_g, self.lam_2, self.lam_c = lam_tr, lam_g, lam_2, lam_c
        self.alpha_schedule = alpha_schedule
        self.device = device
        self.seed, self.verbose = seed, verbose
        self.model = None
        self.mu_ = None; self.sd_ = None

    def _standardize(self, S, fit=False):
        if fit:
            self.mu_ = S.mean(0); self.sd_ = S.std(0) + 1e-6
        return (S - self.mu_) / self.sd_

    def fit(self, S, A, *, transitions=None):
        torch.manual_seed(self.seed)
        S = np.asarray(S, np.float32); A = np.asarray(A, np.float32)
        Sn = self._standardize(S, fit=True)
        dev = torch.device(self.device)
        Xt = torch.tensor(Sn, device=dev); Yt = torch.tensor(A, device=dev)

        tr = None
        if transitions is not None and len(transitions[0]) > 0:
            s1 = (transitions[0] - self.mu_) / self.sd_
            s2 = (transitions[1] - self.mu_) / self.sd_
            tr = (torch.tensor(s1, dtype=torch.float32, device=dev),
                  torch.tensor(s2, dtype=torch.float32, device=dev))

        if self.model is None:
            self.model = TGAKANModule(self.act_dim, self.obs_dim,
                                      seed=self.seed, **self.cfg).to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        a0, a1 = self.alpha_schedule
        N = len(Xt)

        for ep in range(self.epochs):
            self.model.gate.set_alpha(a0 + (a1 - a0) * ep / max(1, self.epochs - 1))
            perm = torch.randperm(N, device=dev)
            tot = 0.0
            for b in range(0, N, self.batch):
                idx = perm[b:b + self.batch]
                s, y = Xt[idx], Yt[idx]
                ahat, _ = self.model(s)
                fid = ((ahat - y) ** 2).sum(-1).mean()
                loss = fid + self.lam_g * self.model.mdl() \
                           + self.lam_2 * self.model.group_lasso() \
                           + self.lam_c * self.model.anova_penalty(s)
                if tr is not None:
                    loss = loss + self.lam_tr * self.model.boundary_loss(*tr)
                opt.zero_grad(); loss.backward(); opt.step()
                tot += fid.detach().item() * len(idx)
            if self.verbose and (ep % 50 == 0 or ep == self.epochs - 1):
                print(f"[tga-kan] ep {ep:4d}  fid_mse={tot / N:.5f}  "
                      f"K_active={self.model.gate.active_clause_count()}")
        return self

    def predict(self, S):
        S = np.asarray(S, np.float32)
        Sn = self._standardize(S, fit=False)
        self.model.eval()
        with torch.no_grad():
            ahat, _ = self.model(torch.tensor(Sn, device=self.device), hard_gate=True)
        self.model.train()
        return ahat.cpu().numpy()

    def explain(self, **kwargs):
        return {
            "active_clauses": self.model.gate.active_clause_count(),
            "n_candidate_pairs": len(self.model.pairs),
            "surface_norms": [
                (e.c2 ** 2).sum(dim=(-1, -2)).sqrt().detach().cpu().numpy()
                if e.c2 is not None else None
                for e in self.model.experts
            ],
        }

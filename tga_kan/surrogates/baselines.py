"""Baselines: global Linear (W s + b) and locally-weighted LIME-style surrogate.

Linear  : one global least-squares map. The §1 'collapse' case.
LIME    : the field-standard XAI posture (Remark 3) — a locally-weighted linear
          model. To make it rollout-able as a global object we fit one local
          linear model per anchor (k-means centroids of d_pi*) with a distance
          kernel, and at predict time blend anchors by the same kernel. This is
          genuinely different from Linear: it recovers piecewise-linear structure
          that a single W cannot.
"""
from __future__ import annotations

import numpy as np
from .base import BaseSurrogate


class LinearSurrogate(BaseSurrogate):
    """a_hat(s) = W s + b, joint over all D outputs (one least-squares solve)."""

    def __init__(self):
        self.W = None  # (act_dim, obs_dim)
        self.b = None  # (act_dim,)

    def fit(self, S, A, *, transitions=None):
        N = S.shape[0]
        X = np.hstack([S, np.ones((N, 1))])               # (N, obs_dim+1)
        coef, *_ = np.linalg.lstsq(X, A, rcond=None)      # (obs_dim+1, act_dim)
        self.W = coef[:-1].T
        self.b = coef[-1]
        return self

    def predict(self, S):
        return S @ self.W.T + self.b

    def explain(self, **kwargs):
        return {"W": self.W, "b": self.b}


class LIMESurrogate(BaseSurrogate):
    """Locally-weighted linear surrogate (LIME-style), made global & rollout-able.

    n_anchors local linear models at k-means centroids; predictions are a
    kernel-weighted blend so the global map is continuous and can be rolled out.
    kernel: w(s, c) = exp(-||s-c||^2 / (2 sigma^2)), sigma = median pairwise / scale.
    """

    def __init__(self, n_anchors=16, kernel_scale=1.0, ridge=1e-3, seed=0):
        self.n_anchors = n_anchors
        self.kernel_scale = kernel_scale
        self.ridge = ridge
        self.seed = seed
        self.centroids = None     # (M, obs_dim)
        self.sigma = None         # scalar bandwidth
        self.local = []           # list of (W_m, b_m)

    def _kmeans(self, S):
        rng = np.random.default_rng(self.seed)
        M = min(self.n_anchors, len(S))
        idx = rng.choice(len(S), M, replace=False)
        C = S[idx].copy()
        for _ in range(25):
            d = ((S[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (N, M)
            assign = d.argmin(1)
            newC = np.array([S[assign == m].mean(0) if (assign == m).any()
                             else C[m] for m in range(M)])
            if np.allclose(newC, C):
                break
            C = newC
        return C

    def fit(self, S, A, *, transitions=None):
        S = np.asarray(S, np.float64); A = np.asarray(A, np.float64)
        self.centroids = self._kmeans(S)
        # bandwidth from median centroid spacing
        cc = ((self.centroids[:, None] - self.centroids[None]) ** 2).sum(-1)
        med = np.median(cc[cc > 0]) if (cc > 0).any() else 1.0
        self.sigma = self.kernel_scale * np.sqrt(med + 1e-12)

        self.local = []
        obs_dim = S.shape[1]
        for c in self.centroids:
            w = np.exp(-((S - c) ** 2).sum(1) / (2 * self.sigma ** 2))  # (N,)
            sw = np.sqrt(w)[:, None]
            X = np.hstack([S, np.ones((len(S), 1))]) * sw
            Y = A * sw
            G = X.T @ X + self.ridge * np.eye(obs_dim + 1)
            coef = np.linalg.solve(G, X.T @ Y)            # (obs_dim+1, act_dim)
            self.local.append((coef[:-1].T, coef[-1]))
        return self

    def predict(self, S):
        S = np.asarray(S, np.float64)
        w = np.exp(-((S[:, None, :] - self.centroids[None]) ** 2).sum(-1)
                   / (2 * self.sigma ** 2))               # (N, M)
        w = w / (w.sum(1, keepdims=True) + 1e-12)
        out = np.zeros((len(S), self.local[0][0].shape[0]))
        for m, (Wm, bm) in enumerate(self.local):
            out += w[:, m:m + 1] * (S @ Wm.T + bm)
        return out

    def explain(self, **kwargs):
        return {"centroids": self.centroids, "sigma": self.sigma,
                "local_models": self.local}

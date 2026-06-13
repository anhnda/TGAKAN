"""1-D B-spline-ish basis for KAN univariate terms.

We use a fixed grid of cubic B-spline basis functions on a normalised input
range. Coefficients over this basis are the learnable psi^k_{d,i}. Second-order
surfaces use the outer product of two 1-D bases (tensor-product spline).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _bspline_basis(x: torch.Tensor, knots: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """Cox–de Boor basis. x:(...,) knots:(G+1,) -> (..., n_basis).

    n_basis = len(knots) - degree - 1. Implemented batched over x.
    """
    x = x.unsqueeze(-1)                       # (..., 1)
    # degree 0
    k = knots
    B = ((x >= k[:-1]) & (x < k[1:])).to(x.dtype)   # (..., len(knots)-1)
    # fix right edge
    B[..., -1] = torch.where(x[..., 0] >= k[-1], torch.ones_like(B[..., -1]), B[..., -1])
    for d in range(1, degree + 1):
        n = B.shape[-1] - 1
        left_den = (k[d:d + n] - k[:n])
        right_den = (k[d + 1:d + 1 + n] - k[1:1 + n])
        left = torch.where(left_den > 0,
                           (x[..., 0:1] - k[:n]) / torch.where(left_den > 0, left_den, torch.ones_like(left_den)),
                           torch.zeros_like(left_den))
        right = torch.where(right_den > 0,
                            (k[d + 1:d + 1 + n] - x[..., 0:1]) / torch.where(right_den > 0, right_den, torch.ones_like(right_den)),
                            torch.zeros_like(right_den))
        B = left * B[..., :n] + right * B[..., 1:n + 1]
    return B                                   # (..., n_basis)


class SplineBasis(nn.Module):
    """Precomputed open-uniform knot vector over [lo, hi]; returns basis features."""

    def __init__(self, n_basis: int = 12, degree: int = 3, lo: float = -3.0, hi: float = 3.0):
        super().__init__()
        self.degree = degree
        self.n_basis = n_basis
        n_inner = n_basis - degree - 1
        if n_inner < 1:
            n_inner = 1
            self.n_basis = degree + 2
        inner = torch.linspace(lo, hi, n_inner + 2)[1:-1]
        knots = torch.cat([
            torch.full((degree + 1,), lo),
            inner,
            torch.full((degree + 1,), hi),
        ])
        self.register_buffer("knots", knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,) -> (N, n_basis)
        return _bspline_basis(x, self.knots, self.degree)

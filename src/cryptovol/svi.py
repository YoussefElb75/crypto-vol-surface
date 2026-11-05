"""Raw SVI smile per expiry, with butterfly-arbitrage diagnostics.

Raw SVI (Gatheral, 2004) for the total implied variance ``w = sigma_BS^2 T``::

    w(k) = a + b * ( rho (k - m) + sqrt((k - m)^2 + sigma^2) )

A slice has no butterfly arbitrage when Durrleman's function

    g(k) = (1 - k w'/(2w))^2 - (w'^2/4) (1/w + 1/4) + w''/2

is non-negative for all ``k`` (Gatheral & Jacquier, 2014), and Lee's
moment bound caps the wing slopes: ``b (1 + |rho|) <= 2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]

PENALTY = 1e3  # weight of the arbitrage penalty residuals during fitting
G_MARGIN = 1e-6  # the penalty pushes g above this small margin, not just above 0


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters ``(a, b, rho, m, sigma)``."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_array(self) -> FloatArray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    # --- smile and derivatives ----------------------------------------
    def total_variance(self, k: ArrayLike) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        out: FloatArray = self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma**2))
        return out

    def dw(self, k: ArrayLike) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        out: FloatArray = self.b * (self.rho + x / np.sqrt(x * x + self.sigma**2))
        return out

    def d2w(self, k: ArrayLike) -> FloatArray:
        x = np.asarray(k, dtype=np.float64) - self.m
        out: FloatArray = self.b * self.sigma**2 / (x * x + self.sigma**2) ** 1.5
        return out

    def implied_vol(self, k: ArrayLike, T: float) -> FloatArray:
        out: FloatArray = np.sqrt(np.maximum(self.total_variance(k), 0.0) / T)
        return out

    def min_total_variance(self) -> float:
        """``min_k w(k) = a + b sigma sqrt(1 - rho^2)``."""
        return float(self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho**2))

    # --- arbitrage diagnostics ----------------------------------------
    def g(self, k: ArrayLike) -> FloatArray:
        """Durrleman's butterfly function; must be ``>= 0`` everywhere."""
        return durrleman_g(
            np.asarray(k, dtype=np.float64), self.total_variance(k), self.dw(k), self.d2w(k)
        )

    def density(self, k: ArrayLike) -> FloatArray:
        """Risk-neutral density of ``x = ln(S_T / F)`` evaluated at ``x = k``."""
        return lognormal_density_from_w(
            np.asarray(k, dtype=np.float64), self.total_variance(k), self.g(k)
        )


def durrleman_g(k: FloatArray, w: FloatArray, w1: FloatArray, w2: FloatArray) -> FloatArray:
    """``g(k)`` from total variance and its first two k-derivatives."""
    w_safe = np.maximum(w, 1e-300)
    out: FloatArray = (
        (1.0 - k * w1 / (2.0 * w_safe)) ** 2 - 0.25 * w1**2 * (1.0 / w_safe + 0.25) + 0.5 * w2
    )
    return out


def lognormal_density_from_w(k: FloatArray, w: FloatArray, g: FloatArray) -> FloatArray:
    """Density of log-moneyness implied by a smile: ``g/sqrt(2 pi w) exp(-d2^2/2)``."""
    w_safe = np.maximum(w, 1e-300)
    d2 = -k / np.sqrt(w_safe) - 0.5 * np.sqrt(w_safe)
    out: FloatArray = g / np.sqrt(2.0 * np.pi * w_safe) * np.exp(-0.5 * d2 * d2)
    return out


@dataclass(frozen=True)
class ButterflyCheck:
    min_g: float
    argmin_k: float
    min_w: float
    wing_slope: float  # b (1 + |rho|), must be <= 2

    @property
    def ok(self) -> bool:
        return self.min_g >= -1e-9 and self.min_w >= 0.0 and self.wing_slope <= 2.0 + 1e-12


def arbitrage_grid(k_min: float = -1.5, k_max: float = 1.5, n: int = 601) -> FloatArray:
    return np.linspace(k_min, k_max, n)


def check_butterfly(p: SVIParams, k_grid: FloatArray | None = None) -> ButterflyCheck:
    """Check ``g >= 0`` on a grid, ``w >= 0`` and Lee's wing bound."""
    kg = arbitrage_grid() if k_grid is None else k_grid
    gv = p.g(kg)
    i = int(np.argmin(gv))
    return ButterflyCheck(
        min_g=float(gv[i]),
        argmin_k=float(kg[i]),
        min_w=p.min_total_variance(),
        wing_slope=p.b * (1.0 + abs(p.rho)),
    )


@dataclass(frozen=True)
class SVIFit:
    params: SVIParams
    T: float
    rmse_volpts: float  # unweighted RMSE of IV, vol points
    n: int
    butterfly: ButterflyCheck
    success: bool


def _residuals(
    x: FloatArray, k: FloatArray, iv: FloatArray, sw: FloatArray, T: float, kg: FloatArray
) -> FloatArray:
    p = SVIParams(*x)
    w = p.total_variance(k)
    model_iv = np.sqrt(np.maximum(w, 1e-12) / T)
    pen_g = PENALTY * np.maximum(G_MARGIN - p.g(kg), 0.0)
    pen_w = PENALTY * max(-p.min_total_variance(), 0.0)
    pen_wing = PENALTY * max(p.b * (1.0 + abs(p.rho)) - 2.0, 0.0)
    return np.concatenate([sw * (model_iv - iv), pen_g, [pen_w, pen_wing]])


def fit_svi(
    k: ArrayLike,
    iv: ArrayLike,
    T: float,
    weights: ArrayLike | None = None,
    k_grid: FloatArray | None = None,
) -> SVIFit:
    """Fit raw SVI to one smile by weighted least squares in IV space.

    Butterfly arbitrage (``g < 0`` on ``k_grid``), negative variance and
    Lee's wing bound are added as penalty residuals, so the optimum is
    arbitrage-free in practice. :func:`check_butterfly` verifies the result.
    Several starting points are tried and the best fit is kept.
    """
    order = np.argsort(np.asarray(k, dtype=np.float64))
    k_ = np.asarray(k, dtype=np.float64)[order]
    iv_ = np.asarray(iv, dtype=np.float64)[order]
    w_ = np.ones_like(k_) if weights is None else np.asarray(weights, dtype=np.float64)[order]
    sw = np.sqrt(w_ / np.mean(w_))
    k_lo, k_hi = float(np.min(k_)), float(np.max(k_))
    span = max(k_hi - k_lo, 0.05)
    kg = (
        arbitrage_grid(min(-1.5, k_lo - span), max(1.5, k_hi + span), 401)
        if k_grid is None
        else k_grid
    )
    w_mkt = iv_**2 * T
    w_max = float(np.max(w_mkt))
    w_atm = float(np.interp(0.0, k_, w_mkt)) if k_lo < 0 < k_hi else float(np.min(w_mkt))

    lb = np.array([-2.0 * w_max, 1e-6, -0.999, k_lo - span, 1e-4])
    ub = np.array([2.0 * w_max, 2.0, 0.999, k_hi + span, 2.0 * span + 1.0])
    best = None
    for rho0 in (-0.4, 0.0, 0.3):
        for s0 in (0.1 * span, 0.3 * span):
            b0 = max(0.5 * (w_max - w_atm) / span, 1e-3)
            a0 = max(w_atm - b0 * s0, 1e-6)
            x0 = np.clip([a0, b0, rho0, 0.0, s0], lb + 1e-9, ub - 1e-9)
            res = least_squares(
                _residuals,
                x0,
                bounds=(lb, ub),
                args=(k_, iv_, sw, T, kg),
                method="trf",
                x_scale="jac",
                max_nfev=4000,
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    p = SVIParams(*best.x)
    err = p.implied_vol(k_, T) - iv_
    return SVIFit(
        params=p,
        T=T,
        rmse_volpts=float(100 * np.sqrt(np.mean(err**2))),
        n=int(k_.size),
        butterfly=check_butterfly(p, kg),
        success=bool(best.success),
    )

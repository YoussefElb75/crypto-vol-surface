"""Fitted volatility surface: total variance, IV, Black-76 prices and Greeks.

Two surfaces are built from the same quotes:

* ``"svi"``: one raw-SVI slice per expiry. Between expiries, total variance
  is interpolated linearly in ``T`` at fixed ``k``. Each slice is
  butterfly-free, but calendar arbitrage between slices is only *checked*,
  not ruled out.
* ``"ssvi"``: a single SSVI surface, free of static arbitrage by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from cryptovol import black76
from cryptovol.ssvi import SSVIFit, fit_ssvi
from cryptovol.svi import SVIFit, fit_svi

FloatArray = NDArray[np.float64]
Model = Literal["svi", "ssvi"]

MIN_IV_SPREAD = 0.005  # floor for the bid-ask IV spread used in weights (0.5 vol pt)


def quote_weights(iv_spread: ArrayLike) -> FloatArray:
    """Fit weights: inverse bid-ask IV spread, floored at 0.5 vol points."""
    out: FloatArray = 1.0 / np.maximum(
        np.nan_to_num(np.asarray(iv_spread, dtype=np.float64), nan=0.05), MIN_IV_SPREAD
    )
    return out


@dataclass
class VolSurface:
    """Implied volatility surface plus forward curve."""

    model: Model
    T_nodes: FloatArray  # fitted expiries (years), increasing
    F_nodes: FloatArray  # forwards at those expiries
    svi: list[SVIFit] = field(default_factory=list)
    ssvi: SSVIFit | None = None

    # --- forward curve --------------------------------------------------
    def forward(self, T: ArrayLike) -> FloatArray:
        """Forward at any T: log-linear in T between nodes, flat carry outside."""
        T_ = np.asarray(T, dtype=np.float64)
        y = np.log(self.F_nodes)
        Tn = self.T_nodes
        slope_lo = (y[1] - y[0]) / (Tn[1] - Tn[0]) if Tn.size > 1 else 0.0
        slope_hi = (y[-1] - y[-2]) / (Tn[-1] - Tn[-2]) if Tn.size > 1 else 0.0
        inside = np.interp(T_, Tn, y)
        lo = y[0] + slope_lo * (T_ - Tn[0])
        hi = y[-1] + slope_hi * (T_ - Tn[-1])
        out: FloatArray = np.exp(np.where(Tn[0] > T_, lo, np.where(Tn[-1] < T_, hi, inside)))
        return out

    # --- volatility -----------------------------------------------------
    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        k_, T_ = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(T, dtype=np.float64)
        )
        if self.model == "ssvi":
            assert self.ssvi is not None
            return self.ssvi.params.total_variance(k_, T_)
        # SVI slices: linear in T at fixed k; scale with T outside the node range.
        W = np.array([f.params.total_variance(k_) for f in self.svi])  # (n_exp, *shape)
        Tn = self.T_nodes
        out = np.empty(k_.shape, dtype=np.float64)
        flat_T = T_.reshape(-1)
        flat_out = out.reshape(-1)
        Wf = W.reshape(W.shape[0], -1)
        for i, t in enumerate(flat_T):
            if t <= Tn[0]:
                flat_out[i] = Wf[0, i] * t / Tn[0]
            elif t >= Tn[-1]:
                flat_out[i] = Wf[-1, i] * t / Tn[-1]
            else:
                j = int(np.searchsorted(Tn, t)) - 1
                a = (t - Tn[j]) / (Tn[j + 1] - Tn[j])
                flat_out[i] = (1 - a) * Wf[j, i] + a * Wf[j + 1, i]
        return out

    def iv(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        """Implied vol at log-moneyness ``k`` and maturity ``T``."""
        out: FloatArray = np.sqrt(np.maximum(self.total_variance(k, T), 0.0) / np.asarray(T))
        return out

    def iv_strike(self, K: ArrayLike, T: ArrayLike) -> FloatArray:
        k = np.log(np.asarray(K, dtype=np.float64) / self.forward(T))
        return self.iv(k, T)

    # --- pricing ----------------------------------------------------------
    def price(
        self,
        K: ArrayLike,
        T: ArrayLike,
        is_call: ArrayLike = True,
        unit: Literal["usd", "coin"] = "usd",
    ) -> FloatArray:
        """Black-76 price from the surface (``df = 1``, Deribit's convention)."""
        F = self.forward(T)
        v = black76.price(F, K, T, self.iv_strike(K, T), is_call)
        return black76.usd_to_coin(v, F) if unit == "coin" else v

    def greeks(self, K: ArrayLike, T: ArrayLike, is_call: ArrayLike = True) -> pd.DataFrame:
        """Sticky-strike Black-76 Greeks at the surface vol (USD, per unit of underlying)."""
        K_, T_, c_ = np.broadcast_arrays(
            np.asarray(K, dtype=np.float64),
            np.asarray(T, dtype=np.float64),
            np.asarray(is_call, dtype=bool),
        )
        K_, T_, c_ = K_.ravel(), T_.ravel(), c_.ravel()
        F = self.forward(T_)
        s = self.iv_strike(K_, T_)
        return pd.DataFrame(
            {
                "K": K_,
                "T": T_,
                "is_call": c_,
                "F": F,
                "iv": s,
                "price_usd": black76.price(F, K_, T_, s, c_),
                "delta": black76.delta(F, K_, T_, s, c_),
                "delta_pa": black76.premium_adjusted_delta(F, K_, T_, s, c_),
                "gamma": black76.gamma(F, K_, T_, s),
                "vega_1pct": 0.01 * black76.vega(F, K_, T_, s),
                "theta_1d": black76.theta(F, K_, T_, s, c_) / 365.0,
            }
        )


@dataclass
class SurfaceFits:
    """SVI slices, the SSVI fit and both surfaces for one currency."""

    svi: VolSurface
    ssvi: VolSurface
    expiries: pd.DataFrame  # one row per fitted expiry: T, F, n, rmse, checks


def fit_surfaces(quotes: pd.DataFrame) -> SurfaceFits:
    """Fit SVI per expiry and SSVI jointly on the ``keep`` rows of cleaned quotes."""
    q = quotes[quotes["keep"]].sort_values(["T", "k"])
    if q.empty:
        raise ValueError("no clean quotes to fit")
    w = quote_weights(q["iv_spread"].to_numpy())
    q = q.assign(weight=w)
    svi_fits: list[SVIFit] = []
    T_nodes, F_nodes, rows = [], [], []
    for T, g in q.groupby("T", sort=True):
        T_f = float(T)  # type: ignore[arg-type]
        fit = fit_svi(g["k"].to_numpy(), g["mid_iv"].to_numpy(), T_f, g["weight"].to_numpy())
        svi_fits.append(fit)
        T_nodes.append(T_f)
        F_nodes.append(float(g["F"].iloc[0]))
        rows.append(
            {
                "expiry": g["expiry"].iloc[0],
                "T": T_f,
                "F": F_nodes[-1],
                "n": fit.n,
                "svi_rmse_volpts": fit.rmse_volpts,
                "svi_butterfly_ok": fit.butterfly.ok,
                "svi_min_g": fit.butterfly.min_g,
            }
        )
    ssvi_fit = fit_ssvi(q["T"].to_numpy(), q["k"].to_numpy(), q["mid_iv"].to_numpy(), w)
    Tn, Fn = np.asarray(T_nodes), np.asarray(F_nodes)
    exp = pd.DataFrame(rows)
    exp["ssvi_rmse_volpts"] = [ssvi_fit.rmse_by_expiry[t] for t in T_nodes]
    exp["ssvi_theta"] = ssvi_fit.params.theta
    svi_surface = VolSurface("svi", Tn, Fn, svi=svi_fits)
    ssvi_surface = VolSurface("ssvi", Tn, Fn, ssvi=ssvi_fit)
    return SurfaceFits(svi=svi_surface, ssvi=ssvi_surface, expiries=exp)


def svi_calendar_gaps(surface: VolSurface, k_grid: FloatArray | None = None) -> FloatArray:
    """``min_k [w_{i+1}(k) - w_i(k)]`` for consecutive SVI slices (negative = arbitrage)."""
    kg = np.linspace(-1.5, 1.5, 601) if k_grid is None else k_grid
    W = np.array([f.params.total_variance(kg) for f in surface.svi])
    out: FloatArray = np.diff(W, axis=0).min(axis=1)
    return out

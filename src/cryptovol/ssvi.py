"""SSVI surface across expiries (Gatheral & Jacquier, 2014).

    w(k, theta) = theta/2 * ( 1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2) ),
    phi(theta)  = eta / ( theta^gamma (1 + theta)^(1 - gamma) )

``theta_t`` is the ATM total variance. Constraints used here:

* **Calendar (GJ Theorem 4.1):** ``theta_t`` non-decreasing in ``t``, enforced
  by fitting non-negative increments, together with
  ``0 <= d(theta phi)/d theta <= phi (1 + sqrt(1 - rho^2)) / rho^2``. For this
  ``phi`` the derivative is ``phi (1 - gamma) / (1 + theta)``, which lies in
  ``[0, phi)``. The right-hand bound is at least ``phi`` for any ``|rho| <= 1``,
  so both inequalities always hold.
* **Butterfly:** ``eta (1 + |rho|) <= 2`` with ``0 < gamma <= 1/2``.
  Then for every theta, ``theta phi (1+|rho|) < 4`` and
  ``theta phi^2 (1+|rho|) <= 4`` (GJ Theorem 4.2). Enforced by
  reparametrising ``eta = u * 2 / (1 + |rho|)`` with ``u`` in ``[0, 1]``.

Both conditions are also verified numerically by :func:`check_ssvi`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from cryptovol.svi import durrleman_g, lognormal_density_from_w

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SSVIParams:
    rho: float
    eta: float
    gamma: float
    T: FloatArray  # fitted expiries (years), increasing
    theta: FloatArray  # ATM total variance at each expiry, non-decreasing

    def phi(self, theta: ArrayLike) -> FloatArray:
        th = np.maximum(np.asarray(theta, dtype=np.float64), 1e-12)
        out: FloatArray = self.eta / (th**self.gamma * (1.0 + th) ** (1.0 - self.gamma))
        return out

    def theta_at(self, T: ArrayLike) -> FloatArray:
        """ATM total variance at any maturity.

        Linear in ``T`` between nodes, through ``(0, 0)`` before the first
        node, and at constant ATM vol after the last. Monotone, so the
        interpolated surface stays free of calendar arbitrage.
        """
        T_ = np.asarray(T, dtype=np.float64)
        nodes_T = np.concatenate([[0.0], self.T])
        nodes_th = np.concatenate([[0.0], self.theta])
        inside = np.interp(T_, nodes_T, nodes_th)
        out: FloatArray = np.where(self.T[-1] < T_, self.theta[-1] * T_ / self.T[-1], inside)
        return out

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        th = self.theta_at(T)
        return ssvi_w(np.asarray(k, dtype=np.float64), th, self.rho, self.phi(th))

    def implied_vol(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        out: FloatArray = np.sqrt(np.maximum(self.total_variance(k, T), 0.0) / np.asarray(T))
        return out

    def derivatives(self, k: ArrayLike, T: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``w, dw/dk, d2w/dk2`` of the slice at maturity ``T``."""
        k_ = np.asarray(k, dtype=np.float64)
        th = float(self.theta_at(T))
        ph = float(self.phi(th))
        r = self.rho
        z = ph * k_ + r
        root = np.sqrt(z * z + 1.0 - r * r)
        w = 0.5 * th * (1.0 + r * ph * k_ + root)
        w1 = 0.5 * th * ph * (r + z / root)
        w2 = 0.5 * th * ph * ph * (1.0 - r * r) / root**3
        return w, w1, w2

    def g(self, k: ArrayLike, T: float) -> FloatArray:
        w, w1, w2 = self.derivatives(k, T)
        return durrleman_g(np.asarray(k, dtype=np.float64), w, w1, w2)

    def density(self, k: ArrayLike, T: float) -> FloatArray:
        w, _, _ = self.derivatives(k, T)
        return lognormal_density_from_w(np.asarray(k, dtype=np.float64), w, self.g(k, T))


def ssvi_w(k: FloatArray, theta: ArrayLike, rho: float, phi: ArrayLike) -> FloatArray:
    th, ph = np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64)
    z = ph * k + rho
    out: FloatArray = 0.5 * th * (1.0 + rho * ph * k + np.sqrt(z * z + 1.0 - rho * rho))
    return out


@dataclass(frozen=True)
class SSVICheck:
    theta_monotone: bool
    butterfly_bound_1: float  # max theta phi (1 + |rho|), must be < 4
    butterfly_bound_2: float  # max theta phi^2 (1 + |rho|), must be <= 4
    min_g: float  # min of g over the k grid and all fitted expiries
    min_calendar_gap: float  # min over k of w(k, T_{i+1}) - w(k, T_i)

    @property
    def ok(self) -> bool:
        return (
            self.theta_monotone
            and self.butterfly_bound_1 < 4.0
            and self.butterfly_bound_2 <= 4.0 + 1e-12
            and self.min_g >= -1e-9
            and self.min_calendar_gap >= -1e-12
        )


def check_ssvi(p: SSVIParams, k_grid: FloatArray | None = None) -> SSVICheck:
    kg = np.linspace(-1.5, 1.5, 601) if k_grid is None else k_grid
    th = p.theta
    ph = p.phi(th)
    gs = [float(np.min(p.g(kg, float(t)))) for t in p.T]
    ws = np.array([p.total_variance(kg, float(t)) for t in p.T])
    gaps = np.diff(ws, axis=0)
    return SSVICheck(
        theta_monotone=bool(np.all(np.diff(th) >= -1e-15)),
        butterfly_bound_1=float(np.max(th * ph * (1 + abs(p.rho)))),
        butterfly_bound_2=float(np.max(th * ph**2 * (1 + abs(p.rho)))),
        min_g=min(gs),
        min_calendar_gap=float(gaps.min()) if gaps.size else 0.0,
    )


@dataclass(frozen=True)
class SSVIFit:
    params: SSVIParams
    rmse_volpts: float
    rmse_by_expiry: dict[float, float]
    check: SSVICheck
    success: bool


def _unpack(x: FloatArray, T: FloatArray) -> SSVIParams:
    rho, u, gamma = x[0], x[1], x[2]
    theta = np.cumsum(x[3:])
    return SSVIParams(
        rho=float(rho), eta=float(u * 2.0 / (1.0 + abs(rho))), gamma=float(gamma), T=T, theta=theta
    )


def fit_ssvi(
    T: ArrayLike,
    k: ArrayLike,
    iv: ArrayLike,
    weights: ArrayLike | None = None,
    theta0: ArrayLike | None = None,
) -> SSVIFit:
    """Fit SSVI jointly to all quotes; ``T``, ``k``, ``iv`` are per quote.

    ``theta0`` (one value per unique expiry, increasing ``T``) is the starting
    ATM total variance; if omitted it comes from interpolating each smile
    at ``k = 0``. The starting curve is made monotone with a running max.
    """
    T_, k_, iv_ = (np.asarray(a, dtype=np.float64) for a in (T, k, iv))
    w_ = np.ones_like(k_) if weights is None else np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(w_ / w_.mean())
    Tu, idx = np.unique(T_, return_inverse=True)
    if theta0 is None:
        th0 = []
        for j, t in enumerate(Tu):
            m = idx == j
            order = np.argsort(k_[m])
            th0.append(float(np.interp(0.0, k_[m][order], iv_[m][order])) ** 2 * t)
        th_init = np.maximum.accumulate(np.asarray(th0))
    else:
        th_init = np.maximum.accumulate(np.asarray(theta0, dtype=np.float64))
    inc0 = np.diff(np.concatenate([[0.0], th_init]))
    inc0 = np.maximum(inc0, 1e-8)

    def resid(x: FloatArray) -> FloatArray:
        p = _unpack(x, Tu)
        th = p.theta[idx]
        w = ssvi_w(k_, th, p.rho, p.phi(th))
        return sw * (np.sqrt(np.maximum(w, 1e-12) / T_) - iv_)

    lb = np.concatenate([[-0.999, 0.0, 0.01], np.zeros(Tu.size)])
    ub = np.concatenate([[0.999, 1.0, 0.5], np.full(Tu.size, np.inf)])
    best = None
    for rho0 in (-0.3, 0.0, 0.2):
        for u0 in (0.3, 0.8):
            x0 = np.concatenate([[rho0, u0, 0.4], inc0])
            res = least_squares(
                resid, x0, bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=5000
            )
            if best is None or res.cost < best.cost:
                best = res
    assert best is not None
    p = _unpack(best.x, Tu)
    err = p.implied_vol(k_, T_) - iv_
    by_exp = {float(t): float(100 * np.sqrt(np.mean(err[idx == j] ** 2))) for j, t in enumerate(Tu)}
    return SSVIFit(
        params=p,
        rmse_volpts=float(100 * np.sqrt(np.mean(err**2))),
        rmse_by_expiry=by_exp,
        check=check_ssvi(p),
        success=bool(best.success),
    )

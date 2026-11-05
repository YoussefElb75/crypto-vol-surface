"""Black-76 prices, Greeks and implied volatility (vectorised).

Conventions
-----------
* ``F`` forward, ``K`` strike, ``T`` year fraction, ``sigma`` annualised vol,
  ``df`` discount factor (defaults to 1, matching Deribit's zero-rate mark model).
* ``is_call`` is a boolean (array): ``True`` for calls, ``False`` for puts.
* Prices are in quote currency (USD). Deribit quotes options in the
  underlying coin; :func:`usd_to_coin` and :func:`coin_to_usd` convert with
  the forward, which is how Deribit's mark prices are built.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x: FloatArray) -> FloatArray:
    out: FloatArray = np.exp(-0.5 * x * x) / _SQRT_2PI
    return out


def _cdf(x: FloatArray) -> FloatArray:
    out: FloatArray = ndtr(x)
    return out


def _arr(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def d1_d2(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Return the Black-76 ``d1`` and ``d2`` terms."""
    F_, K_, T_, s_ = _arr(F), _arr(K), _arr(T), _arr(sigma)
    sd = s_ * np.sqrt(T_)
    d1 = (np.log(F_ / K_) + 0.5 * sd * sd) / sd
    return d1, d1 - sd


def price(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike = True,
    df: ArrayLike = 1.0,
) -> FloatArray:
    """Black-76 option price ``df * (F N(d1) - K N(d2))`` (call) or its put analogue."""
    F_, K_, df_ = _arr(F), _arr(K), _arr(df)
    cp = np.where(np.asarray(is_call, dtype=bool), 1.0, -1.0)
    d1, d2 = d1_d2(F_, K_, T, sigma)
    out: FloatArray = df_ * cp * (F_ * _cdf(cp * d1) - K_ * _cdf(cp * d2))
    return out


def delta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike = True,
    df: ArrayLike = 1.0,
) -> FloatArray:
    """Forward delta ``dV/dF``: ``df*N(d1)`` for calls, ``df*(N(d1)-1)`` for puts."""
    d1, _ = d1_d2(F, K, T, sigma)
    call = np.asarray(is_call, dtype=bool)
    out: FloatArray = _arr(df) * np.where(call, _cdf(d1), _cdf(d1) - 1.0)
    return out


def premium_adjusted_delta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike = True,
) -> FloatArray:
    """Premium-adjusted delta for an inverse (coin-settled) option, with ``df = 1``.

    The premium ``V/F`` is paid in coin, so the coin hedge is
    ``Delta - V/F``. That equals ``(K/F) N(d2)`` for calls and
    ``-(K/F) N(-d2)`` for puts.
    """
    K_, F_ = _arr(K), _arr(F)
    _, d2 = d1_d2(F_, K_, T, sigma)
    call = np.asarray(is_call, dtype=bool)
    out: FloatArray = np.where(call, K_ / F_ * _cdf(d2), -K_ / F_ * _cdf(-d2))
    return out


def gamma(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> FloatArray:
    """``d2V/dF2`` (same for calls and puts)."""
    d1, _ = d1_d2(F, K, T, sigma)
    out: FloatArray = _arr(df) * _pdf(d1) / (_arr(F) * _arr(sigma) * np.sqrt(_arr(T)))
    return out


def vega(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0
) -> FloatArray:
    """``dV/dsigma`` per unit of vol (multiply by 0.01 for one vol point)."""
    d1, _ = d1_d2(F, K, T, sigma)
    out: FloatArray = _arr(df) * _arr(F) * _pdf(d1) * np.sqrt(_arr(T))
    return out


def theta(
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    sigma: ArrayLike,
    is_call: ArrayLike = True,
    df: ArrayLike = 1.0,
) -> FloatArray:
    """Calendar-time decay ``dV/dt = -dV/dT`` per year, with ``F`` held fixed.

    Uses the flat-rate convention ``df = exp(-r T)`` so ``r = -ln(df)/T``.
    """
    T_, df_ = _arr(T), _arr(df)
    d1, _ = d1_d2(F, K, T_, sigma)
    r = -np.log(df_) / T_
    v = price(F, K, T_, sigma, is_call, df_)
    out: FloatArray = -df_ * _arr(F) * _pdf(d1) * _arr(sigma) / (2.0 * np.sqrt(T_)) + r * v
    return out


def usd_to_coin(v_usd: ArrayLike, F: ArrayLike) -> FloatArray:
    """Convert a USD premium to Deribit's coin quote (premium / forward)."""
    out: FloatArray = _arr(v_usd) / _arr(F)
    return out


def coin_to_usd(v_coin: ArrayLike, F: ArrayLike) -> FloatArray:
    """Convert a coin premium to USD (premium * forward)."""
    out: FloatArray = _arr(v_coin) * _arr(F)
    return out


def implied_vol(
    value: ArrayLike,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    is_call: ArrayLike = True,
    df: ArrayLike = 1.0,
    lo: float = 1e-4,
    hi: float = 10.0,
    tol: float = 1e-10,
) -> FloatArray:
    """Invert Black-76 by vectorised bisection on total std-dev, then Newton polish.

    Returns ``nan`` wherever the price breaks the no-arbitrage bounds
    ``intrinsic <= V <= df*F`` (call) or ``df*K`` (put).
    """
    V, F_, K_, T_, df_ = np.broadcast_arrays(_arr(value), _arr(F), _arr(K), _arr(T), _arr(df))
    call = np.broadcast_to(np.asarray(is_call, dtype=bool), V.shape)
    # Normalised call price c = C / (df F), using put-call parity for puts.
    c = V / (df_ * F_) + np.where(call, 0.0, 1.0 - K_ / F_)
    intrinsic = np.maximum(1.0 - K_ / F_, 0.0)
    valid = np.isfinite(c) & (c > intrinsic) & (c < 1.0) & (T_ > 0)
    k = np.log(K_ / F_)

    def norm_call(s: FloatArray) -> FloatArray:
        d1 = -k / s + 0.5 * s
        out: FloatArray = _cdf(d1) - np.exp(k) * _cdf(d1 - s)
        return out

    sqrtT = np.sqrt(np.where(T_ > 0, T_, 1.0))
    a = np.full(V.shape, lo) * sqrtT
    b = np.full(V.shape, hi) * sqrtT
    for _ in range(80):
        m = 0.5 * (a + b)
        above = norm_call(m) > c
        b = np.where(above, m, b)
        a = np.where(above, a, m)
        if np.nanmax(np.where(valid, b - a, 0.0), initial=0.0) < tol:
            break
    # Prices that need a vol above ``hi`` (or below ``lo``) never get bracketed: return nan.
    valid &= (norm_call(np.full(V.shape, hi) * sqrtT) >= c) & (
        norm_call(np.full(V.shape, lo) * sqrtT) <= c
    )
    s = 0.5 * (a + b)
    for _ in range(3):  # Newton polish on s (vega in normalised units is pdf(d1))
        d1 = -k / s + 0.5 * s
        step = (norm_call(s) - c) / np.maximum(_pdf(d1), 1e-300)
        s = np.where(np.abs(step) < 0.1 * s, s - step, s)
    out: FloatArray = np.where(valid, s / sqrtT, np.nan)
    return out

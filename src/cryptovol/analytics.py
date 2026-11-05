"""Smile analytics: ATM term structure, 25-delta RR/BF, errors vs mark, repricing.

25-delta convention
-------------------
Deribit options are inverse: quoted, margined and settled in the coin. Two
deltas are therefore natural:

* ``"forward"``: Black-76 forward delta ``N(d1)`` (calls) or ``N(d1) - 1``
  (puts). This is the **default** for 25-delta quotes. It is the convention
  of Deribit's own Greeks and of most crypto vol-surface reports. At fixed
  vol it is monotone in strike. With a smile it need not be, so the solver
  checks for a single crossing. It matches the standard FX "forward,
  non-premium-adjusted" convention, so results can be compared with the
  literature.
* ``"premium_adjusted"``: the coin hedge ratio of an inverse option,
  ``Delta - V/F``, which equals ``(K/F) N(d2)`` for calls and ``-(K/F) N(-d2)``
  for puts. It is the economically right hedge when the premium is paid in
  coin (as for FX pairs with the premium in foreign currency). The call
  version is *not* monotone in strike: it peaks and then falls, so we take
  the out-of-the-money root, above the peak.

Both are reported. Premium adjustment moves the 25-delta call strike closer
to the money and the 25-delta put strike further out, so RR and BF shift.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr

from cryptovol import black76
from cryptovol.surface import VolSurface

DeltaConvention = Literal["forward", "premium_adjusted"]
K_LO, K_HI = -4.0, 4.0  # log-moneyness search range for delta strikes


def _delta_of_k(
    surface: VolSurface, T: float, k: np.ndarray, call: bool, convention: DeltaConvention
) -> np.ndarray:
    """Delta under ``convention`` at log-moneyness ``k`` (vectorised over ``k``)."""
    w = np.maximum(surface.total_variance(k, T), 1e-16)
    sw = np.sqrt(w)
    d1 = -k / sw + 0.5 * sw
    d2 = d1 - sw
    if convention == "forward":
        out = ndtr(d1) if call else ndtr(d1) - 1.0
    else:
        out = np.exp(k) * ndtr(d2) if call else -np.exp(k) * ndtr(-d2)
    return np.asarray(out, dtype=np.float64)


def delta_strike(
    surface: VolSurface, T: float, delta: float, call: bool, convention: DeltaConvention = "forward"
) -> float:
    """Log-moneyness ``k`` at which the option has the given delta.

    ``delta`` is signed: e.g. ``0.25`` for a call and ``-0.25`` for a put.
    Crossings are located on a grid over ``[K_LO, K_HI]`` and refined with Brent.

    * Premium-adjusted call: the delta rises and then falls, so the target is
      usually crossed twice. The out-of-the-money root, at the largest ``k``,
      is returned.
    * Every other case should have exactly one crossing. If there are several
      (a pathological smile), the result is ambiguous and ``nan`` is returned.

    Returns ``nan`` if the target is never reached.
    """
    grid = np.linspace(K_LO, K_HI, 1601)
    f_grid = _delta_of_k(surface, T, grid, call, convention) - delta
    idx = np.flatnonzero(np.sign(f_grid[:-1]) * np.sign(f_grid[1:]) <= 0)
    if idx.size == 0:
        return float("nan")
    if call and convention == "premium_adjusted":
        i = int(idx[-1])
    elif idx.size == 1 or np.all(np.diff(idx) == 1):  # one crossing (possibly on a node)
        i = int(idx[0])
    else:
        return float("nan")

    def f(k: float) -> float:
        return float(_delta_of_k(surface, T, np.array([k]), call, convention)[0] - delta)

    lo, hi = float(grid[i]), float(grid[i + 1])
    if f(lo) == 0.0:
        return lo
    return float(brentq(f, lo, hi, xtol=1e-12))


def atm_term_structure(surface: VolSurface, T: np.ndarray | None = None) -> pd.DataFrame:
    """ATM-forward (``k = 0``) implied vol by maturity."""
    T_ = surface.T_nodes if T is None else np.asarray(T, dtype=np.float64)
    return pd.DataFrame({"T": T_, "days": T_ * 365.0, "atm_iv": surface.iv(0.0, T_)})


def rr_bf(
    surface: VolSurface, convention: DeltaConvention = "forward", delta: float = 0.25
) -> pd.DataFrame:
    """25-delta risk reversal and butterfly per fitted expiry.

    ``RR = sigma(25d call) - sigma(25d put)``,
    ``BF = (sigma(25d call) + sigma(25d put))/2 - sigma(ATM)``, with ATM taken
    as the ATM forward (``k = 0``).
    """
    rows = []
    for T in surface.T_nodes:
        t = float(T)
        kc = delta_strike(surface, t, delta, True, convention)
        kp = delta_strike(surface, t, -delta, False, convention)
        ivc, ivp, iva = (
            float(surface.iv(x, t)) if np.isfinite(x) else np.nan for x in (kc, kp, 0.0)
        )
        rows.append(
            {
                "T": t,
                "days": t * 365.0,
                "convention": convention,
                "k_call": kc,
                "k_put": kp,
                "iv_call": ivc,
                "iv_put": ivp,
                "iv_atm": iva,
                "rr": ivc - ivp,
                "bf": 0.5 * (ivc + ivp) - iva,
            }
        )
    return pd.DataFrame(rows)


def mark_errors(quotes: pd.DataFrame, surface: VolSurface) -> pd.DataFrame:
    """Fitted IV vs Deribit mark IV and vs mid IV, in vol points, per clean quote."""
    q = quotes[quotes["keep"]].copy()
    q["fit_iv"] = surface.iv(q["k"].to_numpy(), q["T"].to_numpy())
    q["err_vs_mark_volpts"] = 100.0 * (q["fit_iv"] - q["mark_iv"])
    q["err_vs_mid_volpts"] = 100.0 * (q["fit_iv"] - q["mid_iv"])
    q["mid_minus_mark_volpts"] = 100.0 * (q["mid_iv"] - q["mark_iv"])
    return q


def reprice(quotes: pd.DataFrame, surface: VolSurface) -> pd.DataFrame:
    """Reprice each clean quote off the surface; flag whether it lands inside bid-ask.

    Prices are in coin, like Deribit's quotes. The cleaned forward per expiry
    is used, so repricing tests the smile fit, not the forward interpolation.
    """
    q = quotes[quotes["keep"]].copy()
    F, K, T = q["F"].to_numpy(), q["K"].to_numpy(), q["T"].to_numpy()
    s = surface.iv(q["k"].to_numpy(), T)
    q["model_coin"] = black76.usd_to_coin(black76.price(F, K, T, s, q["is_call"].to_numpy()), F)
    q["inside_spread"] = (q["model_coin"] >= q["bid"]) & (q["model_coin"] <= q["ask"])
    half = 0.5 * (q["ask"] - q["bid"])
    q["err_half_spreads"] = (q["model_coin"] - q["mid"]) / half
    return q


def reprice_summary(repriced: pd.DataFrame) -> pd.DataFrame:
    """Share of quotes repriced inside the bid-ask, by expiry."""
    return (
        repriced.groupby("expiry")
        .agg(
            n=("K", "size"),
            inside_share=("inside_spread", "mean"),
            median_abs_err_half_spreads=("err_half_spreads", lambda x: float(np.median(np.abs(x)))),
        )
        .reset_index()
    )


def fit_comparison(
    svi_rp: pd.DataFrame, ssvi_rp: pd.DataFrame, expiries: pd.DataFrame
) -> pd.DataFrame:
    """Per-expiry SVI vs SSVI: RMSE (vol pts) and share repriced inside bid-ask."""
    a = svi_rp.groupby("expiry")["inside_spread"].mean().rename("svi_inside_share")
    b = ssvi_rp.groupby("expiry")["inside_spread"].mean().rename("ssvi_inside_share")
    out = expiries.set_index("expiry").join(a).join(b).reset_index()
    out["days"] = out["T"] * 365.0
    cols = [
        "expiry",
        "days",
        "F",
        "n",
        "svi_rmse_volpts",
        "ssvi_rmse_volpts",
        "svi_inside_share",
        "ssvi_inside_share",
        "svi_butterfly_ok",
    ]
    return out[cols]

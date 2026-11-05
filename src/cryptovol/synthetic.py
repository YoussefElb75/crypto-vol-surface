"""Synthetic snapshot generator with a known SSVI ground truth.

Deribit's terms do not allow raw market data to be redistributed, so the
committed test fixture is synthetic. It uses the same schema as a real
snapshot (see :mod:`cryptovol.snapshots`). Quotes come from a known SSVI
surface, with deterministic bid-ask spreads and a little mark-IV noise, so
tests can check that the pipeline recovers the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from cryptovol import black76
from cryptovol.snapshots import Snapshot
from cryptovol.ssvi import SSVIParams

SNAPSHOT_TIME = datetime(2026, 1, 2, 8, 0, tzinfo=timezone.utc)
EXPIRY_DAYS = (7, 14, 28, 56, 91, 182, 364)


@dataclass(frozen=True)
class SyntheticMarket:
    currency: str
    spot: float
    carry: float  # annualised basis: F = S exp(carry T)
    strike_step: float
    atm_vol: float  # flat ATM vol used for theta(T) = atm_vol^2 T (slightly upward sloped below)
    rho: float
    eta: float
    gamma: float


MARKETS = (
    SyntheticMarket("BTC", 90_000.0, 0.06, 1_000.0, 0.50, -0.25, 1.0, 0.45),
    SyntheticMarket("ETH", 3_000.0, 0.05, 25.0, 0.65, -0.15, 1.2, 0.40),
)


def true_ssvi(m: SyntheticMarket) -> SSVIParams:
    T = np.array(EXPIRY_DAYS, dtype=np.float64) / 365.0
    atm = m.atm_vol * (1.0 + 0.05 * np.sqrt(T))  # gently upward-sloping ATM vol
    return SSVIParams(rho=m.rho, eta=m.eta, gamma=m.gamma, T=T, theta=atm**2 * T)


def _expiry_code(d: datetime) -> str:
    return f"{d.day}{d.strftime('%b').upper()}{d.strftime('%y')}"


def make_synthetic_snapshot(seed: int = 7) -> Snapshot:
    """Build a deterministic synthetic snapshot for BTC and ETH."""
    rng = np.random.default_rng(seed)
    t0_ms = int(SNAPSHOT_TIME.timestamp() * 1000)
    opt_rows, fut_rows, index = [], [], {}
    for m in MARKETS:
        index[m.currency] = m.spot
        p = true_ssvi(m)
        for days, T in zip(EXPIRY_DAYS, p.T, strict=True):
            exp = SNAPSHOT_TIME + timedelta(days=days)
            exp_ms = int(exp.timestamp() * 1000)
            code = _expiry_code(exp)
            F = m.spot * np.exp(m.carry * T)
            fut_half = 0.25e-4 * F
            fut_rows.append(
                {
                    "instrument_name": f"{m.currency}-{code}",
                    "currency": m.currency,
                    "expiration_timestamp": exp_ms,
                    "bid_price": F - fut_half,
                    "ask_price": F + fut_half,
                    "mark_price": F,
                    "open_interest": 1e6,
                    "volume": 100.0,
                    "creation_timestamp": t0_ms,
                }
            )
            sd = m.atm_vol * np.sqrt(T)
            lo, hi = F * np.exp(-3.0 * sd), F * np.exp(3.0 * sd)
            strikes = np.arange(np.ceil(lo / m.strike_step), np.floor(hi / m.strike_step) + 1)
            strikes = strikes * m.strike_step
            for K in strikes:
                k = np.log(K / F)
                iv = float(p.implied_vol(k, T))
                for cp in ("call", "put"):
                    is_call = cp == "call"
                    px = float(black76.price(F, K, T, iv, is_call)) / F  # coin
                    vega_coin = float(black76.vega(F, K, T, iv)) / F
                    half = max(0.004 * vega_coin, 0.00005)  # ~0.4 vol pt half-spread
                    bid = np.floor((px - half) / 1e-4) * 1e-4
                    ask = np.ceil((px + half) / 1e-4) * 1e-4
                    bid = bid if bid > 0 else 0.0
                    mark_iv = 100.0 * (iv + rng.normal(0.0, 0.001))
                    opt_rows.append(
                        {
                            "instrument_name": f"{m.currency}-{code}-{int(K)}-{cp[0].upper()}",
                            "currency": m.currency,
                            "expiration_timestamp": exp_ms,
                            "strike": float(K),
                            "option_type": cp,
                            "bid_price": bid,
                            "ask_price": ask,
                            "mid_price": 0.5 * (bid + ask) if bid > 0 else np.nan,
                            "mark_price": px,
                            "mark_iv": mark_iv,
                            "underlying_price": F,
                            "underlying_index": f"{m.currency}-{code}",
                            "open_interest": float(rng.integers(1, 500)),
                            "volume": float(rng.integers(0, 50)),
                            "creation_timestamp": t0_ms,
                        }
                    )
        fut_rows.append(
            {
                "instrument_name": f"{m.currency}-PERPETUAL",
                "currency": m.currency,
                "expiration_timestamp": 32503680000000,
                "bid_price": m.spot - 0.5,
                "ask_price": m.spot + 0.5,
                "mark_price": m.spot,
                "open_interest": 1e7,
                "volume": 1e4,
                "creation_timestamp": t0_ms,
            }
        )
    return Snapshot(
        timestamp_ms=t0_ms,
        options=pd.DataFrame(opt_rows),
        futures=pd.DataFrame(fut_rows),
        index_prices=index,
        meta={"source": "synthetic (cryptovol.synthetic, known SSVI ground truth)", "seed": seed},
    )

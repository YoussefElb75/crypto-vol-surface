"""From raw snapshot rows to a clean OTM smile per expiry.

Steps
-----
1. Year fraction ``T`` (ACT/365, to Deribit's 08:00 UTC expiry).
2. Forward ``F`` per expiry: mid of the matching dated future when one exists,
   otherwise put-call parity on near-the-money strikes. Deribit quotes
   options in coin, so with ``df = 1`` parity reads
   ``(C - P)_coin = 1 - K/F``, which gives ``F = K / (1 - (C - P)_coin)``.
3. Log-moneyness ``k = ln(K/F)``. Only out-of-the-money options are kept
   (puts for ``K < F``, calls for ``K >= F``).
4. Implied vols from bid, ask and mid (Black-76, coin price times F), next to
   Deribit's mark IV.
5. Quality filters: missing bid or ask, crossed or locked books, prices
   outside the no-arbitrage bounds, wide spreads, stale (inactive) quotes,
   very short expiries, far wings and expiries with too few quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cryptovol import black76
from cryptovol.snapshots import Snapshot

YEAR_MS = 365.0 * 24 * 3600 * 1000


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds for the quote filters."""

    min_days: float = 2.0  # drop expiries closer than this
    max_rel_spread: float = 0.35  # (ask - bid) / mid
    max_iv_spread: float = 0.10  # ask_iv - bid_iv, in vol (0.10 = 10 vol points)
    min_abs_delta: float = 0.02  # drop far wings (forward delta from mark IV)
    drop_stale: bool = True  # drop rows with no open interest and no volume
    min_quotes: int = 6  # drop expiries with fewer clean quotes


def year_fraction(expiry_ms: pd.Series | np.ndarray, now_ms: int) -> np.ndarray:
    """ACT/365 year fraction between ``now_ms`` and each expiry timestamp."""
    return (np.asarray(expiry_ms, dtype=np.float64) - float(now_ms)) / YEAR_MS


def parity_forward(options: pd.DataFrame, f_guess: float, n_strikes: int = 4) -> float:
    """Forward implied from coin-quoted call/put mids at the strikes nearest ``f_guess``.

    Uses ``F = K / (1 - (C - P))`` and returns the median over the
    ``n_strikes`` strikes closest to the guess that have two-sided markets
    for both the call and the put.
    """
    o = options[(options["bid_price"] > 0) & (options["ask_price"] > 0)].copy()
    o["mid"] = 0.5 * (o["bid_price"] + o["ask_price"])
    piv = o.pivot_table(index="strike", columns="option_type", values="mid", aggfunc="first")
    if not {"call", "put"} <= set(piv.columns):
        return float("nan")
    piv = piv.dropna(subset=["call", "put"])
    if piv.empty:
        return float("nan")
    near = piv.iloc[np.argsort(np.abs(piv.index.to_numpy() - f_guess))[:n_strikes]]
    diff = (near["call"] - near["put"]).to_numpy()
    strikes = near.index.to_numpy(dtype=np.float64)
    fwd = strikes / (1.0 - diff)
    fwd = fwd[np.isfinite(fwd) & (fwd > 0)]
    return float(np.median(fwd)) if fwd.size else float("nan")


def forwards_table(snap: Snapshot, currency: str) -> pd.DataFrame:
    """Forward per option expiry: futures mid, parity forward and the chosen value."""
    opts = snap.options[snap.options["currency"] == currency]
    futs = snap.futures[
        (snap.futures["currency"] == currency)
        & ~snap.futures["instrument_name"].str.contains("PERPETUAL")
    ].copy()
    two_sided = (futs["bid_price"] > 0) & (futs["ask_price"] > 0)
    futs["fut_mid"] = np.where(
        two_sided, 0.5 * (futs["bid_price"] + futs["ask_price"]), futs["mark_price"]
    )
    fut_by_exp = futs.set_index("expiration_timestamp")["fut_mid"]
    spot = snap.index_prices[currency]
    rows = []
    for exp, grp in opts.groupby("expiration_timestamp"):
        exp_ms = int(exp)  # type: ignore[call-overload]
        fut = float(fut_by_exp.get(exp_ms, np.nan))
        guess = fut if np.isfinite(fut) else float(grp["underlying_price"].median())
        par = parity_forward(grp, guess)
        use_fut = np.isfinite(fut)
        rows.append(
            {
                "expiration_timestamp": exp_ms,
                "T": float(year_fraction(np.array([exp_ms]), snap.timestamp_ms)[0]),
                "future_mid": fut,
                "parity_forward": par,
                "deribit_underlying": float(grp["underlying_price"].median()),
                "forward": fut if use_fut else par,
                "forward_source": "future" if use_fut else "parity",
            }
        )
    out = pd.DataFrame(rows).sort_values("expiration_timestamp").reset_index(drop=True)
    out["basis_ann"] = np.log(out["forward"] / spot) / out["T"]  # implied carry, per year
    return out


def clean_options(
    snap: Snapshot, currency: str, config: CleaningConfig | None = None
) -> pd.DataFrame:
    """Return one row per OTM option with IVs, flags and a ``keep`` mask.

    Columns: ``expiry`` (UTC timestamp), ``expiration_timestamp``, ``T``, ``F``,
    ``K``, ``k``, ``is_call``, ``bid``/``ask``/``mid`` (coin), ``bid_iv``,
    ``ask_iv``, ``mid_iv``, ``mark_iv`` (decimals), ``iv_spread``,
    ``rel_spread``, ``flag`` (first failed filter, or ``"ok"``) and ``keep``.
    """
    cfg = config or CleaningConfig()
    fwd = forwards_table(snap, currency)
    o = snap.options[snap.options["currency"] == currency].merge(
        fwd[["expiration_timestamp", "T", "forward"]], on="expiration_timestamp", how="left"
    )
    o = o.rename(columns={"forward": "F", "strike": "K"})
    o["is_call"] = o["option_type"].eq("call")
    o = o[o["is_call"] == (o["K"] >= o["F"])].copy()  # OTM only
    o["k"] = np.log(o["K"] / o["F"])
    o["bid"] = o["bid_price"].where(o["bid_price"] > 0)
    o["ask"] = o["ask_price"].where(o["ask_price"] > 0)
    o["mid"] = 0.5 * (o["bid"] + o["ask"])
    F, K, T, call = o["F"].to_numpy(), o["K"].to_numpy(), o["T"].to_numpy(), o["is_call"].to_numpy()
    for side in ("bid", "ask", "mid"):
        usd = black76.coin_to_usd(o[side].to_numpy(dtype=np.float64), F)
        o[f"{side}_iv"] = black76.implied_vol(usd, F, K, np.maximum(T, 1e-8), call)
    o["mark_iv"] = o["mark_iv"].astype(np.float64) / 100.0
    o["iv_spread"] = o["ask_iv"] - o["bid_iv"]
    o["rel_spread"] = (o["ask"] - o["bid"]) / o["mid"]
    fdelta = black76.delta(F, K, np.maximum(T, 1e-8), o["mark_iv"].to_numpy(), call)

    checks = [
        ("short_expiry", o["T"] * 365.0 < cfg.min_days),
        ("no_bid", o["bid"].isna()),
        ("no_ask", o["ask"].isna()),
        ("crossed_book", o["ask"] <= o["bid"]),
        ("no_iv", o["mid_iv"].isna()),
        (
            "wide_spread",
            (o["rel_spread"] > cfg.max_rel_spread) | (o["iv_spread"] > cfg.max_iv_spread),
        ),
        (
            "stale",
            (o["open_interest"].fillna(0) <= 0) & (o["volume"].fillna(0) <= 0)
            if cfg.drop_stale
            else pd.Series(False, index=o.index),
        ),
        ("far_wing", np.abs(fdelta) < cfg.min_abs_delta),
    ]
    flag = pd.Series("ok", index=o.index, dtype=object)
    for name, mask in reversed(checks):  # the first failed check wins
        flag = flag.where(~np.asarray(mask, dtype=bool), name)
    o["flag"] = flag
    o["keep"] = o["flag"].eq("ok")
    counts = o[o["keep"]].groupby("expiration_timestamp")["K"].transform("size")
    o.loc[counts.index[counts < cfg.min_quotes], ["keep", "flag"]] = [False, "thin_expiry"]
    o["expiry"] = pd.to_datetime(o["expiration_timestamp"], unit="ms", utc=True)
    cols = [
        "instrument_name",
        "expiry",
        "expiration_timestamp",
        "T",
        "F",
        "K",
        "k",
        "is_call",
        "bid",
        "ask",
        "mid",
        "mark_price",
        "bid_iv",
        "ask_iv",
        "mid_iv",
        "mark_iv",
        "iv_spread",
        "rel_spread",
        "open_interest",
        "volume",
        "flag",
        "keep",
    ]
    return o[cols].sort_values(["expiration_timestamp", "K"]).reset_index(drop=True)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptovol.cleaning import (
    CleaningConfig,
    clean_options,
    forwards_table,
    parity_forward,
    year_fraction,
)
from cryptovol.snapshots import Snapshot
from cryptovol.synthetic import MARKETS, true_ssvi


def test_year_fraction_is_act_365() -> None:
    assert year_fraction(np.array([365 * 86_400_000]), 0)[0] == pytest.approx(1.0)


@pytest.mark.parametrize("m", MARKETS, ids=lambda m: m.currency)
def test_forwards_from_futures_and_parity(snap: Snapshot, m) -> None:  # type: ignore[no-untyped-def]
    fwd = forwards_table(snap, m.currency)
    expected = m.spot * np.exp(m.carry * fwd["T"].to_numpy())
    np.testing.assert_allclose(fwd["forward"], expected, rtol=1e-9)
    assert (fwd["forward_source"] == "future").all()
    # parity on coin quotes recovers the forward up to quote rounding
    np.testing.assert_allclose(fwd["parity_forward"], expected, rtol=2e-3)
    np.testing.assert_allclose(fwd["basis_ann"], m.carry, rtol=1e-6)


def test_parity_forward_used_when_future_missing(snap: Snapshot) -> None:
    futs = snap.futures[snap.futures["instrument_name"] != "BTC-9JAN26"]
    s2 = Snapshot(snap.timestamp_ms, snap.options, futs, snap.index_prices)
    fwd = forwards_table(s2, "BTC")
    row = fwd.iloc[0]
    assert row["forward_source"] == "parity"
    assert row["forward"] == pytest.approx(
        MARKETS[0].spot * np.exp(MARKETS[0].carry * row["T"]), rel=2e-3
    )


def test_parity_forward_closed_form() -> None:
    F, K = 100.0, 95.0
    c = 0.08
    p = c - (1 - K / F)  # (C - P)_coin = 1 - K/F
    df = pd.DataFrame(
        {"strike": [K, K], "option_type": ["call", "put"], "bid_price": [c, p], "ask_price": [c, p]}
    )
    assert parity_forward(df, 100.0) == pytest.approx(F)


def test_clean_quotes_are_otm_with_correct_moneyness(quotes_btc) -> None:  # type: ignore[no-untyped-def]
    q = quotes_btc
    assert (q["is_call"] == (q["K"] >= q["F"])).all()
    np.testing.assert_allclose(q["k"], np.log(q["K"] / q["F"]))
    kept = q[q["keep"]]
    assert (kept["bid"] > 0).all()
    assert (kept["ask"] > kept["bid"]).all()
    assert kept["rel_spread"].max() <= CleaningConfig().max_rel_spread


def test_mid_iv_close_to_truth_and_mark(quotes_btc) -> None:  # type: ignore[no-untyped-def]
    q = quotes_btc[quotes_btc["keep"]]
    truth = true_ssvi(MARKETS[0])
    iv_true = truth.implied_vol(q["k"].to_numpy(), q["T"].to_numpy())
    assert np.abs(q["mid_iv"] - iv_true).max() < 0.01  # < 1 vol pt (tick rounding)
    assert np.abs(q["mark_iv"] - iv_true).max() < 0.006  # mark noise ~0.1 vol pt
    assert ((q["bid_iv"] <= q["mid_iv"]) & (q["mid_iv"] <= q["ask_iv"])).all()


def _otm_call_names(snap: Snapshot, n: int) -> list[str]:
    """Names of liquid OTM BTC calls in the 91-day expiry (pass every filter by default)."""
    o = snap.options
    fut = snap.futures.set_index("instrument_name")["mark_price"]
    sel = o[
        (o["currency"] == "BTC")
        & (o["option_type"] == "call")
        & o["instrument_name"].str.startswith("BTC-3APR26")
    ]
    F = float(fut["BTC-3APR26"])
    sel = sel[(sel["strike"] > F) & (sel["strike"] < 1.15 * F)]
    return sel["instrument_name"].tolist()[:n]


def test_filters_flag_bad_quotes(snap: Snapshot) -> None:
    names = _otm_call_names(snap, 6)
    o = snap.options.copy()
    by_name = o.set_index("instrument_name").index
    rows = [int(np.flatnonzero(by_name == n)[0]) for n in names]
    i_wide, i_stale, i_nobid, i_noask, i_crossed, i_noiv = rows
    o.loc[i_wide, "ask_price"] = o.loc[i_wide, "bid_price"] * 3 + 0.01
    o.loc[i_stale, ["open_interest", "volume"]] = 0.0
    o.loc[i_nobid, "bid_price"] = 0.0
    o.loc[i_noask, "ask_price"] = np.nan
    o.loc[i_crossed, "ask_price"] = o.loc[i_crossed, "bid_price"]  # locked book
    o.loc[i_noiv, ["bid_price", "ask_price"]] = [1.5, 1.6]  # call worth more than 1 coin
    q = clean_options(Snapshot(snap.timestamp_ms, o, snap.futures, snap.index_prices), "BTC")
    q = q.set_index("instrument_name")
    base = clean_options(snap, "BTC").set_index("instrument_name")
    assert base.loc[names, "keep"].all()
    expected = ["wide_spread", "stale", "no_bid", "no_ask", "crossed_book", "no_iv"]
    for n, flag in zip(names, expected, strict=True):
        assert q.loc[n, "flag"] == flag
        assert not q.loc[n, "keep"]


def test_far_wing_filter(quotes_btc) -> None:  # type: ignore[no-untyped-def]
    wing = quotes_btc[quotes_btc["flag"] == "far_wing"]
    assert not wing.empty
    assert (wing["k"].abs() > quotes_btc.loc[quotes_btc["keep"], "k"].abs().min()).all()
    assert not quotes_btc.loc[quotes_btc["flag"] == "far_wing", "keep"].any()


def test_thin_expiry_filter(snap: Snapshot) -> None:
    q = clean_options(snap, "BTC", CleaningConfig(min_quotes=50))
    thin = q[q["flag"] == "thin_expiry"]
    assert thin["expiration_timestamp"].nunique() >= 1
    assert not thin["keep"].any()
    assert q["keep"].dtype == bool
    counts = q[q["keep"]].groupby("expiration_timestamp").size()
    assert (counts >= 50).all()


def test_parity_forward_needs_both_sides() -> None:
    df = pd.DataFrame(
        {
            "strike": [95.0, 100.0],
            "option_type": ["call", "call"],
            "bid_price": [0.08, 0.05],
            "ask_price": [0.09, 0.06],
        }
    )
    assert np.isnan(parity_forward(df, 100.0))


def test_short_expiry_filter(snap: Snapshot) -> None:
    q = clean_options(snap, "BTC", CleaningConfig(min_days=10))
    short = q["T"] * 365 < 10
    assert short.any()
    assert not q.loc[short, "keep"].any()
    assert (q.loc[short, "flag"] == "short_expiry").all()

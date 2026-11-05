from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptovol.report import analyse, market_atm, write_derived
from cryptovol.snapshots import Snapshot
from cryptovol.synthetic import MARKETS, true_ssvi


@pytest.fixture(scope="module")
def report_btc(snap: Snapshot):  # type: ignore[no-untyped-def]
    return analyse(snap, "BTC")


def test_analyse_end_to_end(report_btc) -> None:  # type: ignore[no-untyped-def]
    r = report_btc
    n_exp = len(r.fits.expiries)
    assert len(r.comparison) == n_exp
    assert set(r.rr_bf["model"]) == {"svi", "ssvi"}
    assert set(r.rr_bf["convention"]) == {"forward", "premium_adjusted"}
    assert len(r.rr_bf) == 4 * n_exp
    assert r.svi_calendar_gaps.shape == (n_exp - 1,)
    assert r.svi_calendar_gaps_in_range.shape == (n_exp - 1,)
    assert (r.svi_calendar_gaps_in_range > 0).all()  # synthetic truth is calendar-free
    assert bool(r.ssvi_params()["arbitrage_free"].iloc[0])
    truth = true_ssvi(MARKETS[0])
    np.testing.assert_allclose(r.atm["ssvi"], np.sqrt(truth.theta / truth.T), atol=2e-3)
    np.testing.assert_allclose(r.atm["mark"], r.atm["ssvi"], atol=5e-3)


def test_write_derived_has_no_per_quote_data(report_btc, tmp_path) -> None:  # type: ignore[no-untyped-def]
    paths = write_derived(report_btc, tmp_path)
    assert {p.name for p in paths} == {
        "btc_fit_comparison.csv",
        "btc_svi_params.csv",
        "btc_ssvi_params.csv",
        "btc_atm_term_structure.csv",
        "btc_rr_bf.csv",
    }
    forbidden = {"bid", "ask", "mid", "mark_price", "instrument_name", "K", "F", "future_mid"}
    for p in paths:
        df = pd.read_csv(p)
        assert not forbidden & set(df.columns), p.name
        assert len(df) <= len(report_btc.rr_bf)  # aggregated: at most one row per expiry/variant


def test_market_atm_does_not_extrapolate() -> None:
    q = pd.DataFrame(
        {
            "expiry": ["a"] * 3 + ["b"] * 3,
            "k": [-0.2, -0.1, 0.1, 0.05, 0.1, 0.2],  # expiry b never brackets k = 0
            "mark_iv": [0.5, 0.45, 0.47, 0.4, 0.41, 0.43],
            "keep": True,
        }
    )
    atm = market_atm(q, "mark_iv")
    assert atm["a"] == pytest.approx(0.46)
    assert np.isnan(atm["b"])

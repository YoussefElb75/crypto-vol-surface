from __future__ import annotations

import numpy as np
import pytest

from cryptovol import black76
from cryptovol.analytics import (
    atm_term_structure,
    delta_strike,
    mark_errors,
    reprice,
    reprice_summary,
    rr_bf,
)
from cryptovol.surface import svi_calendar_gaps
from cryptovol.synthetic import MARKETS, true_ssvi


@pytest.mark.parametrize("model", ["svi", "ssvi"])
def test_repricing_within_bid_ask(quotes_btc, fits_btc, model) -> None:  # type: ignore[no-untyped-def]
    summary = reprice_summary(reprice(quotes_btc, getattr(fits_btc, model)))
    assert (summary["inside_share"] >= 0.99).all()


@pytest.mark.parametrize("model", ["svi", "ssvi"])
def test_surface_call_prices_are_arbitrage_free_in_strike(fits_btc, model) -> None:  # type: ignore[no-untyped-def]
    # Static no-arbitrage in K at fixed T: calls decrease in K, are convex, and the
    # slope stays in [-1, 0]. Tested at nodes and between nodes.
    s = getattr(fits_btc, model)
    T_nodes = s.T_nodes
    Ts = list(T_nodes) + list(0.5 * (T_nodes[:-1] + T_nodes[1:]))
    for T in Ts:
        F = float(s.forward(T))
        K = F * np.exp(np.linspace(-0.8, 0.8, 161))
        C = s.price(K, T, True)
        slope = np.diff(C) / np.diff(K)
        assert (slope <= 1e-12).all()
        assert (slope >= -1 - 1e-12).all()
        assert (np.diff(slope) >= -1e-12).all()


def test_coin_price_is_usd_over_forward(fits_btc) -> None:  # type: ignore[no-untyped-def]
    s = fits_btc.ssvi
    K, T = np.array([80_000.0, 95_000.0, 110_000.0]), 0.3
    np.testing.assert_allclose(
        s.price(K, T, False, unit="coin") * s.forward(T), s.price(K, T, False)
    )


def test_svi_interpolation_between_nodes_is_linear_in_total_variance(fits_btc) -> None:  # type: ignore[no-untyped-def]
    s = fits_btc.svi
    k = np.linspace(-0.4, 0.4, 9)
    T0, T1 = float(s.T_nodes[2]), float(s.T_nodes[3])
    a = 0.3
    Tm = (1 - a) * T0 + a * T1
    expected = (1 - a) * s.svi[2].params.total_variance(k) + a * s.svi[3].params.total_variance(k)
    np.testing.assert_allclose(s.total_variance(k, Tm), expected)
    # before the first node, total variance scales with T
    np.testing.assert_allclose(
        s.total_variance(k, 0.5 * float(s.T_nodes[0])), 0.5 * s.svi[0].params.total_variance(k)
    )


def test_forward_curve_interpolation(fits_btc) -> None:  # type: ignore[no-untyped-def]
    m = MARKETS[0]
    T = np.linspace(0.01, 1.5, 50)
    np.testing.assert_allclose(fits_btc.ssvi.forward(T), m.spot * np.exp(m.carry * T), rtol=1e-9)


def test_svi_surface_passes_through_slices(fits_btc) -> None:  # type: ignore[no-untyped-def]
    s = fits_btc.svi
    k = np.linspace(-0.3, 0.3, 7)
    for fit, T in zip(s.svi, s.T_nodes, strict=True):
        np.testing.assert_allclose(s.total_variance(k, T), fit.params.total_variance(k))
    assert svi_calendar_gaps(s, np.linspace(-0.3, 0.3, 61)).min() > 0  # no crossing in-range


def test_atm_term_structure_matches_truth(fits_btc) -> None:  # type: ignore[no-untyped-def]
    truth = true_ssvi(MARKETS[0])
    atm = atm_term_structure(fits_btc.ssvi)
    np.testing.assert_allclose(atm["atm_iv"], np.sqrt(truth.theta / truth.T), atol=2e-3)


@pytest.mark.parametrize("model", ["svi", "ssvi"])
@pytest.mark.parametrize("convention", ["forward", "premium_adjusted"])
def test_delta_strikes_hit_target(fits_btc, model, convention) -> None:  # type: ignore[no-untyped-def]
    s = getattr(fits_btc, model)
    for T in s.T_nodes:
        t = float(T)
        F = float(s.forward(t))
        for target, call in ((0.25, True), (-0.25, False)):
            k = delta_strike(s, t, target, call, convention)
            K = F * np.exp(k)
            sig = float(s.iv(k, t))
            d = (
                black76.delta(F, K, t, sig, call)
                if convention == "forward"
                else black76.premium_adjusted_delta(F, K, t, sig, call)
            )
            assert float(d) == pytest.approx(target, abs=1e-8)


def test_premium_adjustment_moves_strikes_as_expected(fits_btc) -> None:  # type: ignore[no-untyped-def]
    fwd = rr_bf(fits_btc.ssvi, "forward")
    pa = rr_bf(fits_btc.ssvi, "premium_adjusted")
    assert (pa["k_call"] < fwd["k_call"]).all()  # 25d PA call closer to the money
    assert (pa["k_put"] < fwd["k_put"]).all()  # 25d PA put further OTM
    assert (pa["k_call"] > 0).all()  # OTM branch of the non-monotone PA call delta
    assert (fwd["rr"] < 0).all()  # rho < 0 in the synthetic BTC surface: put skew
    assert (fwd["bf"] > 0).all()


def test_errors_vs_mark_are_small(quotes_btc, fits_btc) -> None:  # type: ignore[no-untyped-def]
    me = mark_errors(quotes_btc, fits_btc.ssvi)
    assert me["err_vs_mark_volpts"].abs().median() < 0.2
    assert me["err_vs_mid_volpts"].abs().max() < 1.0


def test_greeks_table(fits_btc) -> None:  # type: ignore[no-untyped-def]
    g = fits_btc.ssvi.greeks([80_000, 100_000], [0.25, 0.25], [False, True])
    assert list(g["delta"] < 0) == [True, False]
    np.testing.assert_allclose(g["delta_pa"], g["delta"] - g["price_usd"] / g["F"])
    assert (g["gamma"] > 0).all()
    assert (g["vega_1pct"] > 0).all()
    assert (g["theta_1d"] < 0).all()


def test_unreachable_premium_adjusted_delta_returns_nan(fits_btc) -> None:  # type: ignore[no-untyped-def]
    # The premium-adjusted call delta (K/F) N(d2) peaks below 1, so 0.99 is never reached.
    s = fits_btc.ssvi
    assert np.isnan(delta_strike(s, float(s.T_nodes[-1]), 0.99, True, "premium_adjusted"))
    # the forward call delta can reach 0.99 (deep ITM)
    assert np.isfinite(delta_strike(s, float(s.T_nodes[-1]), 0.99, True, "forward"))


def test_greeks_broadcast_scalar_strike_with_array_flags(fits_btc) -> None:  # type: ignore[no-untyped-def]
    g = fits_btc.ssvi.greeks(95_000.0, 0.25, [True, False])
    assert len(g) == 2
    np.testing.assert_allclose(g["delta"].iloc[0] - g["delta"].iloc[1], 1.0)

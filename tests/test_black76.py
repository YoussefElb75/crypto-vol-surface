from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from cryptovol import black76 as b

rng = np.random.default_rng(0)
N = 400
F = rng.uniform(50, 150, N)
K = F * np.exp(rng.uniform(-1.0, 1.0, N))
T = rng.uniform(0.005, 2.0, N)
SIG = rng.uniform(0.1, 1.5, N)
DF = np.exp(-rng.uniform(0.0, 0.08, N) * T)
CALL = rng.random(N) < 0.5


def test_known_value_atm() -> None:
    expected = 100.0 * (2.0 * norm.cdf(0.1) - 1.0)
    assert float(b.price(100.0, 100.0, 1.0, 0.2, True)) == pytest.approx(expected, rel=1e-12)


def test_put_call_parity() -> None:
    c = b.price(F, K, T, SIG, True, DF)
    p = b.price(F, K, T, SIG, False, DF)
    np.testing.assert_allclose(c - p, DF * (F - K), atol=1e-10)


def test_implied_vol_round_trip() -> None:
    v = b.price(F, K, T, SIG, CALL, DF)
    iv = b.implied_vol(v, F, K, T, CALL, DF)
    # skip prices too close to intrinsic to carry information in float64
    informative = b.vega(F, K, T, SIG, DF) > 1e-6
    np.testing.assert_allclose(iv[informative], SIG[informative], rtol=1e-7)


def test_implied_vol_rejects_arbitrage_prices() -> None:
    assert np.isnan(b.implied_vol(-1.0, 100, 100, 1.0, True))
    assert np.isnan(b.implied_vol(0.5, 100, 90, 1.0, True))  # below intrinsic 10
    assert np.isnan(b.implied_vol(101.0, 100, 90, 1.0, True))  # above F
    assert np.isnan(b.implied_vol(91.0, 100, 90, 1.0, False))  # put above K


@pytest.mark.parametrize("call", [True, False])
def test_greeks_match_finite_differences(call: bool) -> None:
    f, k, t, s, df = 100.0, 110.0, 0.5, 0.6, 0.98
    h = 1e-4

    def p(f_: float = f, t_: float = t, s_: float = s, df_: float = df) -> float:
        return float(b.price(f_, k, t_, s_, call, df_))

    assert float(b.delta(f, k, t, s, call, df)) == pytest.approx(
        (p(f + h) - p(f - h)) / (2 * h), rel=1e-6
    )
    assert float(b.gamma(f, k, t, s, df)) == pytest.approx(
        (p(f + 1e-2) - 2 * p() + p(f - 1e-2)) / 1e-4, rel=1e-4
    )
    assert float(b.vega(f, k, t, s, df)) == pytest.approx(
        (p(s_=s + h) - p(s_=s - h)) / (2 * h), rel=1e-6
    )
    # theta at constant rate: df(T) = exp(-r T)
    r = -np.log(df) / t

    def p_t(t_: float) -> float:
        return p(t_=t_, df_=float(np.exp(-r * t_)))

    assert float(b.theta(f, k, t, s, call, df)) == pytest.approx(
        -(p_t(t + h) - p_t(t - h)) / (2 * h), rel=1e-5
    )


def test_premium_adjusted_delta_identity() -> None:
    v = b.price(F, K, T, SIG, CALL)
    np.testing.assert_allclose(
        b.premium_adjusted_delta(F, K, T, SIG, CALL),
        b.delta(F, K, T, SIG, CALL) - v / F,
        atol=1e-12,
    )


def test_coin_usd_round_trip() -> None:
    np.testing.assert_allclose(b.coin_to_usd(b.usd_to_coin(123.4, 98765.0), 98765.0), 123.4)


def test_implied_vol_outside_search_bracket_is_nan() -> None:
    # A price that needs sigma = 12 (above the default hi = 10) must not come back as a number.
    v = b.price(100.0, 100.0, 1.0, 12.0, True)
    assert np.isnan(b.implied_vol(v, 100.0, 100.0, 1.0, True))
    assert float(b.implied_vol(v, 100.0, 100.0, 1.0, True, hi=20.0)) == pytest.approx(12.0)

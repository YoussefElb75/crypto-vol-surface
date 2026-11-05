from __future__ import annotations

import numpy as np
import pytest

from cryptovol.ssvi import SSVIParams, check_ssvi
from cryptovol.synthetic import MARKETS, true_ssvi


def _params(rho: float, eta: float, gamma: float, theta: list[float]) -> SSVIParams:
    th = np.asarray(theta)
    return SSVIParams(rho=rho, eta=eta, gamma=gamma, T=np.linspace(0.1, 1.0, th.size), theta=th)


def test_true_synthetic_surfaces_are_arbitrage_free() -> None:
    for m in MARKETS:
        chk = check_ssvi(true_ssvi(m))
        assert chk.ok, chk


def test_butterfly_violation_is_detected() -> None:
    p = _params(-0.5, 5.0, 0.5, [0.05, 0.2, 0.5, 1.0])  # eta (1+|rho|) = 7.5 > 2
    chk = check_ssvi(p)
    assert chk.butterfly_bound_1 >= 4.0 or chk.butterfly_bound_2 > 4.0
    assert not chk.ok


def test_calendar_violation_is_detected() -> None:
    p = _params(-0.3, 1.0, 0.4, [0.05, 0.10, 0.08, 0.20])  # theta decreases
    chk = check_ssvi(p)
    assert not chk.theta_monotone
    assert chk.min_calendar_gap < 0
    assert not chk.ok


def test_sufficient_condition_bounds_hold_for_all_theta() -> None:
    # eta (1 + |rho|) <= 2 and gamma <= 1/2 imply both GJ bounds for any theta
    rho, gamma = -0.7, 0.5
    eta = 2.0 / (1.0 + abs(rho))
    p = _params(rho, eta, gamma, [1.0])
    th = np.logspace(-6, 3, 500)
    ph = p.phi(th)
    assert (th * ph * (1 + abs(rho)) < 4).all()
    assert (th * ph**2 * (1 + abs(rho)) <= 4 + 1e-12).all()


def test_derivatives_match_finite_differences() -> None:
    p = true_ssvi(MARKETS[0])
    k = np.linspace(-1, 1, 21)
    T = 0.3
    w, w1, w2 = p.derivatives(k, T)
    h = 1e-5
    np.testing.assert_allclose(w, p.total_variance(k, T))
    fd1 = (p.total_variance(k + h, T) - p.total_variance(k - h, T)) / (2 * h)
    np.testing.assert_allclose(w1, fd1, rtol=1e-6)
    fd2 = (p.derivatives(k + h, T)[1] - p.derivatives(k - h, T)[1]) / (2 * h)
    np.testing.assert_allclose(w2, fd2, rtol=1e-5)


def test_interpolated_surface_has_no_calendar_or_butterfly_arbitrage() -> None:
    p = true_ssvi(MARKETS[1])
    k = np.linspace(-1.5, 1.5, 121)
    Ts = np.linspace(0.002, 1.5, 300)  # inside, before and beyond the fitted nodes
    W = np.array([p.total_variance(k, t) for t in Ts])
    assert (np.diff(W, axis=0) >= -1e-14).all()
    assert min(p.g(k, float(t)).min() for t in Ts) >= 0


@pytest.mark.parametrize("ccy", ["BTC", "ETH"])
def test_fit_recovers_true_parameters(ccy, fits_btc, fits_eth) -> None:  # type: ignore[no-untyped-def]
    fits = fits_btc if ccy == "BTC" else fits_eth
    m = next(x for x in MARKETS if x.currency == ccy)
    truth = true_ssvi(m)
    fitted = fits.ssvi.ssvi
    assert fitted is not None
    assert fitted.check.ok
    assert fitted.params.rho == pytest.approx(truth.rho, abs=0.01)
    assert fitted.params.eta == pytest.approx(truth.eta, rel=0.02)
    assert fitted.params.gamma == pytest.approx(truth.gamma, abs=0.02)
    np.testing.assert_allclose(fitted.params.theta, truth.theta, rtol=5e-3)
    assert fitted.rmse_volpts < 0.3

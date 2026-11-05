from __future__ import annotations

import numpy as np
import pytest

from cryptovol.svi import SVIParams, check_butterfly, fit_svi

# Axel Vogt's raw-SVI slice (T = 1) from Gatheral & Jacquier (2014): g(k) < 0 near k = 0.
VOGT = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
GOOD = SVIParams(a=0.02, b=0.08, rho=-0.4, m=0.05, sigma=0.15)


def test_flat_smile_has_g_equal_one() -> None:
    p = SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.1)
    np.testing.assert_allclose(p.g(np.linspace(-1, 1, 11)), 1.0)


def test_derivatives_match_finite_differences() -> None:
    k = np.linspace(-1, 1, 21)
    h = 1e-5
    np.testing.assert_allclose(
        GOOD.dw(k), (GOOD.total_variance(k + h) - GOOD.total_variance(k - h)) / (2 * h), rtol=1e-6
    )
    np.testing.assert_allclose(GOOD.d2w(k), (GOOD.dw(k + h) - GOOD.dw(k - h)) / (2 * h), rtol=1e-5)


def test_min_total_variance_formula() -> None:
    k = np.linspace(-3, 3, 200_001)
    assert GOOD.min_total_variance() == pytest.approx(GOOD.total_variance(k).min(), abs=1e-9)


def test_vogt_slice_has_butterfly_arbitrage() -> None:
    chk = check_butterfly(VOGT)
    assert chk.min_g < 0
    assert not chk.ok


def test_good_slice_is_arbitrage_free_and_density_integrates() -> None:
    chk = check_butterfly(GOOD)
    assert chk.ok
    k = np.linspace(-6, 6, 24001)
    dens = GOOD.density(k)
    assert (dens >= 0).all()
    integral = float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(k)))
    assert integral == pytest.approx(1.0, abs=2e-3)


def test_lee_wing_bound_is_checked() -> None:
    steep = SVIParams(a=0.0, b=1.8, rho=0.5, m=0.0, sigma=0.3)  # b(1+|rho|) = 2.7 > 2
    assert not check_butterfly(steep).ok


def test_fit_recovers_svi_slice() -> None:
    T = 0.5
    k = np.linspace(-0.6, 0.5, 30)
    iv = GOOD.implied_vol(k, T)
    fit = fit_svi(k, iv, T)
    assert fit.rmse_volpts < 0.01
    assert fit.butterfly.ok
    np.testing.assert_allclose(fit.params.implied_vol(k, T), iv, atol=1e-4)


def test_fit_on_arbitrageable_data_returns_arbitrage_free_slice() -> None:
    T = 1.0
    k = np.linspace(-0.6, 0.8, 40)
    iv = VOGT.implied_vol(k, T)
    fit = fit_svi(k, iv, T)
    assert fit.butterfly.ok
    assert fit.rmse_volpts < 3.0  # close, but the arbitrage is removed


def test_fit_on_synthetic_fixture_is_butterfly_free(fits_btc, fits_eth) -> None:  # type: ignore[no-untyped-def]
    for fits in (fits_btc, fits_eth):
        assert fits.expiries["svi_butterfly_ok"].all()
        assert (fits.expiries["svi_rmse_volpts"] < 0.3).all()

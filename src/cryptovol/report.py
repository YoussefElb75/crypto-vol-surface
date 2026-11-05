"""End-to-end pipeline for one currency: clean, fit, analyse, reprice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cryptovol.analytics import (
    atm_term_structure,
    fit_comparison,
    mark_errors,
    reprice,
    rr_bf,
)
from cryptovol.cleaning import CleaningConfig, clean_options, forwards_table
from cryptovol.snapshots import Snapshot
from cryptovol.surface import SurfaceFits, fit_surfaces, svi_calendar_gaps


@dataclass
class CurrencyReport:
    currency: str
    snapshot_time: pd.Timestamp
    forwards: pd.DataFrame
    quotes: pd.DataFrame  # all OTM quotes with flags (keep == used in fits)
    fits: SurfaceFits
    comparison: pd.DataFrame  # per-expiry SVI vs SSVI quality
    atm: pd.DataFrame  # ATM IV per expiry: SVI, SSVI, Deribit mark
    rr_bf: pd.DataFrame  # both delta conventions, both models
    errors_svi: pd.DataFrame  # fitted IV vs mark / mid per quote
    errors_ssvi: pd.DataFrame
    svi_calendar_gaps: np.ndarray  # min_k w_{i+1} - w_i over k in [-1.5, 1.5]
    svi_calendar_gaps_in_range: np.ndarray  # same, on the overlap of both slices' quoted k

    def svi_params(self) -> pd.DataFrame:
        rows = []
        for f, exp in zip(self.fits.svi.svi, self.fits.expiries["expiry"], strict=True):
            p = f.params
            rows.append(
                {
                    "expiry": exp,
                    "T": f.T,
                    "a": p.a,
                    "b": p.b,
                    "rho": p.rho,
                    "m": p.m,
                    "sigma": p.sigma,
                    "min_g": f.butterfly.min_g,
                    "wing_slope": f.butterfly.wing_slope,
                    "butterfly_ok": f.butterfly.ok,
                }
            )
        return pd.DataFrame(rows)

    def ssvi_params(self) -> pd.DataFrame:
        fit = self.fits.ssvi.ssvi
        assert fit is not None
        p, chk = fit.params, fit.check
        return pd.DataFrame([{
            "rho": p.rho, "eta": p.eta, "gamma": p.gamma, "rmse_volpts": fit.rmse_volpts,
            "theta_monotone": chk.theta_monotone, "max_theta_phi_1p_rho": chk.butterfly_bound_1,
            "max_theta_phi2_1p_rho": chk.butterfly_bound_2, "min_g": chk.min_g,
            "min_calendar_gap": chk.min_calendar_gap, "arbitrage_free": chk.ok,
        }])  # fmt: skip


def market_atm(quotes: pd.DataFrame, column: str) -> pd.Series:
    """Market IV interpolated at ``k = 0`` per expiry, from the clean quotes."""
    out = {}
    for exp, g in quotes[quotes["keep"]].groupby("expiry"):
        g = g.sort_values("k")
        k = g["k"].to_numpy()
        inside = len(g) > 1 and k[0] <= 0.0 <= k[-1]  # no extrapolation
        out[exp] = float(np.interp(0.0, k, g[column].to_numpy())) if inside else np.nan
    return pd.Series(out, name=column)


def analyse(snap: Snapshot, currency: str, config: CleaningConfig | None = None) -> CurrencyReport:
    quotes = clean_options(snap, currency, config)
    fits = fit_surfaces(quotes)
    rp_svi, rp_ssvi = reprice(quotes, fits.svi), reprice(quotes, fits.ssvi)
    comparison = fit_comparison(rp_svi, rp_ssvi, fits.expiries)

    atm = atm_term_structure(fits.svi).rename(columns={"atm_iv": "svi"})
    atm["ssvi"] = atm_term_structure(fits.ssvi)["atm_iv"].to_numpy()
    atm["expiry"] = fits.expiries["expiry"].to_numpy()
    atm["mark"] = market_atm(quotes, "mark_iv").reindex(atm["expiry"]).to_numpy()
    atm["mid"] = market_atm(quotes, "mid_iv").reindex(atm["expiry"]).to_numpy()

    frames = []
    for model in ("svi", "ssvi"):
        for conv in ("forward", "premium_adjusted"):
            frames.append(rr_bf(getattr(fits, model), conv).assign(model=model))  # type: ignore[arg-type]
    kept = quotes[quotes["keep"]]
    ranges = kept.groupby("T")["k"].agg(["min", "max"]).sort_index().to_numpy()
    gaps_in = []
    for i in range(len(ranges) - 1):  # overlap of the two slices' quoted k ranges
        lo, hi = max(ranges[i, 0], ranges[i + 1, 0]), min(ranges[i, 1], ranges[i + 1, 1])
        w0 = fits.svi.svi[i].params.total_variance(np.linspace(lo, hi, 201))
        w1 = fits.svi.svi[i + 1].params.total_variance(np.linspace(lo, hi, 201))
        gaps_in.append(float(np.min(w1 - w0)))
    return CurrencyReport(
        currency=currency,
        snapshot_time=pd.Timestamp(snap.time),
        forwards=forwards_table(snap, currency),
        quotes=quotes,
        fits=fits,
        comparison=comparison,
        atm=atm,
        rr_bf=pd.concat(frames, ignore_index=True),
        errors_svi=mark_errors(quotes, fits.svi),
        errors_ssvi=mark_errors(quotes, fits.ssvi),
        svi_calendar_gaps=svi_calendar_gaps(fits.svi),
        svi_calendar_gaps_in_range=np.asarray(gaps_in),
    )


def write_derived(rep: CurrencyReport, out_dir: str | Path) -> list[Path]:
    """Write aggregated, derived results only (no per-quote market data) as CSV."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c = rep.currency.lower()
    tables = {
        f"{c}_fit_comparison.csv": rep.comparison.drop(columns="F"),
        f"{c}_svi_params.csv": rep.svi_params(),
        f"{c}_ssvi_params.csv": rep.ssvi_params(),
        f"{c}_atm_term_structure.csv": rep.atm[["expiry", "days", "svi", "ssvi", "mark"]],
        f"{c}_rr_bf.csv": rep.rr_bf[
            [
                "model",
                "convention",
                "days",
                "iv_atm",
                "iv_call",
                "iv_put",
                "rr",
                "bf",
                "k_call",
                "k_put",
            ]
        ],
    }
    paths = []
    for name, df in tables.items():
        p = out / name
        df.to_csv(p, index=False, float_format="%.6g")
        paths.append(p)
    return paths

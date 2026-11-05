"""Build every README figure (light and dark variants) and the derived CSV tables.

Usage::

    python scripts/make_figures.py                       # latest snapshot in data/snapshots/
    python scripts/make_figures.py --snapshot PATH       # a given snapshot directory
    python scripts/make_figures.py --synthetic           # the committed synthetic fixture

Outputs. With ``--synthetic`` the figures go to ``docs/figures/`` (these are the
figures committed and shown in the README) and the tables to
``data/derived/synthetic/``. With a live snapshot the figures go to
``data/local/figures/`` and the tables to ``data/derived/live/``. Both
``data/local/`` and ``data/derived/`` are git-ignored: Deribit's terms restrict
redistribution of market data and of data derived from it (see DATA.md), so
results from live snapshots stay on your machine. No network access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from cryptovol.plotting import Theme, save_both
from cryptovol.report import CurrencyReport, analyse, write_derived
from cryptovol.snapshots import latest_snapshot_dir, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
DERIVED = ROOT / "data" / "derived" / "synthetic"
SYNTHETIC_STAMP = "synthetic data: generated SSVI surface, not market data"
K_GRID = np.linspace(-1.5, 1.5, 301)


def _stamp(reps: list[CurrencyReport]) -> str:
    return reps[0].snapshot_time.strftime("Deribit snapshot %Y-%m-%d %H:%M UTC")


def _title(fig: plt.Figure, t: Theme, title: str, sub: str) -> None:
    """Left-aligned title; the data source goes in a muted footer (no overlap)."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=t.ink)
    fig.text(
        0.995,
        -0.01,
        sub,
        ha="right",
        va="top",
        fontsize=8,
        color=t.muted,
        transform=fig.transFigure,
    )


# --------------------------------------------------------------------------- 1. surface
def fig_surface(reps: list[CurrencyReport], stamp: str):  # type: ignore[no-untyped-def]
    def draw(t: Theme) -> plt.Figure:
        fig, axes = plt.subplots(1, len(reps), figsize=(11, 4.4), layout="constrained")
        for ax, rep in zip(np.atleast_1d(axes), reps, strict=True):
            s = rep.fits.ssvi
            T_max = float(s.T_nodes[-1])
            days = np.geomspace(float(s.T_nodes[0]), T_max, 160) * 365
            kk = np.linspace(-0.6, 0.6, 121)
            KK, DD = np.meshgrid(kk, days)
            IV = 100 * s.iv(KK, DD / 365)
            m = ax.pcolormesh(kk, days, IV, cmap=t.heat_cmap(), shading="auto", rasterized=True)
            cs = ax.contour(kk, days, IV, levels=8, colors=t.surface, linewidths=0.6, alpha=0.7)
            ax.clabel(cs, fontsize=7, fmt="%.0f", colors=t.ink)
            ax.set_yscale("log")
            ax.set_yticks([3, 7, 14, 30, 60, 90, 180, 365])
            ax.set_yticklabels(["3d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"])
            ax.set_ylim(days[0], days[-1])
            ax.set_xlabel("log-moneyness  k = ln(K/F)")
            ax.set_ylabel("maturity")
            ax.set_title(f"{rep.currency}: SSVI implied vol (%)", loc="left")
            ax.grid(False)
            cb = fig.colorbar(m, ax=ax, shrink=0.9, pad=0.01)
            cb.outline.set_visible(False)
            cb.ax.tick_params(colors=t.muted, labelsize=8)
        _title(fig, t, "Implied volatility surface", stamp)
        return fig

    return save_both(draw, FIG, "surface_heatmap")


def fig_surface_3d(reps: list[CurrencyReport], stamp: str):  # type: ignore[no-untyped-def]
    def draw(t: Theme) -> plt.Figure:
        fig = plt.figure(figsize=(11, 4.8), layout="constrained")
        for i, rep in enumerate(reps):
            ax = fig.add_subplot(1, len(reps), i + 1, projection="3d")
            s = rep.fits.ssvi
            days = np.linspace(float(s.T_nodes[0]), float(s.T_nodes[-1]), 60) * 365
            kk = np.linspace(-0.6, 0.6, 61)
            KK, DD = np.meshgrid(kk, days)
            IV = 100 * s.iv(KK, DD / 365)
            ax.plot_surface(
                KK,
                DD,
                IV,
                cmap=t.heat_cmap(),
                linewidth=0,
                antialiased=True,
                rcount=60,
                ccount=61,
                alpha=0.95,
            )
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0, 0, 0, 0))
                axis._axinfo["grid"]["color"] = t.grid
                axis.label.set_color(t.ink_secondary)
            ax.tick_params(colors=t.muted, labelsize=7.5)
            ax.set_xlabel("k = ln(K/F)", fontsize=8.5)
            ax.set_ylabel("days", fontsize=8.5)
            ax.set_zlabel("IV (%)", fontsize=8.5)
            ax.set_title(rep.currency, loc="left", color=t.ink)
            ax.view_init(elev=24, azim=-58)
        _title(fig, t, "SSVI surface in 3D", stamp)
        return fig

    return save_both(draw, FIG, "surface_3d")


# --------------------------------------------------------------------------- 2. smiles
def fig_smiles(rep: CurrencyReport, stamp: str):  # type: ignore[no-untyped-def]
    exps = rep.fits.expiries
    n = len(exps)
    ncol = 5 if n > 8 else 4
    nrow = int(np.ceil(n / ncol))

    def draw(t: Theme) -> plt.Figure:
        fig, axes = plt.subplots(
            nrow, ncol, figsize=(13, 2.6 * nrow + 0.8), layout="constrained", squeeze=False
        )
        q = rep.quotes[rep.quotes["keep"]]
        c_mkt, c_svi, c_ssvi = t.ink_secondary, t.currency_color(rep.currency), t.series[2]
        for ax, (_, row) in zip(axes.flat, exps.iterrows(), strict=False):
            g = q[q["expiry"] == row["expiry"]]
            kk = np.linspace(g["k"].min() - 0.05, g["k"].max() + 0.05, 200)
            T = row["T"]
            ax.vlines(
                g["k"], 100 * g["bid_iv"], 100 * g["ask_iv"], color=t.muted, lw=1.0, alpha=0.8
            )
            ax.plot(g["k"], 100 * g["mid_iv"], "o", ms=3.8, color=c_mkt, mec=t.surface, mew=0.5)
            ax.plot(kk, 100 * rep.fits.svi.iv(kk, T), color=c_svi, lw=1.8)
            ax.plot(kk, 100 * rep.fits.ssvi.iv(kk, T), color=c_ssvi, lw=1.6, ls="--")
            ax.set_title(f"{row['expiry']:%d %b %y}  ({T * 365:.0f}d)", loc="left", fontsize=9.5)
            ax.tick_params(labelsize=7.5)
        for ax in axes.flat[n:]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("IV (%)")
        for ax in axes[-1, :]:
            ax.set_xlabel("k = ln(K/F)")
        handles = [
            Line2D([], [], color=t.muted, lw=1.0, label="bid-ask IV"),
            Line2D([], [], marker="o", ls="", color=c_mkt, ms=4, label="mid IV"),
            Line2D([], [], color=c_svi, lw=1.8, label="SVI (per expiry)"),
            Line2D([], [], color=c_ssvi, lw=1.6, ls="--", label="SSVI (surface)"),
        ]
        fig.legend(handles=handles, loc="outside upper right", ncol=4)
        _title(fig, t, f"{rep.currency} smiles by expiry", stamp)
        return fig

    return save_both(draw, FIG, f"smiles_{rep.currency.lower()}")


# --------------------------------------------------------------------------- 3. ATM term
def fig_atm(reps: list[CurrencyReport], stamp: str):  # type: ignore[no-untyped-def]
    def draw(t: Theme) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
        for rep in reps:
            c = t.currency_color(rep.currency)
            a = rep.atm
            ax.plot(a["days"], 100 * a["ssvi"], color=c, lw=2, label=f"{rep.currency} SSVI")
            ax.plot(
                a["days"],
                100 * a["mark"],
                "o",
                color=c,
                ms=6,
                mec=t.surface,
                mew=1.2,
                label=f"{rep.currency} Deribit mark (k=0)",
            )
            ax.annotate(
                rep.currency,
                (a["days"].iloc[-1], 100 * a["ssvi"].iloc[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                color=t.ink_secondary,
                fontsize=9,
                fontweight="bold",
            )
        ax.set_xscale("log")
        ax.set_xticks([3, 7, 14, 30, 60, 90, 180, 365])
        ax.set_xticklabels(["3d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"])
        ax.set_xlabel("maturity")
        ax.set_ylabel("ATM-forward implied vol (%)")
        ax.legend(loc="best", ncol=2)
        _title(fig, t, "ATM term structure", stamp)
        return fig

    return save_both(draw, FIG, "atm_term_structure")


# --------------------------------------------------------------------------- 4. RR / BF
def fig_rr_bf(reps: list[CurrencyReport], stamp: str):  # type: ignore[no-untyped-def]
    def draw(t: Theme) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
        for rep in reps:
            c = t.currency_color(rep.currency)
            d = rep.rr_bf[rep.rr_bf["model"] == "svi"]
            for conv, ls, mk in (("forward", "-", "o"), ("premium_adjusted", ":", "s")):
                e = d[d["convention"] == conv]
                lab = f"{rep.currency}, {'forward' if conv == 'forward' else 'premium-adj.'} delta"
                ax1.plot(e["days"], 100 * e["rr"], ls=ls, marker=mk, ms=4, color=c, label=lab)
                ax2.plot(e["days"], 100 * e["bf"], ls=ls, marker=mk, ms=4, color=c, label=lab)
        for ax, title, yl in (
            (ax1, "25-delta risk reversal", "RR = σ(25Δ call) − σ(25Δ put), vol pts"),
            (ax2, "25-delta butterfly", "BF = ½(σ25C + σ25P) − σATM, vol pts"),
        ):
            ax.axhline(0, color=t.axis, lw=1)
            ax.set_xscale("log")
            ax.set_xticks([3, 7, 14, 30, 60, 90, 180, 365])
            ax.set_xticklabels(["3d", "1w", "2w", "1m", "2m", "3m", "6m", "1y"])
            ax.set_title(title, loc="left")
            ax.set_ylabel(yl)
            ax.set_xlabel("maturity")
        ax1.legend(loc="best")
        _title(fig, t, "Skew and convexity term structures (SVI fits)", stamp)
        return fig

    return save_both(draw, FIG, "rr_bf_term_structure")


# --------------------------------------------------------------------------- 5. errors
def fig_errors(reps: list[CurrencyReport], stamp: str):  # type: ignore[no-untyped-def]
    def draw(t: Theme) -> plt.Figure:
        fig, axes = plt.subplots(
            2, len(reps), figsize=(11, 6.6), layout="constrained", sharey="row", squeeze=False
        )
        for j, rep in enumerate(reps):
            e = rep.errors_svi
            exps = sorted(e["expiry"].unique())
            cols = dict(zip(exps, t.tenor_colors(len(exps)), strict=True))
            ax = axes[0, j]
            for exp in exps:
                g = e[e["expiry"] == exp]
                ax.plot(
                    g["k"],
                    g["err_vs_mark_volpts"],
                    "o",
                    ms=3.5,
                    color=cols[exp],
                    mec=t.surface,
                    mew=0.4,
                    label=f"{exp:%d %b %y}",
                )
            ax.axhline(0, color=t.axis, lw=1)
            ax.set_title(f"{rep.currency}: SVI fit − Deribit mark IV", loc="left")
            ax.set_xlabel("k = ln(K/F)")
            if j == 0:
                ax.set_ylabel("vol points")
            if j == len(reps) - 1:
                ax.legend(
                    fontsize=7,
                    loc="upper left",
                    bbox_to_anchor=(1.01, 1.0),
                    title="expiry",
                    title_fontsize=7.5,
                )
            # per-expiry RMSE vs mid: SVI and SSVI
            ax = axes[1, j]
            cmp_ = rep.comparison
            x = np.arange(len(cmp_))
            c = t.currency_color(rep.currency)
            ax.bar(x - 0.2, cmp_["svi_rmse_volpts"], width=0.38, color=c, label="SVI")
            ax.bar(x + 0.2, cmp_["ssvi_rmse_volpts"], width=0.38, color=t.series[2], label="SSVI")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{d:.0f}d" for d in cmp_["days"]], fontsize=7.5)
            ax.set_title(f"{rep.currency}: RMSE vs mid IV by expiry", loc="left")
            if j == 0:
                ax.set_ylabel("RMSE, vol points")
            ax.legend(loc="upper right")
            ax.grid(axis="x", visible=False)
        _title(fig, t, "Fit quality", stamp)
        return fig

    return save_both(draw, FIG, "fit_errors")


# --------------------------------------------------------------------------- 6. arbitrage
def fig_total_variance(rep: CurrencyReport, stamp: str):  # type: ignore[no-untyped-def]
    def draw(t: Theme) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), layout="constrained", sharey=True)
        exps = rep.fits.expiries
        cols = t.tenor_colors(len(exps))
        n_cross: dict[str, int] = {}
        for ax, model in zip(axes, ("svi", "ssvi"), strict=True):
            s = getattr(rep.fits, model)
            W = np.array([s.total_variance(K_GRID, row["T"]) for _, row in exps.iterrows()])
            for (_, row), c, w in zip(exps.iterrows(), cols, W, strict=True):
                ax.plot(K_GRID, w, color=c, lw=1.6, label=f"{row['T'] * 365:.0f}d")
            # mark every k where a longer expiry has LESS total variance (calendar arbitrage)
            bad = np.diff(W, axis=0) < 0
            n_cross[model] = int(bad.any(axis=1).sum())
            for i in range(bad.shape[0]):
                if bad[i].any():
                    ax.plot(
                        K_GRID[bad[i]], W[i + 1][bad[i]], ".", ms=4, color=t.series[1], zorder=5
                    )
            ax.set_yscale("log")
            ax.set_xlabel("k = ln(K/F)")
        n_pairs = len(exps) - 1
        n_in = int((rep.svi_calendar_gaps_in_range < 0).sum())
        axes[0].set_title(
            f"SVI slices: {n_cross['svi']} of {n_pairs} adjacent pairs cross "
            f"({n_in} within quoted strikes)",
            loc="left",
        )
        axes[1].set_title(
            f"SSVI: {n_cross['ssvi']} of {n_pairs} pairs cross (by construction)", loc="left"
        )
        axes[0].set_ylabel("total implied variance σ²T (log scale)")
        handles, labels = axes[1].get_legend_handles_labels()
        handles.append(Line2D([], [], marker=".", ls="", ms=8, color=t.series[1]))
        labels.append("crossing")
        fig.legend(
            handles,
            labels,
            title="maturity",
            fontsize=7.5,
            title_fontsize=8,
            loc="outside right center",
        )
        _title(
            fig,
            t,
            f"{rep.currency} calendar-arbitrage check: total variance must increase with T",
            stamp,
        )
        return fig

    return save_both(draw, FIG, f"arbitrage_total_variance_{rep.currency.lower()}")


def fig_density(reps: list[CurrencyReport], stamp: str):  # type: ignore[no-untyped-def]
    def draw(t: Theme) -> plt.Figure:
        fig, axes = plt.subplots(
            2, len(reps), figsize=(11, 6.4), layout="constrained", squeeze=False
        )
        for j, rep in enumerate(reps):
            exps = rep.fits.expiries
            cols = t.tenor_colors(len(exps))
            for f, (_, row), c in zip(rep.fits.svi.svi, exps.iterrows(), cols, strict=True):
                lab = f"{row['T'] * 365:.0f}d"
                axes[0, j].plot(K_GRID, f.params.g(K_GRID), color=c, lw=1.4, label=lab)
                axes[1, j].plot(K_GRID, f.params.density(K_GRID), color=c, lw=1.4, label=lab)
            axes[0, j].axhline(0, color=t.muted, lw=1, ls="--")
            ming = rep.svi_params()["min_g"].min()
            axes[0, j].set_title(
                f"{rep.currency}: Durrleman g(k), SVI slices (min {ming:.3f})", loc="left"
            )
            axes[0, j].set_ylim(bottom=min(-0.05, ming * 1.2))
            axes[1, j].set_title(f"{rep.currency}: implied density of ln(S_T/F)", loc="left")
            axes[1, j].set_xlabel("k = ln(K/F)")
            axes[1, j].set_xlim(-1.0, 1.0)
        axes[0, 0].set_ylabel("g(k)  (≥ 0 ⇔ no butterfly arbitrage)")
        axes[1, 0].set_ylabel("density")
        axes[1, -1].legend(title="maturity", fontsize=7, title_fontsize=7.5, ncol=2)
        _title(fig, t, "Butterfly-arbitrage check: g(k) ≥ 0 and a positive density", stamp)
        return fig

    return save_both(draw, FIG, "arbitrage_density")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    global FIG, DERIVED
    if not args.synthetic:
        FIG = ROOT / "data" / "local" / "figures"
        DERIVED = ROOT / "data" / "derived" / "live"
    if args.synthetic:
        path = latest_snapshot_dir(ROOT / "tests" / "fixtures" / "synthetic")
    else:
        path = args.snapshot or latest_snapshot_dir(ROOT / "data" / "snapshots")
    if path is None:
        raise SystemExit("no snapshot found: run scripts/refresh_snapshots.py or use --synthetic")
    snap = load_snapshot(path)
    reps = [analyse(snap, c) for c in snap.currencies()]
    stamp = SYNTHETIC_STAMP if args.synthetic else _stamp(reps)

    fig_surface(reps, stamp)
    fig_surface_3d(reps, stamp)
    for rep in reps:
        fig_smiles(rep, stamp)
        fig_total_variance(rep, stamp)
    fig_atm(reps, stamp)
    fig_rr_bf(reps, stamp)
    fig_errors(reps, stamp)
    fig_density(reps, stamp)

    summary: dict[str, object] = {"snapshot": str(reps[0].snapshot_time), "source": stamp}
    for rep in reps:
        write_derived(rep, DERIVED)
        cmp_ = rep.comparison
        fwd = rep.rr_bf[(rep.rr_bf["model"] == "svi")]
        summary[rep.currency] = {
            "n_quotes_used": int(rep.quotes["keep"].sum()),
            "n_quotes_otm": len(rep.quotes),
            "n_expiries": len(cmp_),
            "svi_rmse_volpts_median": float(cmp_["svi_rmse_volpts"].median()),
            "ssvi_rmse_volpts_median": float(cmp_["ssvi_rmse_volpts"].median()),
            "svi_inside_spread": float(cmp_["svi_inside_share"].mean()),
            "ssvi_inside_spread": float(cmp_["ssvi_inside_share"].mean()),
            "svi_err_vs_mark_median_abs_volpts": float(
                rep.errors_svi["err_vs_mark_volpts"].abs().median()
            ),
            "mid_minus_mark_median_abs_volpts": float(
                rep.errors_svi["mid_minus_mark_volpts"].abs().median()
            ),
            "svi_all_butterfly_free": bool(cmp_["svi_butterfly_ok"].all()),
            "svi_calendar_crossings_wide": int((rep.svi_calendar_gaps < 0).sum()),
            "svi_calendar_crossings_in_range": int((rep.svi_calendar_gaps_in_range < 0).sum()),
            "ssvi": rep.ssvi_params().iloc[0].to_dict(),
            "rr25_fwd_1m_volpts": float(
                np.interp(
                    30,
                    fwd[fwd.convention == "forward"]["days"],
                    100 * fwd[fwd.convention == "forward"]["rr"],
                )
            ),
            "rr25_pa_1m_volpts": float(
                np.interp(
                    30,
                    fwd[fwd.convention == "premium_adjusted"]["days"],
                    100 * fwd[fwd.convention == "premium_adjusted"]["rr"],
                )
            ),
        }
    DERIVED.mkdir(parents=True, exist_ok=True)
    (DERIVED / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

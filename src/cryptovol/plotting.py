"""Matplotlib figures that read well in both GitHub light and dark themes.

Each figure is drawn twice, once per theme, with that theme's own steps of the
same palette. The README switches between them with ``<picture>`` and
``prefers-color-scheme``. Colour roles:

* categorical: BTC = blue, ETH = orange (fixed per entity, never by rank);
  a third slot (aqua) is used only for a third series;
* sequential (tenor, IV level): a single blue ramp, light to dark;
* text and axes use neutral ink, never series colours.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

BLUE_RAMP = {
    100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec",
    350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab",
    600: "#184f95", 650: "#104281", 700: "#0d366b",
}  # fmt: skip


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]  # blue, orange, aqua
    tenor_steps: tuple[int, int]  # ordinal ramp range (short tenor -> long tenor)
    heat_steps: tuple[int, ...]  # continuous ramp, low -> high

    def currency_color(self, ccy: str) -> str:
        return self.series[0] if ccy == "BTC" else self.series[1]

    def tenor_colors(self, n: int) -> list[str]:
        """``n`` ordinal blues from short (near the surface) to long tenor."""
        steps = sorted(BLUE_RAMP)
        lo, hi = self.tenor_steps
        pool = [s for s in steps if min(lo, hi) <= s <= max(lo, hi)]
        if lo > hi:
            pool = pool[::-1]
        idx = np.linspace(0, len(pool) - 1, max(n, 1))
        return [BLUE_RAMP[pool[int(np.rint(i))]] for i in idx]

    def heat_cmap(self) -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            f"heat_{self.name}", [BLUE_RAMP[s] for s in self.heat_steps]
        )


LIGHT = Theme(
    name="light", surface="#ffffff", ink="#1f2328", ink_secondary="#52514e", muted="#6e6d68",
    grid="#e1e0d9", axis="#c3c2b7", series=("#2a78d6", "#eb6834", "#1baf7a"),
    tenor_steps=(250, 700), heat_steps=(100, 200, 300, 400, 500, 600, 700),
)  # fmt: skip
DARK = Theme(
    name="dark", surface="#0d1117", ink="#e6edf3", ink_secondary="#c3c2b7", muted="#9a9890",
    grid="#262a30", axis="#3d424a", series=("#3987e5", "#d95926", "#199e70"),
    tenor_steps=(600, 150), heat_steps=(700, 600, 500, 400, 300, 200, 100),
)  # fmt: skip
THEMES = (LIGHT, DARK)


def rc_params(t: Theme) -> dict[str, Any]:
    return {
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "axes.edgecolor": t.axis,
        "axes.labelcolor": t.ink_secondary,
        "axes.titlecolor": t.ink,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": t.grid,
        "grid.linewidth": 0.6,
        "xtick.color": t.muted,
        "ytick.color": t.muted,
        "xtick.labelcolor": t.muted,
        "ytick.labelcolor": t.muted,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": t.ink,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": t.ink_secondary,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    }


@contextmanager
def themed(t: Theme) -> Iterator[None]:
    with mpl.rc_context(rc_params(t)):  # type: ignore[arg-type]  # rc keys are Literal-typed
        yield


def save_both(draw: Callable[[Theme], Figure], out_dir: str | Path, name: str) -> list[Path]:
    """Draw ``draw(theme)`` once per theme, save ``<name>_light.png`` and ``<name>_dark.png``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in THEMES:
        with themed(t):
            fig = draw(t)
            p = out / f"{name}_{t.name}.png"
            fig.savefig(p, metadata={"Software": None})
            plt.close(fig)
            paths.append(p)
    return paths


def suptitle(fig: Figure, t: Theme, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title with an optional muted subtitle line."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=t.ink)
    if subtitle:
        fig.text(
            0.01,
            0.935,
            subtitle,
            ha="left",
            va="top",
            fontsize=9,
            color=t.muted,
            transform=fig.transFigure,
        )

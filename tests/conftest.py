"""Shared fixtures. Tests read only the committed synthetic snapshot, never the network."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cryptovol.cleaning import clean_options
from cryptovol.snapshots import Snapshot, latest_snapshot_dir, load_snapshot
from cryptovol.surface import SurfaceFits, fit_surfaces

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def snap() -> Snapshot:
    path = latest_snapshot_dir(FIXTURE_ROOT)
    assert path is not None, "synthetic fixture missing: run scripts/make_synthetic_fixture.py"
    return load_snapshot(path)


@pytest.fixture(scope="session")
def quotes_btc(snap: Snapshot) -> pd.DataFrame:
    return clean_options(snap, "BTC")


@pytest.fixture(scope="session")
def fits_btc(quotes_btc: pd.DataFrame) -> SurfaceFits:
    return fit_surfaces(quotes_btc)


@pytest.fixture(scope="session")
def fits_eth(snap: Snapshot) -> SurfaceFits:
    return fit_surfaces(clean_options(snap, "ETH"))

"""Download a fresh Deribit snapshot (BTC and ETH options and futures).

Usage::

    python scripts/refresh_snapshots.py [--root data/snapshots] [--currencies BTC ETH]

This is the only part of the project that touches the network. Snapshots go
to ``data/snapshots/`` (git-ignored). Deribit's terms limit market data and
derived data to personal use, so raw snapshots are never committed; see DATA.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cryptovol.deribit import DeribitClient
from cryptovol.snapshots import fetch_snapshot, save_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(ROOT / "data" / "snapshots"))
    parser.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    args = parser.parse_args()
    snap = fetch_snapshot(DeribitClient(), tuple(args.currencies))
    out = save_snapshot(snap, args.root)
    n_opt = snap.options.groupby("currency").size().to_dict()
    print(f"saved {out}  options={n_opt}  index={snap.index_prices}")


if __name__ == "__main__":
    main()

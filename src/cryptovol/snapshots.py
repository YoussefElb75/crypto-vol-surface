"""Dated market snapshots: build from the API, save to and load from Parquet.

A snapshot is a directory ``<root>/<YYYY-MM-DDTHHMMSSZ>/`` that holds:

* ``options.parquet``: one row per option (instrument metadata and book summary)
* ``futures.parquet``: one row per future, including the perpetual
* ``meta.json``: snapshot timestamp (ms, UTC), index prices, source

Raw Deribit snapshots are personal-use data under Deribit's terms (see
``DATA.md``). They are written to ``data/snapshots/``, which is git-ignored.
The committed test fixture in ``tests/fixtures`` is synthetic and uses the
same schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cryptovol.deribit import DeribitClient

OPTION_COLUMNS = [
    "instrument_name",
    "currency",
    "expiration_timestamp",
    "strike",
    "option_type",
    "bid_price",
    "ask_price",
    "mid_price",
    "mark_price",
    "mark_iv",
    "underlying_price",
    "underlying_index",
    "open_interest",
    "volume",
    "creation_timestamp",
]
FUTURE_COLUMNS = [
    "instrument_name",
    "currency",
    "expiration_timestamp",
    "bid_price",
    "ask_price",
    "mark_price",
    "open_interest",
    "volume",
    "creation_timestamp",
]
PERPETUAL_EXPIRY = 32503680000000  # Deribit's expiry stamp for perpetuals (year 3000)


@dataclass
class Snapshot:
    """Option and future quotes for one or more currencies at one time."""

    timestamp_ms: int
    options: pd.DataFrame
    futures: pd.DataFrame
    index_prices: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def time(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ms / 1000, tz=timezone.utc)

    @property
    def stamp(self) -> str:
        return self.time.strftime("%Y-%m-%dT%H%M%SZ")

    def currencies(self) -> list[str]:
        return sorted(self.options["currency"].unique().tolist())

    def for_currency(self, currency: str) -> Snapshot:
        return Snapshot(
            timestamp_ms=self.timestamp_ms,
            options=self.options[self.options["currency"] == currency].reset_index(drop=True),
            futures=self.futures[self.futures["currency"] == currency].reset_index(drop=True),
            index_prices={currency: self.index_prices[currency]},
            meta=dict(self.meta),
        )


def fetch_snapshot(client: DeribitClient, currencies: tuple[str, ...] = ("BTC", "ETH")) -> Snapshot:
    """Download a snapshot of options, futures and index prices."""
    opt_frames, fut_frames, index = [], [], {}
    t0 = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    for ccy in currencies:
        inst = client.get_instruments(ccy, "option")
        book = client.get_book_summary(ccy, "option")
        inst = inst[["instrument_name", "expiration_timestamp", "strike", "option_type"]]
        opt = book.merge(inst, on="instrument_name", how="inner")
        opt["currency"] = ccy
        opt_frames.append(opt.reindex(columns=OPTION_COLUMNS))

        finst = client.get_instruments(ccy, "future")
        fbook = client.get_book_summary(ccy, "future")
        fut = fbook.merge(
            finst[["instrument_name", "expiration_timestamp"]], on="instrument_name", how="inner"
        )
        fut["currency"] = ccy
        fut_frames.append(fut.reindex(columns=FUTURE_COLUMNS))
        index[ccy] = client.get_index_price(ccy)
    return Snapshot(
        timestamp_ms=t0,
        options=pd.concat(opt_frames, ignore_index=True),
        futures=pd.concat(fut_frames, ignore_index=True),
        index_prices=index,
        meta={"source": "deribit public API v2", "base_url": client.base_url},
    )


def save_snapshot(snap: Snapshot, root: str | Path) -> Path:
    """Write ``snap`` to ``root/<stamp>/`` and return that directory."""
    out = Path(root) / snap.stamp
    out.mkdir(parents=True, exist_ok=True)
    snap.options.to_parquet(out / "options.parquet", index=False)
    snap.futures.to_parquet(out / "futures.parquet", index=False)
    meta = {"timestamp_ms": snap.timestamp_ms, "index_prices": snap.index_prices, **snap.meta}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def load_snapshot(path: str | Path) -> Snapshot:
    """Load a snapshot directory written by :func:`save_snapshot`."""
    p = Path(path)
    meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
    ts = int(meta.pop("timestamp_ms"))
    index = {str(k): float(v) for k, v in meta.pop("index_prices").items()}
    return Snapshot(
        timestamp_ms=ts,
        options=pd.read_parquet(p / "options.parquet"),
        futures=pd.read_parquet(p / "futures.parquet"),
        index_prices=index,
        meta=meta,
    )


def latest_snapshot_dir(root: str | Path) -> Path | None:
    """Most recent snapshot directory under ``root``, or ``None``."""
    dirs = sorted(d for d in Path(root).glob("*") if (d / "meta.json").exists())
    return dirs[-1] if dirs else None

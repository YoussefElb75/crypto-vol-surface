from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from cryptovol.deribit import DeribitClient, DeribitError
from cryptovol.snapshots import (
    FUTURE_COLUMNS,
    OPTION_COLUMNS,
    Snapshot,
    load_snapshot,
    save_snapshot,
)


def test_fixture_schema(snap: Snapshot) -> None:
    assert list(snap.options.columns) == OPTION_COLUMNS
    assert list(snap.futures.columns) == FUTURE_COLUMNS
    assert snap.currencies() == ["BTC", "ETH"]
    assert snap.stamp == "2026-01-02T080000Z"


def test_save_load_round_trip(snap: Snapshot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    back = load_snapshot(save_snapshot(snap, tmp_path))
    assert back.timestamp_ms == snap.timestamp_ms
    assert back.index_prices == snap.index_prices
    pd.testing.assert_frame_equal(back.options, snap.options)
    pd.testing.assert_frame_equal(back.futures, snap.futures)


def test_for_currency(snap: Snapshot) -> None:
    eth = snap.for_currency("ETH")
    assert set(eth.options["currency"]) == {"ETH"}
    assert set(eth.index_prices) == {"ETH"}


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeSession:
    """Stands in for requests.Session so that no test touches the network."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _FakeResponse:
        self.calls.append((url, params))
        return _FakeResponse(self.payload)


def _client(payload: dict[str, Any]) -> tuple[DeribitClient, _FakeSession]:
    sess = _FakeSession(payload)
    return DeribitClient(session=sess, min_interval=0), sess  # type: ignore[arg-type]


def test_client_only_allows_public_methods() -> None:
    client, sess = _client({"result": []})
    with pytest.raises(ValueError):
        client._get("private/get_account_summary")
    assert sess.calls == []


def test_client_parses_results_and_errors() -> None:
    client, sess = _client({"result": {"index_price": 123.5}})
    assert client.get_index_price("BTC") == 123.5
    assert sess.calls[0][0].endswith("public/get_index_price")
    assert sess.calls[0][1] == {"index_name": "btc_usd"}

    bad, _ = _client({"error": {"code": 1}})
    with pytest.raises(DeribitError):
        bad.get_index_price("BTC")


def test_client_serialises_booleans() -> None:
    client, sess = _client({"result": [{"instrument_name": "X"}]})
    client.get_instruments("BTC")
    assert sess.calls[0][1]["expired"] == "false"


class _HTMLResponse:
    status_code = 502

    def json(self) -> dict[str, Any]:
        raise ValueError("not JSON")

    def raise_for_status(self) -> None:
        raise RuntimeError("HTTP 502")


def test_client_non_json_response_raises_http_error() -> None:
    sess = _FakeSession({})
    sess.get = lambda url, params, timeout: _HTMLResponse()  # type: ignore[assignment,method-assign,misc,return-value]
    client = DeribitClient(session=sess, min_interval=0)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="502"):
        client.get_index_price("BTC")

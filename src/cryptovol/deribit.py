"""Minimal read-only client for Deribit's public JSON-RPC API (v2).

Only unauthenticated ``public/*`` methods are used: no API keys, no private
endpoints, no trading. See ``DATA.md`` for Deribit's terms of use.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://www.deribit.com/api/v2"
USER_AGENT = "crypto-vol-surface/0.1 (+https://github.com/YoussefElb75/crypto-vol-surface)"


class DeribitError(RuntimeError):
    """Raised when the API returns an error payload."""


class DeribitClient:
    """Tiny HTTP client for Deribit's public market-data methods.

    Parameters
    ----------
    base_url:
        API root (use ``https://test.deribit.com/api/v2`` for the testnet).
    min_interval:
        Minimum delay in seconds between two requests. Deribit applies
        credit-based rate limits; a snapshot only needs a handful of calls,
        so a conservative pause keeps us far below them.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = 15.0,
        min_interval: float = 0.25,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_call = 0.0

    def _get(self, method: str, **params: Any) -> Any:
        if not method.startswith("public/"):
            raise ValueError("only public/* methods are allowed")
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        clean = {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in params.items()}
        resp = self.session.get(f"{self.base_url}/{method}", params=clean, timeout=self.timeout)
        self._last_call = time.monotonic()
        try:
            payload = resp.json()
        except ValueError:  # non-JSON body (e.g. an HTML 5xx page)
            resp.raise_for_status()
            raise DeribitError(f"{method}: non-JSON response") from None
        if "error" in payload:  # JSON-RPC errors come with a 4xx status and a JSON body
            raise DeribitError(f"{method}: {payload['error']}")
        resp.raise_for_status()
        return payload["result"]

    # --- raw methods -------------------------------------------------------
    def get_instruments(self, currency: str, kind: str = "option") -> pd.DataFrame:
        """Active instruments (``public/get_instruments``)."""
        rows = self._get("public/get_instruments", currency=currency, kind=kind, expired=False)
        return pd.DataFrame(rows)

    def get_book_summary(self, currency: str, kind: str = "option") -> pd.DataFrame:
        """Best bid/ask, mark price, mark IV, underlying per instrument
        (``public/get_book_summary_by_currency``)."""
        rows = self._get("public/get_book_summary_by_currency", currency=currency, kind=kind)
        return pd.DataFrame(rows)

    def get_index_price(self, currency: str) -> float:
        """Deribit index (spot reference) for ``currency`` in USD."""
        res = self._get("public/get_index_price", index_name=f"{currency.lower()}_usd")
        return float(res["index_price"])

    def get_order_book(self, instrument_name: str, depth: int = 1) -> dict[str, Any]:
        """Order book of one instrument, including ``bid_iv``/``ask_iv``/``mark_iv``."""
        result: dict[str, Any] = self._get(
            "public/get_order_book", instrument_name=instrument_name, depth=depth
        )
        return result

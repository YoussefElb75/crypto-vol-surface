"""Regenerate the committed synthetic test fixture (no network).

Usage::

    python scripts/make_synthetic_fixture.py

Writes ``tests/fixtures/synthetic/<stamp>/``, which has the same schema as a
real Deribit snapshot but quotes generated from a known SSVI surface.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from cryptovol.snapshots import save_snapshot
from cryptovol.synthetic import make_synthetic_snapshot

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "synthetic"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    path = save_snapshot(make_synthetic_snapshot(), OUT)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

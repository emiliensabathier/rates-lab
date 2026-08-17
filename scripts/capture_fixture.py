"""Freeze the FRED payloads the test suite replays.

Run this deliberately, never automatically. Re-capturing moves every published figure, and
the point of the fixture is that moving them is a decision somebody made and recorded in a
commit, not something that happened because a test ran on a Tuesday.

    python scripts/capture_fixture.py            # from the on-disk cache
    python scripts/capture_fixture.py --refresh  # pull fresh, then freeze

Payloads are trimmed to the first year the nominal panel can start. The one-month bill
series begins 2001-07, so nothing before 2001 can ever enter a complete cross-section, and
carrying forty extra years of DGS10 would quadruple the fixture for no coverage.
"""

from __future__ import annotations

import argparse
import io
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from rlab.data import fred
from rlab.pipeline import BENCHMARK_SERIES, NOMINAL_PILLARS, REAL_PILLARS

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fred"
FIRST_KEPT = "2001-01-01"


def trim(text: str, first_kept: str) -> str:
    """Drop observations before `first_kept`, preserving the FRED export format exactly."""
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    dates = pd.to_datetime(frame["observation_date"])
    kept = frame[dates >= pd.Timestamp(first_kept)]
    buffer = io.StringIO()
    kept.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    series = [*NOMINAL_PILLARS, *REAL_PILLARS, BENCHMARK_SERIES]
    cache = Path(args.cache_dir)
    fred.load(series, cache_dir=cache, refresh=args.refresh)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for series_id in series:
        payload = trim((cache / f"{series_id}.csv").read_text(encoding="utf-8"), FIRST_KEPT)
        (FIXTURE_DIR / f"{series_id}.csv").write_text(payload, encoding="utf-8")
        total += len(payload)
        print(f"  {series_id:14s} {len(payload):>8,} bytes")

    stamp = datetime.now(UTC).date().isoformat()
    (FIXTURE_DIR / "CAPTURED").write_text(stamp + "\n", encoding="utf-8")
    print(f"froze {len(series)} series, {total:,} bytes, captured {stamp}")
    print("remember to update CAPTURED in tests/fixtures/frozen.py if it changed")


if __name__ == "__main__":  # pragma: no cover
    main()

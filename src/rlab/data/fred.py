"""FRED series loading over the keyless CSV endpoint, with an on-disk cache.

The fetcher is injected so the whole module is testable without network access.
"""

from __future__ import annotations

import io
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from rlab.errors import DataError

BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

# FRED writes a missing observation as an empty field. Older exports used a dot. Both are
# accepted because guessing wrong turns a bank holiday into a zero yield, which then
# propagates silently through the whole bootstrap.
MISSING_MARKERS = ("", ".")


class Fetcher(Protocol):
    """Returns the raw CSV text for one FRED series id."""

    def __call__(self, series_id: str) -> str: ...


def http_fetcher(series_id: str) -> str:
    with urllib.request.urlopen(BASE_URL.format(series_id=series_id), timeout=60) as handle:
        return handle.read().decode("utf-8")


def parse_csv(text: str) -> pd.Series:
    """Parse one FRED CSV export into a float series indexed by observation date."""
    frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    if len(frame.columns) != 2 or frame.columns[0] != "observation_date":
        raise DataError(f"unexpected FRED header {list(frame.columns)}, expected observation_date")
    series_id = frame.columns[1]
    if frame.empty:
        raise DataError(f"FRED export for {series_id} has no observation rows")
    kept = frame[~frame[series_id].str.strip().isin(MISSING_MARKERS)]
    try:
        raw_values = kept[series_id].astype(float).to_numpy()
    except ValueError as exc:
        raise DataError(f"FRED export for {series_id} has a non-numeric value") from exc
    values = pd.Series(
        raw_values,
        index=pd.to_datetime(kept["observation_date"]),
        name=series_id,
    )
    return values


def load(
    series_ids: list[str],
    cache_dir: Path,
    fetcher: Fetcher = http_fetcher,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load several FRED series into one frame, caching each series to disk.

    The fetch timestamp is stored beside the payload so a stale cache can be reported
    rather than trusted indefinitely.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    columns = []
    for series_id in series_ids:
        payload = cache_dir / f"{series_id}.csv"
        stamp = cache_dir / f"{series_id}.json"
        if refresh or not payload.exists():
            text = fetcher(series_id)
            payload.write_text(text, encoding="utf-8")
            stamp.write_text(
                json.dumps({"fetched_at": datetime.now(UTC).isoformat()}), encoding="utf-8"
            )
        columns.append(parse_csv(payload.read_text(encoding="utf-8")))
    return pd.concat(columns, axis=1)

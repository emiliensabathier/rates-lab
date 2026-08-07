"""Euro area AAA yield curve from the ECB Data Portal, over the keyless CSV endpoint.

The ECB publishes both its zero-coupon spot rates and the Svensson parameters it fitted to
produce them. Loading both is what makes the external validation in rlab.curve possible.
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

BASE_URL = (
    "https://data-api.ecb.europa.eu/service/data/YC/B.U2.EUR.4F.G_N_A.SV_C_YM.{key}"
    "?format=csvdata"
)

SVENSSON_KEYS: tuple[str, ...] = ("BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2")

# SR_1M is not published: the endpoint returns HTTP 404 for it. The curve therefore starts
# at three months, and nothing downstream may assume a one-month euro pillar.
SPOT_KEYS: dict[str, float] = {
    "SR_3M": 0.25,
    "SR_6M": 0.5,
    "SR_1Y": 1.0,
    "SR_2Y": 2.0,
    "SR_3Y": 3.0,
    "SR_5Y": 5.0,
    "SR_7Y": 7.0,
    "SR_10Y": 10.0,
    "SR_20Y": 20.0,
    "SR_30Y": 30.0,
}


class Fetcher(Protocol):
    """Returns the raw CSV text for one ECB yield curve key."""

    def __call__(self, key: str) -> str: ...


def http_fetcher(key: str) -> str:
    with urllib.request.urlopen(BASE_URL.format(key=key), timeout=60) as handle:
        return handle.read().decode("utf-8")


def parse_csv(text: str) -> pd.Series:
    """Reduce one ECB csvdata export to a float series indexed by observation date."""
    frame = pd.read_csv(io.StringIO(text), dtype=str)
    for column in ("TIME_PERIOD", "OBS_VALUE", "DATA_TYPE_FM"):
        if column not in frame.columns:
            raise DataError(f"ECB export is missing column {column}")
    key = frame["DATA_TYPE_FM"].iloc[0]
    return pd.Series(
        frame["OBS_VALUE"].astype(float).to_numpy(),
        index=pd.to_datetime(frame["TIME_PERIOD"]),
        name=key,
    )


def load(
    keys: list[str],
    cache_dir: Path,
    fetcher: Fetcher = http_fetcher,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load several ECB curve keys into one frame, caching each key to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    columns = []
    for key in keys:
        payload = cache_dir / f"ecb_{key}.csv"
        stamp = cache_dir / f"ecb_{key}.json"
        if refresh or not payload.exists():
            text = fetcher(key)
            payload.write_text(text, encoding="utf-8")
            stamp.write_text(
                json.dumps({"fetched_at": datetime.now(UTC).isoformat()}), encoding="utf-8"
            )
        columns.append(parse_csv(payload.read_text(encoding="utf-8")))
    return pd.concat(columns, axis=1)

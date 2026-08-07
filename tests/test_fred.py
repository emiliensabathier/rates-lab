from pathlib import Path

import pandas as pd
import pytest

from rlab.data import fred
from rlab.errors import DataError

HOLIDAY_CSV = """observation_date,DGS10
2025-12-24,4.15
2025-12-25,
2025-12-26,4.14
"""

DOT_CSV = """observation_date,DGS10
2025-12-24,4.15
2025-12-25,.
2025-12-26,4.14
"""


def test_empty_field_is_a_missing_observation_not_a_zero():
    series = fred.parse_csv(HOLIDAY_CSV)
    assert list(series.index) == [pd.Timestamp("2025-12-24"), pd.Timestamp("2025-12-26")]
    assert series.iloc[0] == pytest.approx(4.15)


def test_legacy_dot_marker_is_also_a_missing_observation():
    assert len(fred.parse_csv(DOT_CSV)) == 2


def test_unexpected_header_raises_rather_than_guessing():
    with pytest.raises(DataError, match="header"):
        fred.parse_csv("date;value\n2025-12-24;4.15\n")


def test_empty_payload_raises():
    with pytest.raises(DataError, match="DGS10.*no observation rows"):
        fred.parse_csv("observation_date,DGS10\n")


def test_non_numeric_value_raises():
    with pytest.raises(DataError, match="DGS10"):
        fred.parse_csv("observation_date,DGS10\n2025-12-24,abc\n")


def test_all_missing_markers_return_an_empty_series_not_an_error():
    series = fred.parse_csv("observation_date,DGS10\n2025-12-24,\n2025-12-25,.\n")
    assert len(series) == 0


def test_load_joins_series_on_a_union_index(tmp_path: Path):
    payloads = {
        "DGS2": "observation_date,DGS2\n2026-08-04,4.20\n2026-08-05,4.18\n",
        "DGS10": "observation_date,DGS10\n2026-08-05,4.63\n",
    }
    frame = fred.load(["DGS2", "DGS10"], tmp_path, fetcher=lambda sid: payloads[sid])
    assert list(frame.columns) == ["DGS2", "DGS10"]
    assert frame.loc["2026-08-05", "DGS10"] == pytest.approx(4.63)
    assert pd.isna(frame.loc["2026-08-04", "DGS10"])


def test_second_load_uses_the_cache_and_does_not_refetch(tmp_path: Path):
    calls = []

    def counting_fetcher(series_id: str) -> str:
        calls.append(series_id)
        return "observation_date,DGS10\n2026-08-05,4.63\n"

    fred.load(["DGS10"], tmp_path, fetcher=counting_fetcher)
    fred.load(["DGS10"], tmp_path, fetcher=counting_fetcher)
    assert calls == ["DGS10"]


def test_refresh_forces_a_refetch(tmp_path: Path):
    calls = []

    def counting_fetcher(series_id: str) -> str:
        calls.append(series_id)
        return "observation_date,DGS10\n2026-08-05,4.63\n"

    fred.load(["DGS10"], tmp_path, fetcher=counting_fetcher)
    fred.load(["DGS10"], tmp_path, fetcher=counting_fetcher, refresh=True)
    assert calls == ["DGS10", "DGS10"]

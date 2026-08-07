from pathlib import Path

import pandas as pd
import pytest

from rlab.data import ecb
from rlab.errors import DataError

SAMPLE = (
    "KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,"
    "TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
    "YC.B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y,B,U2,EUR,4F,G_N_A,SV_C_YM,SR_10Y,"
    "2026-08-06,3.146678757,A\n"
)


def test_parse_keeps_only_time_period_and_value():
    series = pd.Series(ecb.parse_csv(SAMPLE))
    assert series.index[0] == pd.Timestamp("2026-08-06")
    assert series.iloc[0] == pytest.approx(3.146678757)
    assert series.name == "SR_10Y"


def test_missing_columns_raise():
    with pytest.raises(DataError, match="OBS_VALUE"):
        ecb.parse_csv("KEY,TIME_PERIOD\nx,2026-08-06\n")


def test_spot_keys_cover_three_months_to_thirty_years_and_exclude_one_month():
    assert ecb.SPOT_KEYS["SR_3M"] == pytest.approx(0.25)
    assert ecb.SPOT_KEYS["SR_30Y"] == pytest.approx(30.0)
    assert "SR_1M" not in ecb.SPOT_KEYS


def test_empty_payload_raises():
    header = (
        "KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,"
        "TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
    )
    with pytest.raises(DataError, match="no observation rows"):
        ecb.parse_csv(header)


def test_non_numeric_value_raises():
    bad = SAMPLE.replace("3.146678757", "not-a-number")
    with pytest.raises(DataError, match="SR_10Y"):
        ecb.parse_csv(bad)


def test_load_joins_keys_on_a_union_index(tmp_path: Path):
    payloads = {
        "SR_10Y": SAMPLE,
        "BETA0": SAMPLE.replace("SR_10Y", "BETA0").replace("3.146678757", "1.2866240787"),
    }
    frame = ecb.load(["SR_10Y", "BETA0"], tmp_path, fetcher=lambda key: payloads[key])
    assert frame.loc["2026-08-06", "BETA0"] == pytest.approx(1.2866240787)


@pytest.mark.network
def test_live_endpoint_still_serves_the_published_parameters(tmp_path: Path):
    frame = ecb.load(list(ecb.SVENSSON_KEYS), tmp_path)
    assert set(frame.columns) == set(ecb.SVENSSON_KEYS)
    assert frame.dropna().shape[0] > 0

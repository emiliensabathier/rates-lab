import io
import zipfile

import pandas as pd
import pytest

from rlab.data import spf
from rlab.errors import DataError

SHEET = """<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="s"><v>0</v></c><c r="B1" t="s"><v>1</v></c>
<c r="C1" t="s"><v>2</v></c></row>
{rows}
</sheetData></worksheet>"""

STRINGS = """<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<si><t>YEAR</t></si><si><t>QUARTER</t></si><si><t>{column}</t></si><si><t>#N/A</t></si></sst>"""


def _workbook(rows: str, column: str = "BILL10") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", SHEET.format(rows=rows))
        archive.writestr("xl/sharedStrings.xml", STRINGS.format(column=column))
    return buffer.getvalue()


def _row(n: int, year: str, quarter: str, value: str) -> str:
    return (f'<row r="{n}"><c r="A{n}"><v>{year}</v></c><c r="B{n}"><v>{quarter}</v></c>'
            f"<c r=\"C{n}\"{value}</c></row>")


def test_parses_survey_quarters_into_decimals():
    rows = _row(2, "1992.0", "1.0", "><v>5.25</v>") + _row(3, "1993.0", "1.0", "><v>4.5</v>")
    series = spf.parse_xlsx(_workbook(rows))
    assert list(series.index) == [pd.Period("1992Q1"), pd.Period("1993Q1")]
    assert series.iloc[0] == pytest.approx(0.0525)


def test_not_available_cells_are_skipped_not_zeroed():
    rows = _row(2, "1991.0", "4.0", ' t="s"><v>3</v>') + _row(3, "1992.0", "1.0", "><v>5.25</v>")
    assert list(spf.parse_xlsx(_workbook(rows)).index) == [pd.Period("1992Q1")]


def test_a_workbook_for_another_variable_is_refused():
    with pytest.raises(DataError, match="BILL10"):
        spf.parse_xlsx(_workbook(_row(2, "1992.0", "1.0", "><v>5.25</v>"), column="TBILL"))


def test_a_workbook_without_observations_is_refused():
    with pytest.raises(DataError, match="no observation"):
        spf.parse_xlsx(_workbook(""))


def test_a_payload_that_is_not_a_workbook_is_refused():
    with pytest.raises(DataError, match="workbook"):
        spf.parse_xlsx(b"<html>moved</html>")


def test_the_published_file_parses():
    from tests.fixtures import frozen

    series = spf.parse_xlsx(frozen.survey_fetcher())
    assert series.index[0] == pd.Period("1992Q1")
    assert series.loc[pd.Period("2026Q1")] == pytest.approx(0.030)


def test_load_caches_the_payload(tmp_path):
    calls = []
    payload = _workbook(_row(2, "1992.0", "1.0", "><v>5.25</v>"))

    def fetcher() -> bytes:
        calls.append(1)
        return payload

    first = spf.load(tmp_path, fetcher=fetcher)
    second = spf.load(tmp_path, fetcher=fetcher)
    assert calls == [1]
    assert first.equals(second)
    spf.load(tmp_path, fetcher=fetcher, refresh=True)
    assert calls == [1, 1]

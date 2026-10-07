"""The Survey of Professional Forecasters' ten-year bill expectation, from the Philadelphia Fed.

``BILL10`` is the median forecast of the three-month Treasury bill rate averaged over the
next ten years, asked in each first-quarter survey since 1992. It is the survey counterpart
of the expectations component a term structure model computes, and the anchor Kim-Wright
uses where ACM has none, which is what makes it the natural test of the level gap.

The file is published as a one-sheet workbook. It is read with the standard library
(``zipfile`` and ``xml``) rather than a spreadsheet package: three columns do not justify
a dependency.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

import pandas as pd

from rlab.errors import DataError

URL = (
    "https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
    "survey-of-professional-forecasters/data-files/files/median_bill10_level.xlsx"
)
VARIABLE = "BILL10"
PAYLOAD = "median_bill10_level.xlsx"
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


class Fetcher(Protocol):
    """Returns the raw workbook bytes."""

    def __call__(self) -> bytes: ...


def http_fetcher() -> bytes:
    with urllib.request.urlopen(URL, timeout=60) as handle:
        return handle.read()


def _cells(row) -> dict[str, str | None]:
    """Column letter to numeric text; shared-string cells (headers, ``#N/A``) map to None."""
    return {
        cell.get("r").rstrip("0123456789"): (
            None if cell.get("t") == "s" else cell.findtext("m:v", namespaces=_NS)
        )
        for cell in row.findall("m:c", _NS)
    }


def parse_xlsx(payload: bytes) -> pd.Series:
    """Survey quarter to median ``BILL10``, as a decimal. Missing quarters are dropped."""
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except (zipfile.BadZipFile, KeyError) as exc:
        raise DataError("the SPF payload is not the expected one-sheet workbook") from exc
    labels = [node.text for node in strings.iter(f"{{{_NS['m']}}}t")]
    if VARIABLE not in labels:
        raise DataError(f"the SPF workbook does not carry {VARIABLE}; found {labels}")

    values = {}
    for row in sheet.findall(".//m:row", _NS)[1:]:
        cells = _cells(row)
        if cells.get("C") is None:
            continue
        quarter = pd.Period(year=int(float(cells["A"])), quarter=int(float(cells["B"])), freq="Q")
        values[quarter] = float(cells["C"]) / 100.0
    if not values:
        raise DataError(f"the SPF workbook has no observation of {VARIABLE}")
    return pd.Series(values, name=VARIABLE).sort_index()


def load(cache_dir: Path, fetcher: Fetcher = http_fetcher, refresh: bool = False) -> pd.Series:
    """Load ``BILL10``, caching the workbook and its fetch time beside the FRED payloads."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = cache_dir / PAYLOAD
    if refresh or not payload.exists():
        payload.write_bytes(fetcher())
        (cache_dir / f"{VARIABLE}.json").write_text(
            json.dumps({"fetched_at": datetime.now(UTC).isoformat()}), encoding="utf-8"
        )
    return parse_xlsx(payload.read_bytes())

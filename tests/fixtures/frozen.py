"""The frozen FRED capture the suite replays, and the report is rendered from.

Nothing here touches the network. `fetcher` satisfies `rlab.data.fred.Fetcher`, so the
whole pipeline runs against these payloads exactly as it would against the live endpoint --
the only thing swapped is where the bytes come from.

Re-capture with `scripts/capture_fixture.py`, and update `CAPTURED` when you do.
"""

from __future__ import annotations

from pathlib import Path

from rlab.errors import DataError

CAPTURED = "2026-08-17"
DIRECTORY = Path(__file__).resolve().parent / "fred"


def fetcher(series_id: str) -> str:
    """Serve one frozen FRED payload, refusing rather than pretending it is empty."""
    payload = DIRECTORY / f"{series_id}.csv"
    if not payload.exists():
        raise DataError(
            f"{series_id} is not in the frozen capture; re-run scripts/capture_fixture.py"
        )
    return payload.read_text(encoding="utf-8")


def survey_fetcher() -> bytes:
    """Serve the frozen SPF workbook, captured 2026-10-07; its last survey, 2026Q1, predates
    the FRED capture, so freezing it later changes nothing the panel can reach."""
    return (Path(__file__).resolve().parent / "spf" / "median_bill10_level.xlsx").read_bytes()

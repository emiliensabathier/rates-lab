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

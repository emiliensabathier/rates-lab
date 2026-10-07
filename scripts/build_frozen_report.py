"""Render the committed report from the frozen fixture, offline.

The committed `reports/rates.html` is this script's output, and `tests/test_report.py`
asserts the two stay identical outside the chart images. That is what makes the published
page the artefact the suite verifies rather than a separate live pull that happens to
agree with it.

    python scripts/build_frozen_report.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures import frozen  # noqa: E402
from rlab.pipeline import run  # noqa: E402
from rlab.report.build import render  # noqa: E402

OUTPUT = ROOT / "reports" / "rates.html"


def build() -> str:
    """Run the whole pipeline against the frozen payloads and return the page."""
    with tempfile.TemporaryDirectory() as scratch:
        result = run(cache_dir=Path(scratch), fetcher=frozen.fetcher,
                     survey_fetcher=frozen.survey_fetcher)
    return render(result, generated_on=frozen.CAPTURED)


def main() -> None:
    html = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(html):,} bytes) from the {frozen.CAPTURED} capture")


if __name__ == "__main__":  # pragma: no cover
    main()

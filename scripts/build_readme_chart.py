"""Render the one chart the README carries, from the frozen capture.

GitHub shows a committed HTML file as source, not as a page, so the report's charts are
invisible to anyone browsing the repository. This writes the single figure that makes the
repository's point as a raster the README can embed directly.

Run it after `scripts/build_frozen_report.py`, from the same frozen inputs, so the picture
and the page cannot disagree.

    python scripts/build_readme_chart.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from fixtures import frozen  # noqa: E402
from rlab.pipeline import run  # noqa: E402
from rlab.report.charts import decomposition_figure  # noqa: E402

OUTPUT = ROOT / "docs" / "term-premium.png"
DPI = 130


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        result = run(cache_dir=Path(scratch), fetcher=frozen.fetcher,
                     survey_fetcher=frozen.survey_fetcher)

    figure = decomposition_figure(result.observed_10y, result.expectations_10y)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, format="png", dpi=DPI, bbox_inches="tight")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes) from the {frozen.CAPTURED} capture")


if __name__ == "__main__":  # pragma: no cover
    main()

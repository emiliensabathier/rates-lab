"""The committed page is the artefact the suite verifies, not a separate live pull."""

import re
import tempfile
from pathlib import Path

import pytest
from tests.fixtures import frozen

from rlab.pipeline import run
from rlab.report.build import render

COMMITTED = Path(__file__).resolve().parents[1] / "reports" / "rates.html"

SVG_BLOCK = re.compile(r"<svg\b.*?</svg>", re.DOTALL)
EXPECTED_CHARTS = 5


def _without_charts(page: str) -> tuple[str, int]:
    """The page with each inline SVG replaced by a marker, plus the chart count.

    A byte-for-byte comparison of the whole page cannot survive CI: matplotlib's SVG
    output drifts between versions and platforms, and the committed page is generated on
    one machine while CI renders on another. What the comparison is for -- proving the
    published page is the artefact this suite verifies, not a stale or hand-edited file --
    rests entirely on the numbers and the prose, which are compared exactly. The chart
    count is asserted separately so a silently dropped chart still fails.
    """
    return SVG_BLOCK.sub("[CHART]", page), len(SVG_BLOCK.findall(page))


@pytest.fixture(scope="module")
def rendered() -> str:
    with tempfile.TemporaryDirectory() as scratch:
        result = run(cache_dir=Path(scratch), fetcher=frozen.fetcher)
    return render(result, generated_on=frozen.CAPTURED)


def test_the_committed_report_matches_the_frozen_fixture(rendered):
    committed, committed_charts = _without_charts(
        COMMITTED.read_text(encoding="utf-8")
    )
    fresh, fresh_charts = _without_charts(rendered)
    assert committed == fresh
    assert committed_charts == fresh_charts == EXPECTED_CHARTS


def test_the_report_names_every_month_it_refused(rendered):
    with tempfile.TemporaryDirectory() as scratch:
        result = run(cache_dir=Path(scratch), fetcher=frozen.fetcher)
    for date in result.refusals:
        assert date in rendered


def test_the_report_states_the_level_gap_next_to_the_correlation(rendered):
    """Both halves of the comparison, on the same page, or the page is misleading."""
    assert "The correlation is the claim. The level is not." in rendered
    assert "Correlation of levels" in rendered
    assert "Mean gap, this estimate minus Kim-Wright" in rendered


def test_the_report_says_a_breakeven_is_not_expected_inflation(rendered):
    assert "not what the market expects" in rendered


def test_the_page_is_self_contained(rendered):
    """No CDN, no external asset: the report has to open offline.

    The charts are stripped before the check. Matplotlib's SVG carries w3.org and
    creativecommons.org namespace URIs, which are identifiers rather than addresses --
    nothing fetches them — and asserting on the raw page would flag those and hide a real
    remote reference in the noise.
    """
    body, _ = _without_charts(rendered)
    assert "http://" not in body
    assert "https://" not in body
    assert "<script" not in body
    assert "<link" not in body
    assert "src=" not in body

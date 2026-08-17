"""Replay the frozen capture and lock every published figure.

Nothing here touches the network. When these numbers change, either the model changed or
the fixture was re-captured; both are deliberate acts.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.fixtures import frozen

from rlab.errors import DataError
from rlab.pipeline import (
    MATURITY_MONTHS,
    NOMINAL_PILLARS,
    Result,
    agreement,
    build_panel,
    month_end_quotes,
    run,
)

EXPECTED_MONTHS = 295
EXPECTED_FIRST = "2001-07-31"
EXPECTED_LAST = "2026-08-31"

# The seven months whose published quotes do not admit a positive, decreasing discount
# curve. Listed rather than counted: a refusal appearing somewhere new is a different
# fact from one more refusal.
EXPECTED_REFUSALS = {
    "2008-11-30",
    "2011-10-31",
    "2011-11-30",
    "2014-11-30",
    "2015-06-30",
    "2015-09-30",
    "2021-11-30",
}

EXPECTED_TEN_YEAR = {
    "observed": 0.04627299,
    "fitted": 0.04636127,
    "expectations": 0.02411359,
    "term_premium": 0.02224768,
}

EXPECTED_AGREEMENT = {
    "correlation_levels": 0.925241,
    "correlation_changes": 0.841905,
    "mean_gap": 0.00979,
    "observations": 295.0,
}

EXPECTED_EXPLAINED = (0.91978, 0.074898, 0.00464)
EXPECTED_BREAKEVEN_10Y = 0.02230972
EXPECTED_BREAKEVEN_5Y5Y = 0.02275058


@pytest.fixture(scope="module")
def frozen_result() -> Result:
    """One pipeline run against the frozen payloads, shared by every test here."""
    with tempfile.TemporaryDirectory() as scratch:
        return run(cache_dir=Path(scratch), fetcher=frozen.fetcher)


def test_the_panel_spans_the_window_the_one_month_bill_allows(frozen_result):
    assert len(frozen_result.dates) == EXPECTED_MONTHS
    assert frozen_result.dates[0].date().isoformat() == EXPECTED_FIRST
    assert frozen_result.dates[-1].date().isoformat() == EXPECTED_LAST
    assert frozen_result.panel.shape == (EXPECTED_MONTHS, MATURITY_MONTHS.size)


def test_the_refused_months_are_the_ones_that_imply_a_negative_forward(frozen_result):
    assert set(frozen_result.refusals) == EXPECTED_REFUSALS
    for reason in frozen_result.refusals.values():
        assert "positive and decreasing" in reason


def test_the_frozen_run_pins_its_published_ten_year_decomposition(frozen_result):
    latest = {
        "observed": float(frozen_result.observed_10y.iloc[-1]),
        "fitted": float(frozen_result.fitted_10y.iloc[-1]),
        "expectations": float(frozen_result.expectations_10y.iloc[-1]),
        "term_premium": float(frozen_result.term_premium_10y.iloc[-1]),
    }
    for name, expected in EXPECTED_TEN_YEAR.items():
        assert latest[name] == pytest.approx(expected, rel=1e-6)


def test_the_decomposition_is_an_identity_at_every_maturity(frozen_result):
    """fitted = expectations + premium, by construction rather than by fit."""
    rebuilt = frozen_result.acm.risk_neutral + frozen_result.acm.term_premium
    assert np.allclose(rebuilt, frozen_result.acm.fitted, atol=1e-15)


def test_the_model_prices_the_curve_it_was_fitted_to(frozen_result):
    errors = frozen_result.fit_error_bp
    assert errors.mean() < 3.0
    assert errors.max() < 50.0


def test_the_estimate_tracks_the_published_kim_wright_series(frozen_result):
    """The correlation is the claim, and it is pinned. So is the level gap."""
    stats = agreement(frozen_result)
    for name, expected in EXPECTED_AGREEMENT.items():
        assert stats[name] == pytest.approx(expected, rel=1e-4)


def test_the_level_gap_against_kim_wright_is_published_not_hidden(frozen_result):
    """A test that exists to stop the disclaimer being quietly dropped.

    This estimate sits about a percentage point above the Kim-Wright series. If a future
    change closes that gap the assertion below fails, which is the intended prompt to
    rewrite the caveat rather than leave a stale one on the page.
    """
    stats = agreement(frozen_result)
    assert stats["mean_gap"] > 0.005


def test_three_components_carry_the_curve(frozen_result):
    assert frozen_result.pca.explained == pytest.approx(EXPECTED_EXPLAINED, rel=1e-4)
    assert frozen_result.pca.explained.sum() > 0.999


def test_the_breakevens_are_pinned(frozen_result):
    assert frozen_result.breakeven_spot_10y == pytest.approx(EXPECTED_BREAKEVEN_10Y, rel=1e-6)
    assert frozen_result.breakeven_forward_5y5y == pytest.approx(
        EXPECTED_BREAKEVEN_5Y5Y, rel=1e-6
    )


def test_the_policy_path_covers_two_years_in_quarterly_steps(frozen_result):
    assert len(frozen_result.policy_path) == 8
    assert frozen_result.policy_path["start"].iloc[0] == pytest.approx(0.0)
    assert frozen_result.policy_path["start"].iloc[-1] == pytest.approx(1.75)


def test_a_missing_series_raises_rather_than_silently_shrinking_the_panel():
    frame = pd.DataFrame({"DGS1MO": [1.0]}, index=pd.DatetimeIndex(["2020-01-31"]))
    with pytest.raises(DataError, match="missing series"):
        month_end_quotes(frame, NOMINAL_PILLARS)


def test_a_frame_with_no_complete_row_raises():
    index = pd.DatetimeIndex(["2020-01-31", "2020-02-28"])
    frame = pd.DataFrame({name: [1.0, np.nan] for name in NOMINAL_PILLARS}, index=index)
    frame.iloc[0, 0] = np.nan
    with pytest.raises(DataError, match="every pillar"):
        month_end_quotes(frame, NOMINAL_PILLARS)


def test_a_panel_where_every_month_refuses_raises_rather_than_estimating_on_nothing():
    """An inverted-to-absurdity curve cannot bootstrap, and the refusal has to surface."""
    index = pd.DatetimeIndex(["2020-01-31", "2020-02-29"])
    absurd = {name: [5.0, 5.0] for name in NOMINAL_PILLARS}
    absurd["DGS10"] = [-0.9, -0.9]
    with pytest.raises(DataError, match="no panel to estimate on"):
        build_panel(pd.DataFrame(absurd, index=index), NOMINAL_PILLARS)

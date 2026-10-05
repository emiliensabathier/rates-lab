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
    consecutive_months,
    month_end_quotes,
    run,
)

EXPECTED_MONTHS = 301
EXPECTED_FIRST = "2001-07-31"
# The capture is dated 2026-08-17, so August had not finished trading: the last curve is
# July's, dated by the day it printed, not a month-end label in the future.
EXPECTED_LAST = "2026-07-31"

EXPECTED_TEN_YEAR = {
    "observed": 0.04744533,
    "fitted": 0.04752934,
    "expectations": 0.02471124,
    "term_premium": 0.02281809,
}

EXPECTED_AGREEMENT = {
    "correlation_levels": 0.920822,
    "correlation_changes": 0.827809,
    "naive_correlation_levels": 0.850367,
    "naive_correlation_changes": 0.946386,
    "mean_gap": 0.00976565,
    "latest_gap": 0.01413709,
    "latest_kim_wright": 0.008681,
    "observations": 301.0,
}

EXPECTED_EXPLAINED = (0.893028, 0.0756697, 0.0212761)
EXPECTED_BREAKEVEN_10Y = 0.02265397
EXPECTED_BREAKEVEN_5Y5Y = 0.02300663


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


def test_no_month_of_the_capture_implies_a_negative_forward(frozen_result):
    """The seven months a linearly interpolated curve refused were interpolation artefacts.

    Six had a one-month bill a basis point or two above the three-month bill, which linear
    zero interpolation turns into a negative forward just short of three months; the
    seventh printed 0.00% at both and was refused for a zero forward. On the monotone
    curve every month builds, so the panel has no gaps.
    """
    assert frozen_result.refusals == {}
    assert frozen_result.consecutive.all()


def test_every_curve_is_dated_by_the_day_it_printed(frozen_result):
    business_month_ends = frozen_result.dates + pd.offsets.BMonthEnd(0)
    assert (frozen_result.dates <= business_month_ends).all()
    # 2001-09-28 was the last trading day of September 2001; a calendar label would say 30.
    assert pd.Timestamp("2001-09-28") in frozen_result.dates


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
    rebuilt = frozen_result.acm.expectations + frozen_result.acm.term_premium
    assert np.allclose(rebuilt, frozen_result.acm.fitted, atol=1e-15)


def test_the_model_prices_the_curve_it_was_fitted_to(frozen_result):
    errors = frozen_result.fit_error_bp
    assert errors.mean() < 3.0
    assert errors.max() < 50.0


def test_the_estimate_tracks_the_published_kim_wright_series(frozen_result):
    """Every comparison figure is pinned, the naive benchmarks and the level gap included."""
    stats = agreement(frozen_result)
    for name, expected in EXPECTED_AGREEMENT.items():
        assert stats[name] == pytest.approx(expected, rel=1e-4)


def test_in_monthly_changes_the_yield_alone_beats_the_model(frozen_result):
    """The fact the README leads its caveats with. If a change ever reverses it, the
    sentence has to be rewritten rather than left standing."""
    stats = agreement(frozen_result)
    assert stats["naive_correlation_changes"] > stats["correlation_changes"]


def test_in_levels_the_model_beats_the_yield_alone(frozen_result):
    stats = agreement(frozen_result)
    assert stats["correlation_levels"] > stats["naive_correlation_levels"]


def test_the_level_gap_against_kim_wright_is_published_not_hidden(frozen_result):
    """A test that exists to stop the disclaimer being quietly dropped.

    This estimate sits about a percentage point above the Kim-Wright series. If a future
    change closes that gap the assertion below fails, which is the intended prompt to
    rewrite the caveat rather than leave a stale one on the page.
    """
    stats = agreement(frozen_result)
    assert stats["mean_gap"] > 0.005


def test_three_components_carry_the_monthly_curve_changes(frozen_result):
    assert frozen_result.pca.explained == pytest.approx(EXPECTED_EXPLAINED, rel=1e-4)
    assert frozen_result.pca.explained.sum() > 0.98


def test_the_breakevens_are_pinned(frozen_result):
    assert frozen_result.breakeven_spot_10y == pytest.approx(EXPECTED_BREAKEVEN_10Y, rel=1e-6)
    assert frozen_result.breakeven_forward_5y5y == pytest.approx(EXPECTED_BREAKEVEN_5Y5Y, rel=1e-6)


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


def _daily(dates: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex(dates)
    return pd.DataFrame({name: [1.0] * len(index) for name in NOMINAL_PILLARS}, index=index)


def test_a_month_is_dated_by_its_last_quote_not_by_the_calendar():
    quotes = month_end_quotes(_daily(["2024-03-27", "2024-03-28", "2024-04-30"]), NOMINAL_PILLARS)
    assert list(quotes.index) == [pd.Timestamp("2024-03-28"), pd.Timestamp("2024-04-30")]


def test_a_final_month_that_has_not_finished_is_dropped():
    quotes = month_end_quotes(_daily(["2026-07-31", "2026-08-13"]), NOMINAL_PILLARS)
    assert list(quotes.index) == [pd.Timestamp("2026-07-31")]


def test_consecutive_months_flags_the_pair_that_straddles_a_gap():
    dates = pd.DatetimeIndex(["2011-09-30", "2011-10-31", "2011-12-30", "2012-01-31"])
    assert list(consecutive_months(dates)) == [True, False, True]

import numpy as np
import pandas as pd
import pytest

from rlab.curve.bootstrap import PiecewiseZeroCurve
from rlab.errors import DataError
from rlab.inflation import (
    REAL_CURVE_START,
    breakeven,
    forward_breakeven,
    require_real_coverage,
)


def flat(rate: float) -> PiecewiseZeroCurve:
    return PiecewiseZeroCurve(np.array([0.25, 30.0]), np.array([rate, rate]))


def test_breakeven_is_the_gap_between_the_nominal_and_real_zero():
    assert breakeven(flat(0.046), flat(0.024), 10.0) == pytest.approx(0.022)


def test_forward_breakeven_of_two_flat_curves_equals_the_spot_breakeven():
    assert forward_breakeven(flat(0.046), flat(0.024), 5.0, 10.0) == pytest.approx(0.022)


def test_forward_breakeven_reads_the_gap_between_the_two_forward_rates():
    nominal = PiecewiseZeroCurve(np.array([5.0, 10.0]), np.array([0.04, 0.05]))
    real = PiecewiseZeroCurve(np.array([5.0, 10.0]), np.array([0.02, 0.02]))
    expected = nominal.forward(5.0, 10.0) - real.forward(5.0, 10.0)
    assert forward_breakeven(nominal, real, 5.0, 10.0) == pytest.approx(expected)


def test_a_thirty_year_real_curve_before_2010_raises():
    with pytest.raises(DataError, match="DFII30"):
        require_real_coverage(pd.Timestamp("2005-06-01"), 30.0)


def test_any_real_curve_before_2003_raises():
    with pytest.raises(DataError, match="2003-01-02"):
        require_real_coverage(pd.Timestamp("2001-06-01"), 10.0)


def test_coverage_passes_inside_the_published_window():
    require_real_coverage(pd.Timestamp("2026-08-05"), 30.0)


def test_the_real_curve_starts_when_the_tips_series_do():
    assert REAL_CURVE_START == pd.Timestamp("2003-01-02")

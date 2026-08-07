import numpy as np
import pytest

from rlab.curve.bootstrap import PiecewiseZeroCurve
from rlab.errors import ModelError
from rlab.policy import implied_change_bp, implied_path


def flat(rate: float) -> PiecewiseZeroCurve:
    return PiecewiseZeroCurve(np.array([0.25, 30.0]), np.array([rate, rate]))


def test_a_flat_curve_prices_no_policy_change():
    assert implied_change_bp(flat(0.04)) == pytest.approx(0.0, abs=1e-8)


def test_an_upward_sloping_curve_prices_hikes():
    curve = PiecewiseZeroCurve(np.array([0.25, 2.0]), np.array([0.03, 0.05]))
    assert implied_change_bp(curve, horizon_years=1.0) > 0


def test_a_downward_sloping_curve_prices_cuts():
    curve = PiecewiseZeroCurve(np.array([0.25, 2.0]), np.array([0.05, 0.03]))
    assert implied_change_bp(curve, horizon_years=1.0) < 0


def test_the_path_covers_the_horizon_at_the_requested_step():
    frame = implied_path(flat(0.04), horizon_years=1.0, step_years=0.25)
    assert list(frame["start"]) == [0.0, 0.25, 0.5, 0.75]


def test_the_quoted_forward_is_the_simple_equivalent_of_the_continuous_one():
    frame = implied_path(flat(0.04), horizon_years=0.5, step_years=0.25)
    row = frame.iloc[0]
    expected = (np.exp(row["forward_continuous"] * 0.25) - 1) * 4
    assert row["forward_quoted"] == pytest.approx(expected)


def test_a_step_that_does_not_divide_the_horizon_raises():
    with pytest.raises(ModelError, match="whole number"):
        implied_path(flat(0.04), horizon_years=1.0, step_years=0.3)

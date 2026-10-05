import numpy as np
import pytest

from rlab.curve.bootstrap import MonotoneCurve, PiecewiseZeroCurve, bootstrap
from rlab.errors import ModelError

# The published CMT curve on 2026-08-05, in percent. The 20-year sits above the 30-year;
# that inversion is real and the bootstrap must carry it through rather than smooth it.
CMT_2026_08_05 = {
    1 / 12: 3.77,
    0.25: 3.89,
    0.5: 3.98,
    1.0: 4.03,
    2.0: 4.18,
    3.0: 4.24,
    5.0: 4.33,
    7.0: 4.47,
    10.0: 4.63,
    20.0: 5.18,
    30.0: 5.17,
}
MATURITIES = np.array(sorted(CMT_2026_08_05))
PAR = np.array([CMT_2026_08_05[m] for m in MATURITIES]) / 100


def test_round_trip_recovers_every_published_par_yield():
    # The load-bearing test. Bootstrapping is exact at the pillars by construction, so any
    # failure here is a bug in the arithmetic and not a tolerance that needs loosening.
    curve = bootstrap(MATURITIES, PAR)
    for maturity, published in zip(MATURITIES, PAR, strict=True):
        if maturity < 2.0:
            continue  # bills are priced by their own convention, checked separately
        assert curve.par(maturity) * 1e4 == pytest.approx(published * 1e4, abs=0.1)


def test_bills_use_simple_interest_up_to_six_months():
    curve = bootstrap(MATURITIES, PAR)
    assert curve.discount(0.5) == pytest.approx(1 / (1 + 0.0398 * 0.5), rel=1e-12)


def test_the_one_year_bill_uses_semiannual_compounding():
    curve = bootstrap(MATURITIES, PAR)
    assert curve.discount(1.0) == pytest.approx(1 / (1 + 0.0403 / 2) ** 2, rel=1e-12)


def test_discount_factors_are_positive_and_decreasing():
    curve = bootstrap(MATURITIES, PAR)
    grid = np.linspace(0.1, 30.0, 300)
    discounts = curve.discount(grid)
    assert np.all(discounts > 0)
    assert np.all(np.diff(discounts) < 0)


def test_a_flat_par_curve_bootstraps_to_the_same_flat_zero_curve():
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    curve = bootstrap(maturities, np.full_like(maturities, 0.04))
    # A flat par curve implies a flat zero curve at the same semiannual rate.
    expected_continuous = 2 * np.log(1 + 0.04 / 2)
    assert curve.zero(10.0) == pytest.approx(expected_continuous, abs=1e-8)


def test_a_par_yield_that_implies_a_negative_discount_factor_raises():
    # An absurdly steep curve drives the last discount factor through zero. Producing a
    # number there would be publishing an arbitrage.
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    par = np.array([0.04, 0.04, 0.04, 0.04, 5.0])
    with pytest.raises(ModelError):
        bootstrap(maturities, par)


def test_unsorted_maturities_raise():
    with pytest.raises(ModelError, match="increasing"):
        bootstrap(np.array([2.0, 1.0]), np.array([0.04, 0.04]))


def test_nan_in_par_yields_raises():
    maturities = np.array([0.5, 1.0, 2.0])
    par = np.array([0.04, np.nan, 0.04])
    with pytest.raises(ModelError, match="par_yields contains non-finite values"):
        bootstrap(maturities, par)


def test_inf_in_maturities_raises():
    maturities = np.array([0.5, 1.0, np.inf])
    par = np.array([0.04, 0.04, 0.04])
    with pytest.raises(ModelError, match="maturities contains non-finite values"):
        bootstrap(maturities, par)


def test_piecewise_curve_interpolates_linearly_on_zero_rates():
    curve = PiecewiseZeroCurve(np.array([1.0, 3.0]), np.array([0.02, 0.04]))
    assert curve.zero(2.0) == pytest.approx(0.03)


def test_piecewise_curve_extrapolates_flat_beyond_its_range():
    curve = PiecewiseZeroCurve(np.array([1.0, 3.0]), np.array([0.02, 0.04]))
    assert curve.zero(50.0) == pytest.approx(0.04)


def test_the_bootstrap_returns_a_monotone_curve_not_a_linear_one():
    assert isinstance(bootstrap(MATURITIES, PAR), MonotoneCurve)


def test_the_forward_curve_has_no_jump_at_any_interior_pillar():
    # Linear interpolation of zero rates puts a jump in the instantaneous forward at every
    # pillar, and a policy path read off it inherits those jumps as fake steps. The monotone
    # cubic on log discount factors is C1, so the forward is continuous across each pillar.
    curve = bootstrap(MATURITIES, PAR)
    width = 1e-4
    for pillar in MATURITIES[1:-1]:
        left = curve.forward(pillar - 2 * width, pillar - width)
        right = curve.forward(pillar + width, pillar + 2 * width)
        assert abs(left - right) * 1e4 < 0.5


def test_quarterly_forwards_inside_one_segment_are_not_equally_spaced():
    # The audit artefact: on a linear-in-zeros curve, three-month forwards between the one-
    # and two-year pillars rise by identical steps. Anything smoother breaks that pattern.
    curve = bootstrap(MATURITIES, PAR)
    starts = np.array([1.0, 1.25, 1.5, 1.75])
    forwards = np.array([curve.forward(s, s + 0.25) for s in starts])
    steps = np.diff(forwards) * 1e4
    assert np.ptp(steps) > 0.1


def test_the_one_month_bills_reprice_exactly_on_the_monotone_curve():
    curve = bootstrap(MATURITIES, PAR)
    assert curve.discount(1 / 12) == pytest.approx(1 / (1 + 0.0377 / 12), rel=1e-12)


def test_a_flat_front_end_is_a_zero_forward_and_is_accepted():
    # 2015-09-30 printed 0.00% at one and three months. A zero forward rate is not an
    # arbitrage, so the month must build rather than be refused as a negative forward.
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0])
    par = np.array([0.0, 0.0, 0.0008, 0.0033, 0.0064])
    curve = bootstrap(maturities, par)
    assert curve.forward(1 / 12, 0.25) == pytest.approx(0.0, abs=1e-12)


def test_an_inverted_bill_front_end_is_refused_as_a_negative_forward():
    # One month at 0.05% and three months at 0.01% means less interest for holding three
    # months than for holding one: the one-to-three-month forward is negative, and no
    # interpolation choice can remove that.
    maturities = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0])
    par = np.array([0.0005, 0.0001, 0.0006, 0.0012, 0.0025])
    with pytest.raises(ModelError, match="negative forward"):
        bootstrap(maturities, par)

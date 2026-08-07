import numpy as np
import pytest

from rlab.curve.base import Curve
from rlab.errors import ModelError


class FlatCurve(Curve):
    """A curve whose continuously compounded zero rate is the same at every maturity."""

    def __init__(self, rate: float):
        self._rate = rate

    def discount(self, t):
        return np.exp(-self._rate * np.asarray(t, dtype=float))


def test_zero_of_a_flat_curve_is_the_flat_rate():
    assert FlatCurve(0.04).zero(7.0) == pytest.approx(0.04)


def test_forward_of_a_flat_curve_is_the_flat_rate():
    # On a flat curve every forward equals the spot: the no-arbitrage definition
    # f(t1,t2) = (z2*t2 - z1*t1)/(t2-t1) collapses to z. Note this alone is not enough
    # regression protection: see test_forward_of_a_sloped_curve_matches_the_closed_form below
    # for why.
    assert FlatCurve(0.04).forward(2.0, 5.0) == pytest.approx(0.04)


class SlopedCurve(Curve):
    """A curve whose continuously compounded zero rate rises linearly with maturity."""

    def discount(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-(0.02 + 0.005 * t) * t)


def test_forward_of_a_sloped_curve_matches_the_closed_form():
    # zero(t) = 0.02 + 0.005*t here, so zero(2) = 0.03 and zero(5) = 0.045. On a flat curve
    # the correct formula f(t1,t2) = (z2*t2 - z1*t1)/(t2-t1) and the transposed-operand bug
    # (z1*t2 - z2*t1)/(t2-t1) agree, so the flat-curve test above cannot tell them apart. Here
    # they diverge: the correct formula gives (0.045*5 - 0.03*2)/3 = 0.055, the transposed one
    # gives (0.03*5 - 0.045*2)/3 = 0.02. Asserting the literal 0.055 (not a value recomputed
    # from the curve) is what makes this test able to fail on the wrong formula.
    assert SlopedCurve().forward(2.0, 5.0) == pytest.approx(0.055)


def test_par_yield_of_a_flat_curve_matches_the_flat_rate_in_semiannual_terms():
    curve = FlatCurve(0.04)
    semiannual_equivalent = 2 * (np.exp(0.04 / 2) - 1)
    assert curve.par(10.0) == pytest.approx(semiannual_equivalent, abs=1e-10)


def test_zero_at_zero_maturity_raises_rather_than_dividing_by_zero():
    with pytest.raises(ModelError, match="maturity"):
        FlatCurve(0.04).zero(0.0)


def test_forward_with_reversed_maturities_raises():
    with pytest.raises(ModelError, match="t2 > t1"):
        FlatCurve(0.04).forward(5.0, 2.0)

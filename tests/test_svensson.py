import numpy as np
import pytest

from rlab.curve.svensson import SvenssonCurve, SvenssonParams

ECB_2026_08_06 = SvenssonParams(
    beta0=1.2866240787,
    beta1=0.7895412895,
    beta2=2.0356419294,
    beta3=7.441779832,
    tau1=1.0016124277,
    tau2=15.5479286615,
)

ECB_PUBLISHED_SPOT = {
    0.25: 2.260013643,
    0.5: 2.3921412994,
    1.0: 2.5529047267,
    2.0: 2.6725142222,
    5.0: 2.8034029096,
    10.0: 3.146678757,
    30.0: 3.5969961326,
}


@pytest.mark.parametrize(("maturity", "published"), sorted(ECB_PUBLISHED_SPOT.items()))
def test_evaluator_reproduces_the_ecb_published_spot_rates(maturity: float, published: float):
    # The ECB publishes both the parameters and the curve they generate. Reproducing the
    # second from the first, to a hundredth of a basis point, pins the functional form and
    # the units without relying on anyone's memory of the paper.
    curve = SvenssonCurve(ECB_2026_08_06, compounding="annual")
    assert curve.rate(maturity) * 100 == pytest.approx(published, abs=1e-6)


def test_the_short_end_limit_is_beta0_plus_beta1():
    curve = SvenssonCurve(ECB_2026_08_06, compounding="annual")
    assert curve.rate(1e-8) * 100 == pytest.approx(
        ECB_2026_08_06.beta0 + ECB_2026_08_06.beta1, abs=1e-4
    )


def test_the_long_end_limit_is_beta0():
    # The maturity has to be absurd, and that is the point of the test. Svensson's curvature
    # terms decay like 1/t, not exponentially, so at t=500 the second hump still contributes
    # 0.237 of a percentage point — twenty-three times a 1e-2 tolerance. Only at t=1e6 does
    # the gap fall to 1.2e-4. Asserting the limit at a plausible maturity would be asserting
    # something false.
    curve = SvenssonCurve(ECB_2026_08_06, compounding="annual")
    assert curve.rate(1e6) * 100 == pytest.approx(ECB_2026_08_06.beta0, abs=1e-3)


def test_annual_compounding_discounts_differently_from_continuous():
    # beta0 is 5.0 and not 0.05 because SvenssonParams holds betas in PERCENT — rate() is
    # what divides by 100. A 0.05 here would be five basis points, and the discount factors
    # below would be wrong by a factor the test would not catch.
    flat = SvenssonParams(beta0=5.0, beta1=0.0, beta2=0.0, beta3=0.0, tau1=1.0, tau2=10.0)
    annual = SvenssonCurve(flat, compounding="annual").discount(10.0)
    continuous = SvenssonCurve(flat, compounding="continuous").discount(10.0)
    assert annual == pytest.approx(1.05**-10)
    assert continuous == pytest.approx(np.exp(-0.5))


def test_round_trip_through_the_parameter_array():
    restored = SvenssonParams.from_array(ECB_2026_08_06.as_array())
    assert restored == ECB_2026_08_06

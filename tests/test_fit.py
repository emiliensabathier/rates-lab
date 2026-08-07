import numpy as np
import pytest

from rlab.curve.fit import fit_svensson
from rlab.curve.svensson import SvenssonCurve, SvenssonParams
from rlab.errors import ModelError

ECB_2026_08_06 = SvenssonParams(
    beta0=1.2866240787,
    beta1=0.7895412895,
    beta2=2.0356419294,
    beta3=7.441779832,
    tau1=1.0016124277,
    tau2=15.5479286615,
)
ECB_MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def test_fitting_the_ecb_curve_reproduces_the_ecb_curve():
    # The external validation. We fit our own parameters to the ECB's published spot rates
    # and require the curve we get back to match the curve their parameters generate, to
    # within one basis point at every pillar.
    #
    # The test is on the CURVE, not on the betas. Svensson parameters are not uniquely
    # identified when tau1 and tau2 approach each other, so two different beta vectors can
    # describe the same curve — asserting equality of betas would fail on correct code.
    official = SvenssonCurve(ECB_2026_08_06, compounding="annual")
    published = official.rate(ECB_MATURITIES)

    fitted = fit_svensson(ECB_MATURITIES, published, compounding="annual")

    worst_bp = np.max(np.abs(fitted.rate(ECB_MATURITIES) - published)) * 1e4
    assert worst_bp < 1.0


def test_a_flat_input_curve_is_fitted_flat():
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    rates = np.full_like(maturities, 0.04)
    fitted = fit_svensson(maturities, rates, compounding="continuous")
    assert np.allclose(fitted.rate(maturities), 0.04, atol=1e-5)


def test_an_unfittable_curve_raises_instead_of_publishing_a_bad_fit():
    # A sawtooth cannot be represented by four betas and two decays. The fitter must say so
    # rather than return the least-bad nonsense.
    maturities = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    rates = np.array([0.01, 0.09, 0.01, 0.09, 0.01, 0.09])
    with pytest.raises(ModelError, match="basis points"):
        fit_svensson(maturities, rates, compounding="continuous")


def test_mismatched_input_lengths_raise():
    with pytest.raises(ModelError, match="same length"):
        fit_svensson(np.array([1.0, 2.0]), np.array([0.04]), compounding="continuous")


def test_fewer_than_six_pillars_raises():
    maturities = np.array([1.0, 2.0, 5.0])
    with pytest.raises(ModelError, match="at least 6"):
        fit_svensson(maturities, np.full_like(maturities, 0.04), compounding="continuous")

"""Bootstrapping zero rates from constant-maturity Treasury par yields.

Constant-maturity Treasury quotes are par yields, not zero rates: a five-year CMT of 4.33%
is the coupon a five-year Treasury would carry to price at par. Discounting cash flows with
them directly is the single most common way to get a curve wrong, so this module solves the
discount factors that reprice each par bond exactly, one pillar at a time.

Conventions, and they are the Treasury's, not a convenience:
  - tenors up to six months are bill quotes on a coupon-equivalent basis, simple interest;
  - the one-year bill is quoted semiannually compounded;
  - two years and beyond are semiannual-coupon par bonds, actual/actual on the coupons.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from rlab.curve.base import Curve
from rlab.errors import ModelError

CMT_TENORS: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}

# Continuously compounded zero rates are searched in this bracket. brentq is used instead of
# a Newton step precisely because it fails loudly when no root exists in the bracket: a
# solver that clipped to the boundary would return a presentable and wrong curve.
ZERO_BRACKET = (-0.05, 1.00)


class PiecewiseZeroCurve(Curve):
    """Continuously compounded zeros at pillars, linear in between, flat outside."""

    def __init__(self, times: np.ndarray, zeros: np.ndarray):
        self.times = np.asarray(times, dtype=float)
        self.zeros = np.asarray(zeros, dtype=float)
        if self.times.shape != self.zeros.shape:
            raise ModelError("times and zeros must have the same length")

    def zero(self, t: float | np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float), self.times, self.zeros)

    def discount(self, t: float | np.ndarray) -> np.ndarray:
        maturity = np.asarray(t, dtype=float)
        return np.exp(-self.zero(maturity) * maturity)


def _bill_discount(maturity: float, par_yield: float) -> float:
    if maturity <= 0.5:
        return 1 / (1 + par_yield * maturity)
    return 1 / (1 + par_yield / 2) ** 2


def bootstrap(maturities: np.ndarray, par_yields: np.ndarray) -> PiecewiseZeroCurve:
    """Solve the zero curve that reprices every published par yield exactly."""
    maturities = np.asarray(maturities, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    if maturities.shape != par_yields.shape:
        raise ModelError("maturities and par yields must have the same length")
    if np.any(np.diff(maturities) <= 0):
        raise ModelError("maturities must be strictly increasing")

    times: list[float] = []
    zeros: list[float] = []

    for maturity, par_yield in zip(maturities, par_yields, strict=True):
        if maturity <= 1.0:
            discount = _bill_discount(float(maturity), float(par_yield))
            times.append(float(maturity))
            zeros.append(-np.log(discount) / maturity)
            continue

        periods = int(round(maturity * 2))
        if abs(periods - maturity * 2) > 1e-9:
            raise ModelError(f"coupon pillar {maturity} is not a whole number of half-years")
        coupon_times = np.arange(1, periods + 1) / 2

        def price_gap(candidate_zero: float, _m=maturity, _c=par_yield, _t=coupon_times) -> float:
            trial = PiecewiseZeroCurve(
                np.array(times + [_m], dtype=float), np.array(zeros + [candidate_zero])
            )
            discounts = trial.discount(_t)
            return float(_c / 2 * discounts.sum() + discounts[-1] - 1)

        low, high = ZERO_BRACKET
        if price_gap(low) * price_gap(high) > 0:
            raise ModelError(
                f"no zero rate in {ZERO_BRACKET} reprices the {maturity:g}-year par yield "
                f"of {par_yield:.4%}"
            )
        solved = brentq(price_gap, low, high, xtol=1e-14, rtol=1e-15)
        times.append(float(maturity))
        zeros.append(float(solved))

    curve = PiecewiseZeroCurve(np.array(times), np.array(zeros))
    grid = np.linspace(min(times), max(times), 500)
    discounts = curve.discount(grid)
    if np.any(discounts <= 0) or np.any(np.diff(discounts) >= 0):
        raise ModelError("bootstrapped discount factors are not positive and decreasing")
    return curve

"""Bootstrapping zero rates from constant-maturity Treasury par yields.

Constant-maturity Treasury quotes are par yields, not zero rates: a five-year CMT of 4.33%
is the coupon a five-year Treasury would carry to price at par. Discounting cash flows with
them directly is the single most common way to get a curve wrong, so this module solves the
discount factors that reprice each par bond exactly.

Conventions:
  - tenors up to six months are bill quotes on a bond-equivalent basis, simple interest
    over the tenor as a fraction of a year (1/12, 1/4, 1/2 -- no day-count calendar);
  - the one-year bill is quoted semiannually compounded;
  - two years and beyond are par bonds paying half the coupon every half year, on a regular
    half-year schedule. No day-count calendar is modelled: a CMT is a par yield on a
    notional bond issued on a coupon date, so every period is exactly half a year.

Interpolation between pillars is a monotone cubic (PCHIP) on the log discount factor,
anchored at zero at maturity zero. Linear interpolation of zero rates is the textbook
default and it is wrong for anything read off forwards: the instantaneous forward jumps at
every pillar, and a policy path drawn from it shows steps that are an artefact of where the
Treasury happens to publish. PCHIP is C1, so forwards are continuous, and it is monotone, so
pillars with non-negative forwards between them never acquire a negative forward from the
interpolation itself. Because PCHIP is not local, a later pillar moves the shape of earlier
segments, so the coupon pillars are solved jointly rather than one at a time.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq, root

from rlab.curve.base import Curve
from rlab.errors import ModelError

# Continuously compounded zero rates are searched in this bracket. brentq is used instead of
# a Newton step precisely because it fails loudly when no root exists in the bracket: a
# solver that clipped to the boundary would return a presentable and wrong curve.
ZERO_BRACKET = (-0.05, 1.00)

# A forward below this, in decimals, is negative rather than numerically zero. A flat front
# end (2015-09-30 printed 0.00% at one and three months) is a zero forward, which is not an
# arbitrage, and the month must build.
FORWARD_TOLERANCE = 1e-10
# Repricing tolerance of the joint solve, in price per unit face.
REPRICE_TOLERANCE = 1e-12
CHECK_POINTS = 2000


class PiecewiseZeroCurve(Curve):
    """Continuously compounded zeros at pillars, linear in between, flat outside.

    The starting point of the joint solve, and a simple hand-built curve for tests. Its
    forwards jump at every pillar, which is why `bootstrap` does not return it.
    """

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


class MonotoneCurve(Curve):
    """Log discount factors by monotone cubic (PCHIP) through the origin and the pillars.

    Beyond the last pillar the zero rate is held flat. Nothing in this package reads the
    curve past its last pillar; the extrapolation only has to be defined, not right.
    """

    def __init__(self, times: np.ndarray, zeros: np.ndarray):
        self.times = np.asarray(times, dtype=float)
        self.zeros = np.asarray(zeros, dtype=float)
        if self.times.shape != self.zeros.shape:
            raise ModelError("times and zeros must have the same length")
        if self.times.size == 0 or self.times[0] <= 0:
            raise ModelError("pillar maturities must be positive")
        knots = np.concatenate([[0.0], self.times])
        log_discount = np.concatenate([[0.0], self.zeros * self.times])
        self._spline = PchipInterpolator(knots, log_discount, extrapolate=False)

    def discount(self, t: float | np.ndarray) -> np.ndarray:
        maturity = np.asarray(t, dtype=float)
        last = self.times[-1]
        inside = self._spline(np.clip(maturity, 0.0, last))
        log_discount = np.where(maturity > last, self.zeros[-1] * maturity, inside)
        return np.exp(-log_discount)


def _bill_discount(maturity: float, par_yield: float) -> float:
    if maturity <= 0.5:
        return 1 / (1 + par_yield * maturity)
    return 1 / (1 + par_yield / 2) ** 2


def _validate(maturities: np.ndarray, par_yields: np.ndarray) -> None:
    if maturities.shape != par_yields.shape:
        raise ModelError("maturities and par yields must have the same length")
    if not np.all(np.isfinite(maturities)):
        raise ModelError("maturities contains non-finite values (NaN or inf)")
    if not np.all(np.isfinite(par_yields)):
        raise ModelError("par_yields contains non-finite values (NaN or inf)")
    if np.any(np.diff(maturities) <= 0):
        raise ModelError("maturities must be strictly increasing")


def _coupon_times(maturity: float) -> np.ndarray:
    periods = int(round(maturity * 2))
    if abs(periods - maturity * 2) > 1e-9:
        raise ModelError(f"coupon pillar {maturity} is not a whole number of half-years")
    return np.arange(1, periods + 1) / 2


def _par_gap(curve: Curve, maturity: float, par_yield: float) -> float:
    """Price of the par bond minus par, per unit face."""
    discounts = curve.discount(_coupon_times(maturity))
    return float(par_yield / 2 * discounts.sum() + discounts[-1] - 1)


def _with_coupon_zeros(zeros: np.ndarray, coupon: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A new zero vector with the coupon pillars replaced; the input is left untouched."""
    updated = zeros.copy()
    updated[coupon] = values
    return updated


def _linear_bootstrap(maturities: np.ndarray, par_yields: np.ndarray) -> np.ndarray:
    """Pillar zeros under linear interpolation, solved one pillar at a time.

    The starting point of the joint solve, and the place where quotes that admit no
    discount curve at all are caught, by brentq failing to bracket a root.
    """
    times: list[float] = []
    zeros: list[float] = []
    for maturity, par_yield in zip(maturities, par_yields, strict=True):
        if maturity <= 1.0:
            discount = _bill_discount(float(maturity), float(par_yield))
            times.append(float(maturity))
            zeros.append(-np.log(discount) / maturity)
            continue

        def price_gap(candidate: float, _m=float(maturity), _c=float(par_yield)) -> float:
            trial = PiecewiseZeroCurve(np.array([*times, _m]), np.array([*zeros, candidate]))
            return _par_gap(trial, _m, _c)

        low, high = ZERO_BRACKET
        if price_gap(low) * price_gap(high) > 0:
            raise ModelError(
                f"no zero rate in {ZERO_BRACKET} reprices the {maturity:g}-year par yield "
                f"of {par_yield:.4%}"
            )
        solved = float(brentq(price_gap, low, high, xtol=1e-14, rtol=1e-15))
        times.append(float(maturity))
        zeros.append(solved)
    return np.array(zeros)


def _require_no_negative_forward(curve: MonotoneCurve) -> None:
    grid = np.linspace(0.0, float(curve.times[-1]), CHECK_POINTS)
    log_discount = -np.log(curve.discount(grid))
    lowest = float((np.diff(log_discount) / np.diff(grid)).min())
    if lowest < -FORWARD_TOLERANCE:
        raise ModelError(
            f"the quotes imply a negative forward rate (lowest {lowest * 1e4:.1f} bp), "
            "so no positive, non-increasing discount curve reprices them"
        )


def bootstrap(maturities: np.ndarray, par_yields: np.ndarray) -> MonotoneCurve:
    """Solve the monotone-cubic zero curve that reprices every published par yield exactly."""
    maturities = np.asarray(maturities, dtype=float)
    par_yields = np.asarray(par_yields, dtype=float)
    _validate(maturities, par_yields)

    zeros = _linear_bootstrap(maturities, par_yields)
    coupon = maturities > 1.0

    if coupon.any():

        def gaps(candidate: np.ndarray) -> np.ndarray:
            trial = MonotoneCurve(maturities, _with_coupon_zeros(zeros, coupon, candidate))
            pairs = zip(maturities[coupon], par_yields[coupon], strict=True)
            return np.array([_par_gap(trial, m, c) for m, c in pairs])

        solution = root(gaps, zeros[coupon], method="hybr", options={"xtol": 1e-14})
        worst = float(np.max(np.abs(gaps(solution.x))))
        if worst > REPRICE_TOLERANCE:
            raise ModelError(
                f"the joint bootstrap leaves a par bond mispriced by {worst:.2e} "
                f"({solution.message})"
            )
        zeros = _with_coupon_zeros(zeros, coupon, solution.x)

    curve = MonotoneCurve(maturities, zeros)
    _require_no_negative_forward(curve)
    return curve

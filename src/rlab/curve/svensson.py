"""The Nelson-Siegel-Svensson parametric curve.

Four betas and two decays. The form below is the one the ECB publishes parameters for, and
`tests/test_svensson.py` proves it by regenerating their published spot rates from their
published parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rlab.curve.base import Curve
from rlab.errors import ModelError

Compounding = Literal["annual", "continuous"]


@dataclass(frozen=True)
class SvenssonParams:
    """Six Svensson parameters. Betas are in percent, taus in years."""

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.beta0, self.beta1, self.beta2, self.beta3, self.tau1, self.tau2], dtype=float
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> SvenssonParams:
        return cls(*(float(v) for v in values))


def _loading(t: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Slope and curvature loadings for one decay parameter."""
    x = t / tau
    # x -> 0 makes (1 - exp(-x)) / x indeterminate; its limit is 1 and the curvature term's
    # limit is 0. Substituting the limits keeps the short end finite instead of NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(x > 1e-8, (1 - np.exp(-x)) / np.where(x > 1e-8, x, 1.0), 1.0)
    curvature = slope - np.exp(-x)
    return slope, curvature


class SvenssonCurve(Curve):
    """A Svensson curve, aware of the compounding convention its rates are quoted in."""

    def __init__(self, params: SvenssonParams, compounding: Compounding):
        if params.tau1 <= 0 or params.tau2 <= 0:
            raise ModelError(f"Svensson decays must be positive, got {params.tau1}, {params.tau2}")
        if compounding not in ("annual", "continuous"):
            raise ModelError(f"unknown compounding {compounding!r}")
        self.params = params
        self.compounding = compounding

    def rate(self, t: float | np.ndarray) -> np.ndarray:
        """The Svensson rate, as a decimal, in this curve's own compounding convention."""
        maturity = np.asarray(t, dtype=float)
        p = self.params
        slope1, curve1 = _loading(maturity, p.tau1)
        _, curve2 = _loading(maturity, p.tau2)
        percent = p.beta0 + p.beta1 * slope1 + p.beta2 * curve1 + p.beta3 * curve2
        return percent / 100

    def discount(self, t: float | np.ndarray) -> np.ndarray:
        maturity = np.asarray(t, dtype=float)
        rate = self.rate(maturity)
        if self.compounding == "annual":
            return (1 + rate) ** -maturity
        return np.exp(-rate * maturity)

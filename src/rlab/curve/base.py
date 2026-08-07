"""The single curve interface every consumer talks to.

One abstract method, `discount`. Zero rates, forward rates and par yields are all derived
from it here, so the arithmetic exists in exactly one place and a new curve construction
cannot quietly disagree with an old one about what a forward rate means.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from rlab.errors import ModelError


class Curve(ABC):
    """A discount curve. Maturities are in years, rates are decimals."""

    @abstractmethod
    def discount(self, t: float | np.ndarray) -> np.ndarray:
        """Present value of one unit paid at maturity `t`. Contract: `t >= 0`."""

    def zero(self, t: float | np.ndarray) -> np.ndarray:
        """Continuously compounded zero rate."""
        maturity = np.asarray(t, dtype=float)
        if np.any(maturity <= 0):
            raise ModelError("zero rate undefined at maturity <= 0")
        return -np.log(self.discount(maturity)) / maturity

    def forward(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between two maturities."""
        if not t2 > t1:
            raise ModelError(f"forward rate requires t2 > t1, got t1={t1} t2={t2}")
        return float(
            (self.zero(t2) * t2 - self.zero(t1) * t1) / (t2 - t1)
        )

    def par(self, t: float) -> float:
        """Semiannual-coupon par yield at maturity `t`, the CMT quoting convention."""
        if t <= 0:
            raise ModelError("par yield undefined at maturity <= 0")
        periods = int(round(t * 2))
        if abs(periods - t * 2) > 1e-9:
            raise ModelError(f"par yield needs a whole number of semiannual periods, got {t}")
        times = np.arange(1, periods + 1) / 2
        discounts = self.discount(times)
        return float(2 * (1 - discounts[-1]) / discounts.sum())

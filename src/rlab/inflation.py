"""Breakeven inflation from the nominal and real curves.

The reserve that has to travel with every number in this module, and which the report
states in words: a breakeven is expected inflation plus an inflation risk premium minus a
liquidity premium. Nothing public separates the three. A breakeven of 2.3% is what the
market charges, not what the market expects.
"""

from __future__ import annotations

import pandas as pd

from rlab.curve.base import Curve
from rlab.errors import DataError

REAL_CURVE_START = pd.Timestamp("2003-01-02")
THIRTY_YEAR_REAL_START = pd.Timestamp("2010-02-22")


def require_real_coverage(as_of: pd.Timestamp, tenor_years: float) -> None:
    """Raise unless the TIPS series needed for this tenor was published on this date."""
    if as_of < REAL_CURVE_START:
        raise DataError(
            f"no real curve before {REAL_CURVE_START.date()} (TIPS series start there), "
            f"asked for {as_of.date()}"
        )
    if tenor_years > 10.0 and as_of < THIRTY_YEAR_REAL_START:
        raise DataError(
            f"DFII30 starts on {THIRTY_YEAR_REAL_START.date()}, asked for {as_of.date()}"
        )


def breakeven(nominal: Curve, real: Curve, t: float) -> float:
    """Spot breakeven inflation at maturity `t`."""
    return float(nominal.zero(t) - real.zero(t))


def forward_breakeven(nominal: Curve, real: Curve, t1: float, t2: float) -> float:
    """Forward breakeven between two maturities. The 5y5y is t1=5, t2=10."""
    return float(nominal.forward(t1, t2) - real.forward(t1, t2))

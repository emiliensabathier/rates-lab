"""The policy path the curve prices.

A sequence of three-month forward rates read off the curve is the market's implied path for
the short rate. It is not a forecast and it is not, on its own, an expectation either —
rlab.termpremium is what separates the two. This module only reads the curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rlab.curve.base import Curve
from rlab.errors import ModelError

QUARTER = 0.25


def implied_path(
    curve: Curve, horizon_years: float = 2.0, step_years: float = QUARTER
) -> pd.DataFrame:
    """Three-month forward rates starting every `step_years` out to the horizon."""
    steps = horizon_years / step_years
    if abs(round(steps) - steps) > 1e-9:
        raise ModelError(
            f"step {step_years} must divide horizon {horizon_years} a whole number of times"
        )

    starts = np.arange(round(steps)) * step_years
    rows = []
    for start in starts:
        begin = max(float(start), 1e-6)
        continuous = curve.forward(begin, begin + QUARTER)
        rows.append(
            {
                "start": float(start),
                "forward_continuous": continuous,
                # Policy rates are quoted as simple annualised rates, so the chart converts
                # once, here, rather than in the plotting code where nobody would find it.
                "forward_quoted": (np.exp(continuous * QUARTER) - 1) / QUARTER,
            }
        )
    return pd.DataFrame(rows)


def implied_change_bp(curve: Curve, horizon_years: float = 1.0) -> float:
    """Change in the three-month forward rate between today and the horizon, in bp."""
    path = implied_path(curve, horizon_years=horizon_years + QUARTER, step_years=QUARTER)
    first = path.iloc[0]["forward_quoted"]
    last = path.iloc[-1]["forward_quoted"]
    return float((last - first) * 1e4)

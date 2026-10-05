"""The policy path the curve prices.

A sequence of three-month forward rates read off the curve is the market's implied path for
the short rate. It is not a forecast and it is not, on its own, an expectation either —
rlab.termpremium is what separates the two. This module only reads the curve.

How smooth the path looks is decided by the curve's interpolation, not here. On a curve
linear in zero rates the forwards jump at every published pillar and rise in identical
steps between them, which reads as a policy path and is only the shape of the interpolant;
`rlab.curve.bootstrap` builds a monotone cubic on log discount factors so that it is not.
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

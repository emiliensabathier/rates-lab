"""Fitting a Svensson curve to observed rates.

The two decay parameters are swept on a grid; at each grid point the four betas solve a
plain linear least squares problem. Optimising all six jointly is the textbook approach and
it is also where Svensson fits are known to land in local minima, which makes the
parameters jump between neighbouring days and stop meaning anything. Sweeping the two
nonlinear parameters costs a few hundred linear solves and removes the failure mode.
"""

from __future__ import annotations

import numpy as np

from rlab.curve.svensson import Compounding, SvenssonCurve, SvenssonParams, _loading
from rlab.errors import ModelError

# Decay grids in years. The lower bound keeps the short-end loading from collapsing onto a
# single pillar; the upper bound spans the thirty-year point.
TAU1_GRID = np.exp(np.linspace(np.log(0.1), np.log(5.0), 60))
TAU2_GRID = np.exp(np.linspace(np.log(2.0), np.log(30.0), 60))


def _design(maturities: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    slope1, curve1 = _loading(maturities, tau1)
    _, curve2 = _loading(maturities, tau2)
    return np.column_stack([np.ones_like(maturities), slope1, curve1, curve2])


def fit_svensson(
    maturities: np.ndarray,
    rates: np.ndarray,
    compounding: Compounding,
    max_error_bp: float = 5.0,
) -> SvenssonCurve:
    """Fit a Svensson curve to `rates` (decimals) observed at `maturities` (years)."""
    maturities = np.asarray(maturities, dtype=float)
    rates = np.asarray(rates, dtype=float)
    if maturities.shape != rates.shape:
        raise ModelError("maturities and rates must have the same length")
    if maturities.size < 6:
        raise ModelError(f"a Svensson fit needs at least 6 pillars, got {maturities.size}")

    target = rates * 100
    best: tuple[float, np.ndarray, float, float] | None = None
    for tau1 in TAU1_GRID:
        for tau2 in TAU2_GRID:
            if tau2 <= tau1 * 1.05:
                # Nearly equal decays make the two curvature loadings collinear, so the betas
                # stop being identified. Skipping that region is what keeps the fit stable
                # from one day to the next.
                continue
            design = _design(maturities, tau1, tau2)
            betas, *_ = np.linalg.lstsq(design, target, rcond=None)
            residual = float(np.sum((design @ betas - target) ** 2))
            if best is None or residual < best[0]:
                best = (residual, betas, tau1, tau2)

    if best is None:
        raise ModelError("no admissible decay pair on the Svensson grid")

    _, betas, tau1, tau2 = best
    params = SvenssonParams(
        beta0=float(betas[0]),
        beta1=float(betas[1]),
        beta2=float(betas[2]),
        beta3=float(betas[3]),
        tau1=float(tau1),
        tau2=float(tau2),
    )
    curve = SvenssonCurve(params, compounding=compounding)

    worst_bp = float(np.max(np.abs(curve.rate(maturities) - rates)) * 1e4)
    if worst_bp > max_error_bp:
        raise ModelError(
            f"Svensson fit is off by {worst_bp:.2f} basis points at its worst pillar, "
            f"above the {max_error_bp:.2f} limit"
        )
    return curve

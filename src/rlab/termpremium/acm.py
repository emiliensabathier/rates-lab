"""Adrian, Crump and Moench (2013): decomposing a yield into expectations and a premium.

The point of the module, in one sentence: a forward rate is not an expectation, and saying
"the curve prices three cuts" without splitting the two is the claim this project exists to
avoid making.

Four steps, three of which are plain regressions:
  1. principal components of the yield panel give the pricing factors;
  2. a VAR(1) on those factors gives their innovations;
  3. regressing excess holding-period returns on the innovations and the lagged factors
     gives the exposures and, through them, the price of risk;
  4. the affine recursion turns factors into log bond prices, run twice — once under the
     estimated risk-neutral dynamics, once with the price of risk switched off.

The gap between the two runs is the term premium. That subtraction is an identity, not an
estimate, which is what the first test pins down.

Two conventions worth stating, because both are places an implementation silently goes
wrong. First, the price of risk enters the risk-neutral dynamics directly, `mu - lambda0`
and `phi - lambda1`, not scaled by the innovation covariance: the regression already
returns lambda in those units, and multiplying by sigma a second time shrinks the whole
premium by roughly five orders of magnitude, leaving a decomposition that always prints
zero. Second, `fitted` is the curve priced under the risk-neutral dynamics — the one that
reproduces the observed panel — while `risk_neutral` is the counterfactual with the price
of risk set to zero, which is the expectations component and does *not* track the data.

The estimator wants a monthly maturity grid, because a one-month holding period turns an
n-month bond into an (n-1)-month bond and that leg has to be on the grid rather than
interpolated. `rlab.curve.fit` produces exactly such a grid from a handful of pillars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlab.curve.pca import decompose
from rlab.errors import ModelError

# Ten years of monthly data. A five-factor VAR estimated on less is a number with no
# information in it, and it would still print.
MIN_OBSERVATIONS = 120

MONTHS_PER_YEAR = 12


@dataclass(frozen=True)
class ACMResult:
    """Yields are annualised and continuously compounded, shaped (months, maturities)."""

    fitted: np.ndarray
    risk_neutral: np.ndarray
    term_premium: np.ndarray
    maturities_months: np.ndarray
    lambda0: np.ndarray
    lambda1: np.ndarray


def _ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares with an explicit pseudo-inverse, mapping regressors to targets."""
    return np.linalg.pinv(x) @ y


def _validate(panel: np.ndarray, maturities: np.ndarray, n_factors: int) -> None:
    if panel.ndim != 2 or panel.shape[1] != maturities.size:
        raise ModelError(f"panel shape {panel.shape} does not match {maturities.size} maturities")
    if np.any(~np.isfinite(panel)):
        raise ModelError("yield panel contains missing or non-finite values")
    if panel.shape[0] < MIN_OBSERVATIONS:
        raise ModelError(
            f"ACM needs at least {MIN_OBSERVATIONS} monthly observations, got {panel.shape[0]}"
        )
    if n_factors > maturities.size:
        raise ModelError(f"cannot extract {n_factors} factors from {maturities.size} maturities")
    expected = np.arange(1, maturities.size + 1, dtype=float)
    if not np.array_equal(maturities, expected):
        raise ModelError(
            "ACM needs a monthly maturity grid 1, 2, ... months, because the one-month "
            "holding period leaves an (n-1)-month bond that has to be on the grid; "
            "sample a fitted curve with rlab.curve.fit to build one"
        )


def _price_of_risk(
    panel: np.ndarray,
    maturities: np.ndarray,
    factors: np.ndarray,
    innovations: np.ndarray,
    sigma: np.ndarray,
    n_factors: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Regress excess holding-period returns to get lambda0, lambda1 and the residual scale."""
    months = panel.shape[0]
    log_prices = -panel * maturities / MONTHS_PER_YEAR
    # The one-month bond is the riskless leg over a one-month holding period. It is the
    # first column of the grid, and using any longer maturity here inflates the leg in
    # proportion and drags the estimated price of risk with it.
    short_rate = panel[:, 0] / MONTHS_PER_YEAR

    # Holding an n-month bond for one month leaves an (n-1)-month bond, which on a monthly
    # grid is the neighbouring column. Columns run n = 2 ... N.
    excess = log_prices[1:, :-1] - log_prices[:-1, 1:] - short_rate[:-1, None]

    regressors = np.column_stack([np.ones(months - 1), innovations, factors[:-1]])
    loadings = _ols(regressors, excess)
    intercept = loadings[0]                           # (N-1,)
    beta = loadings[1 : 1 + n_factors]                # (K, N-1)
    exposure = loadings[1 + n_factors :]              # (K, N-1)
    residuals = excess - regressors @ loadings
    residual_variance = np.mean(residuals**2, axis=0)  # (N-1,)

    # Jensen term: the expected excess return carries half the variance of the return
    # itself, so it has to come back out before the intercept is read as compensation.
    convexity = np.array([float(b @ sigma @ b) for b in beta.T])
    gram = np.linalg.pinv(beta @ beta.T)
    lambda0 = gram @ beta @ (intercept + 0.5 * (convexity + residual_variance))
    lambda1 = gram @ beta @ exposure.T
    return lambda0, lambda1, float(np.mean(residual_variance))


def _yields_from_recursion(
    delta0: float,
    delta1: np.ndarray,
    sigma: np.ndarray,
    residual_variance: float,
    drift: np.ndarray,
    transition: np.ndarray,
    factors: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """Price every maturity by the affine recursion, then annualise."""
    horizon = maturities.size
    a_n, b_n = -delta0, -delta1
    a_all = np.empty(horizon)
    b_all = np.empty((horizon, b_n.size))
    a_all[0], b_all[0] = a_n, b_n
    for n in range(1, horizon):
        a_next = a_n + b_n @ drift + 0.5 * (b_n @ sigma @ b_n + residual_variance) - delta0
        b_next = transition.T @ b_n - delta1
        a_n, b_n = a_next, b_next
        a_all[n], b_all[n] = a_n, b_n

    log_prices = a_all[None, :] + factors @ b_all.T
    return -log_prices / maturities[None, :] * MONTHS_PER_YEAR


def estimate(
    yields: np.ndarray, maturities_months: np.ndarray, n_factors: int = 5
) -> ACMResult:
    """Decompose a monthly panel of continuously compounded zero yields."""
    panel = np.asarray(yields, dtype=float)
    maturities = np.asarray(maturities_months, dtype=float)
    _validate(panel, maturities, n_factors)

    months = panel.shape[0]

    # Step 1 — pricing factors.
    factors = decompose(panel, n_components=n_factors).scores          # (months, K)

    # Step 2 — VAR(1) on the factors.
    lagged = np.column_stack([np.ones(months - 1), factors[:-1]])
    coefficients = _ols(lagged, factors[1:])                           # (1+K, K)
    mu = coefficients[0]                                               # (K,)
    phi = coefficients[1:].T                                           # (K, K)
    innovations = factors[1:] - lagged @ coefficients                  # (months-1, K)
    sigma = innovations.T @ innovations / innovations.shape[0]

    # Step 3 — the price of risk, from excess holding-period returns.
    lambda0, lambda1, residual_variance = _price_of_risk(
        panel, maturities, factors, innovations, sigma, n_factors
    )

    # Step 4 — the short rate as an affine function of the factors, then the recursion,
    # run once under the risk-neutral dynamics and once with the price of risk removed.
    short_loadings = _ols(
        np.column_stack([np.ones(months), factors]), panel[:, 0] / MONTHS_PER_YEAR
    )
    delta0, delta1 = float(short_loadings[0]), short_loadings[1:]

    def price(drift: np.ndarray, transition: np.ndarray) -> np.ndarray:
        return _yields_from_recursion(
            delta0, delta1, sigma, residual_variance, drift, transition, factors, maturities
        )

    fitted = price(mu - lambda0, phi - lambda1)
    risk_neutral = price(mu, phi)

    return ACMResult(
        fitted=fitted,
        risk_neutral=risk_neutral,
        term_premium=fitted - risk_neutral,
        maturities_months=maturities,
        lambda0=lambda0,
        lambda1=lambda1,
    )

"""From eleven published quotes to a term premium, in one pass.

The chain is the whole point of the package, so it lives in one file where a reader can
see every step and every place a step is allowed to refuse:

    CMT par yields -> bootstrap -> monthly zero grid -> PCA -> ACM -> term premium

Two choices decide what the output means, and both are here rather than buried.

The maturity grid stops at ten years. It has to: ACM prices a monthly grid out to its
longest maturity, and the twenty- and thirty-year CMT series carry the 2002-2006 gap when
the thirty-year bond was not issued. Requiring them would delete four years of history to
buy maturities the decomposition does not use.

The monthly grid is sampled from the bootstrapped curve, which is linear in zero space
between the nine published pillars. The cross-section therefore carries nine independent
points dressed as a hundred and twenty, and the ACM fit should be read with that in mind:
it is fitting an interpolation, not a hundred and twenty separate quotes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rlab.curve.base import Curve
from rlab.curve.bootstrap import bootstrap
from rlab.curve.pca import PCAResult, decompose
from rlab.data import fred
from rlab.errors import DataError, ModelError
from rlab.inflation import breakeven, forward_breakeven, require_real_coverage
from rlab.policy import implied_path
from rlab.termpremium.acm import ACMResult, estimate

# Nine nominal pillars, out to the ten years the ACM grid ends at.
NOMINAL_PILLARS: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
}

# TIPS constant maturities, for the breakeven. Thirty years is loaded but not required:
# it starts in 2010 and the breakevens published here stop at ten.
REAL_PILLARS: dict[str, float] = {"DFII5": 5.0, "DFII10": 10.0}

# The Federal Reserve Board's Kim-Wright ten-year term premium. A different model fitted by
# a different method, which is exactly why it is worth comparing against: agreement in the
# dynamics is evidence, agreement in the level would only mean the two share an assumption.
BENCHMARK_SERIES = "THREEFYTP10"

MATURITY_MONTHS = np.arange(1, 121, dtype=float)
MONTHS_PER_YEAR = 12
TEN_YEARS = 10.0
FIVE_YEARS = 5.0
N_FACTORS = 5
N_COMPONENTS = 3


@dataclass(frozen=True)
class Result:
    """Everything the report shows, and nothing it computes itself."""

    dates: pd.DatetimeIndex
    panel: np.ndarray
    acm: ACMResult
    pca: PCAResult
    benchmark: pd.Series
    latest_curve: Curve
    policy_path: pd.DataFrame
    breakeven_spot_10y: float
    breakeven_forward_5y5y: float
    refusals: dict[str, str]

    @property
    def term_premium_10y(self) -> pd.Series:
        return pd.Series(self.acm.term_premium[:, -1], index=self.dates, name="term_premium")

    @property
    def expectations_10y(self) -> pd.Series:
        return pd.Series(self.acm.risk_neutral[:, -1], index=self.dates, name="expectations")

    @property
    def fitted_10y(self) -> pd.Series:
        return pd.Series(self.acm.fitted[:, -1], index=self.dates, name="fitted")

    @property
    def observed_10y(self) -> pd.Series:
        return pd.Series(self.panel[:, -1], index=self.dates, name="observed")

    @property
    def fit_error_bp(self) -> np.ndarray:
        return np.abs(self.acm.fitted - self.panel) * 1e4


def month_end_quotes(daily: pd.DataFrame, pillars: dict[str, float]) -> pd.DataFrame:
    """The last day of each month on which every pillar printed, as decimals."""
    missing = [series for series in pillars if series not in daily.columns]
    if missing:
        raise DataError(f"missing series {missing} in the loaded frame")
    complete = daily[list(pillars)].dropna()
    if complete.empty:
        raise DataError("no date has a quote for every pillar")
    return complete.resample("ME").last().dropna() / 100.0


def build_panel(quotes: pd.DataFrame, pillars: dict[str, float]) -> tuple[
    pd.DatetimeIndex, np.ndarray, dict[str, str]
]:
    """Bootstrap each month's curve and sample it on the monthly maturity grid.

    A month whose quotes do not admit a positive, decreasing discount curve is dropped and
    the reason recorded. That happens: at the end of 2008 and again in 2011 the front of
    the bill curve implies a negative forward rate, and a curve builder that smoothed it
    away would be inventing a quote nobody published.
    """
    maturities = np.array(list(pillars.values()), dtype=float)
    rows: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    refusals: dict[str, str] = {}

    for date, row in quotes.iterrows():
        try:
            curve = bootstrap(maturities, row.to_numpy())
        except ModelError as error:
            refusals[date.date().isoformat()] = str(error)
            continue
        rows.append(np.asarray(curve.zero(MATURITY_MONTHS / MONTHS_PER_YEAR)))
        dates.append(date)

    if not rows:
        raise DataError("every month refused to bootstrap; there is no panel to estimate on")
    return pd.DatetimeIndex(dates), np.array(rows), refusals


def _real_curve(daily: pd.DataFrame, as_of: pd.Timestamp) -> Curve:
    """Bootstrap the TIPS curve as of one date, from the two published pillars used here.

    The first pillar is five years, so everything shorter is flat at the five-year real
    zero. That is an approximation, and it is harmless for the two numbers taken off this
    curve -- both are read at five years or beyond.
    """
    quotes = daily[list(REAL_PILLARS)].dropna()
    available = quotes.loc[:as_of]
    if available.empty:
        raise DataError(f"no TIPS quote at or before {as_of.date()}")
    latest = available.iloc[-1]
    require_real_coverage(pd.Timestamp(available.index[-1]), TEN_YEARS)
    return bootstrap(np.array(list(REAL_PILLARS.values()), dtype=float), latest.to_numpy() / 100.0)


def run(
    cache_dir: Path,
    refresh: bool = False,
    fetcher: fred.Fetcher = fred.http_fetcher,
) -> Result:
    """Load, bootstrap, decompose, estimate. Every figure the report prints starts here."""
    series = [*NOMINAL_PILLARS, *REAL_PILLARS, BENCHMARK_SERIES]
    daily = fred.load(series, cache_dir=cache_dir, fetcher=fetcher, refresh=refresh)

    quotes = month_end_quotes(daily, NOMINAL_PILLARS)
    dates, panel, refusals = build_panel(quotes, NOMINAL_PILLARS)

    pca = decompose(panel, n_components=N_COMPONENTS)
    acm = estimate(panel, MATURITY_MONTHS, n_factors=N_FACTORS)

    benchmark = (
        daily[BENCHMARK_SERIES].resample("ME").last().reindex(dates).astype(float) / 100.0
    )
    benchmark.name = "kim_wright"

    as_of = dates[-1]
    latest_curve = bootstrap(
        np.array(list(NOMINAL_PILLARS.values()), dtype=float), quotes.loc[as_of].to_numpy()
    )
    real_curve = _real_curve(daily, as_of)

    return Result(
        dates=dates,
        panel=panel,
        acm=acm,
        pca=pca,
        benchmark=benchmark,
        latest_curve=latest_curve,
        policy_path=implied_path(latest_curve, horizon_years=2.0),
        breakeven_spot_10y=breakeven(latest_curve, real_curve, TEN_YEARS),
        breakeven_forward_5y5y=forward_breakeven(
            latest_curve, real_curve, FIVE_YEARS, TEN_YEARS
        ),
        refusals=refusals,
    )


def agreement(result: Result) -> dict[str, float]:
    """How the estimate compares with the published Kim-Wright series.

    Three numbers, and the order matters: the correlations are the claim, the mean gap is
    the disclaimer. Reporting the first without the second would be picking the flattering
    half of a comparison this package went looking for.
    """
    joined = pd.concat([result.term_premium_10y, result.benchmark], axis=1).dropna()
    changes = joined.diff().dropna()
    return {
        "correlation_levels": float(joined.iloc[:, 0].corr(joined.iloc[:, 1])),
        "correlation_changes": float(changes.iloc[:, 0].corr(changes.iloc[:, 1])),
        "mean_gap": float((joined.iloc[:, 0] - joined.iloc[:, 1]).mean()),
        "observations": float(len(joined)),
    }

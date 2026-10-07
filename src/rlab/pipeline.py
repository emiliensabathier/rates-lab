"""From eleven published quotes to a term premium, in one pass.

The chain is the whole point of the package, so it lives in one file where a reader can
see every step and every place a step is allowed to refuse:

    CMT par yields -> bootstrap -> monthly zero grid -> PCA -> ACM -> term premium

Two choices decide what the output means, and both are here rather than buried.

The maturity grid stops at ten years. It has to: ACM prices a monthly grid out to its
longest maturity, and the twenty- and thirty-year CMT series carry the 2002-2006 gap when
the thirty-year bond was not issued. Requiring them would delete four years of history to
buy maturities the decomposition does not use.

The monthly grid is sampled from the bootstrapped curve, a monotone cubic on log discount
factors through the nine published pillars. The cross-section therefore carries nine
independent points dressed as a hundred and twenty, and the ACM fit should be read with
that in mind: it is fitting an interpolation, not a hundred and twenty separate quotes.

Each month is dated by the day its last complete set of quotes printed, not by the calendar
month-end, and a final month that has not finished trading at capture time is dropped:
labelling a mid-month curve with the 31st would put a date on the page that had not yet
happened when the data was pulled, and would make the last VAR step half a month long.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rlab.curve.base import Curve
from rlab.curve.bootstrap import bootstrap
from rlab.curve.pca import PCAResult, decompose
from rlab.data import fred, spf
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
    consecutive: np.ndarray
    panel: np.ndarray
    acm: ACMResult
    pca: PCAResult
    benchmark: pd.Series
    survey: pd.Series
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
        return pd.Series(self.acm.expectations[:, -1], index=self.dates, name="expectations")

    @property
    def fitted_10y(self) -> pd.Series:
        return pd.Series(self.acm.fitted[:, -1], index=self.dates, name="fitted")

    @property
    def observed_10y(self) -> pd.Series:
        return pd.Series(self.panel[:, -1], index=self.dates, name="observed")

    @property
    def fit_error_bp(self) -> np.ndarray:
        return np.abs(self.acm.fitted - self.panel) * 1e4

    @property
    def average_short_rate(self) -> float:
        """Sample mean of the one-month zero rate, the level ACM expectations revert to."""
        return float(self.panel[:, 0].mean())


def month_end_quotes(daily: pd.DataFrame, pillars: dict[str, float]) -> pd.DataFrame:
    """The last day of each month on which every pillar printed, as decimals.

    Indexed by that day, not by the calendar month-end. The final month is dropped when its
    last quote is earlier than the month's last business day, because then the month had not
    finished when the data was captured. A final month whose last business day was a market
    holiday would be dropped too; that costs one month at the very end, never a wrong date.
    """
    missing = [series for series in pillars if series not in daily.columns]
    if missing:
        raise DataError(f"missing series {missing} in the loaded frame")
    complete = daily[list(pillars)].dropna()
    if complete.empty:
        raise DataError("no date has a quote for every pillar")
    last_per_month = complete.groupby(complete.index.to_period("M")).tail(1)
    final = last_per_month.index[-1]
    if final < final + pd.offsets.BMonthEnd(0):
        last_per_month = last_per_month.iloc[:-1]
    if last_per_month.empty:
        raise DataError("no month has finished trading in the loaded frame")
    return last_per_month / 100.0


def consecutive_months(dates: pd.DatetimeIndex) -> np.ndarray:
    """For each adjacent pair of dates, whether the second is in the month after the first."""
    ordinals = np.array([period.ordinal for period in dates.to_period("M")])
    return np.diff(ordinals) == 1


def build_panel(
    quotes: pd.DataFrame, pillars: dict[str, float]
) -> tuple[pd.DatetimeIndex, np.ndarray, dict[str, str]]:
    """Bootstrap each month's curve and sample it on the monthly maturity grid.

    A month whose quotes imply a negative forward rate, so that no positive, non-increasing
    discount curve reprices them, is dropped and the reason recorded. On the frozen capture
    no month is: the seven months an earlier, linearly interpolated version refused were
    artefacts of that interpolation (and, once, of treating a zero forward as negative). The
    check stays, because a curve builder that smoothed a genuine one away would be inventing
    a quote nobody published, and `consecutive_months` keeps the VAR honest if it fires.
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

    The first pillar is five years, so everything shorter is interpolated from zero at
    maturity zero and means nothing. That is harmless for the two numbers taken off this
    curve: the spot is read at ten years and the 5y5y forward between the two pillars, and
    both depend only on the pillar discount factors.
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
    survey_fetcher: spf.Fetcher = spf.http_fetcher,
) -> Result:
    """Load, bootstrap, decompose, estimate. Every figure the report prints starts here."""
    series = [*NOMINAL_PILLARS, *REAL_PILLARS, BENCHMARK_SERIES]
    daily = fred.load(series, cache_dir=cache_dir, fetcher=fetcher, refresh=refresh)

    quotes = month_end_quotes(daily, NOMINAL_PILLARS)
    dates, panel, refusals = build_panel(quotes, NOMINAL_PILLARS)

    consecutive = consecutive_months(dates)

    # The variance decomposition is of monthly changes, the standard way to ask what moves
    # the curve; on levels the first component soaks up the persistence of rates and
    # overstates how much of the movement it explains. ACM, below, takes its pricing
    # factors from levels, as the method specifies.
    pca = decompose(np.diff(panel, axis=0)[consecutive], n_components=N_COMPONENTS)
    acm = estimate(panel, MATURITY_MONTHS, n_factors=N_FACTORS, consecutive=consecutive)

    # Same-day comparison: Kim-Wright on the day the curve was built, or not at all.
    benchmark = daily[BENCHMARK_SERIES].reindex(dates).astype(float) / 100.0
    benchmark.name = "kim_wright"
    survey = spf.load(cache_dir, fetcher=survey_fetcher, refresh=refresh)

    as_of = dates[-1]
    latest_curve = bootstrap(
        np.array(list(NOMINAL_PILLARS.values()), dtype=float), quotes.loc[as_of].to_numpy()
    )
    real_curve = _real_curve(daily, as_of)

    return Result(
        dates=dates,
        consecutive=consecutive,
        panel=panel,
        acm=acm,
        pca=pca,
        benchmark=benchmark,
        survey=survey,
        latest_curve=latest_curve,
        policy_path=implied_path(latest_curve, horizon_years=2.0),
        breakeven_spot_10y=breakeven(latest_curve, real_curve, TEN_YEARS),
        breakeven_forward_5y5y=forward_breakeven(latest_curve, real_curve, FIVE_YEARS, TEN_YEARS),
        refusals=refusals,
    )


def _one_month_changes(frame: pd.DataFrame) -> pd.DataFrame:
    """Month-on-month changes, keeping only pairs that are genuinely one month apart."""
    steps = consecutive_months(pd.DatetimeIndex(frame.index))
    return frame.diff().iloc[1:][steps]


def agreement(result: Result) -> dict[str, float]:
    """How the estimate compares with the published Kim-Wright series.

    Each correlation comes with a naive benchmark: the same correlation computed with the
    observed ten-year yield in place of the model's premium. Kim-Wright attributes most of a
    monthly yield move to the premium, so the yield alone tracks its changes closely, and a
    model correlation is only evidence of anything to the extent it beats that. The mean gap
    is the disclaimer on the level.
    """
    joined = pd.concat(
        [result.term_premium_10y, result.benchmark, result.observed_10y], axis=1
    ).dropna()
    changes = _one_month_changes(joined)
    model, published, observed = "term_premium", "kim_wright", "observed"
    return {
        "correlation_levels": float(joined[model].corr(joined[published])),
        "correlation_changes": float(changes[model].corr(changes[published])),
        "naive_correlation_levels": float(joined[observed].corr(joined[published])),
        "naive_correlation_changes": float(changes[observed].corr(changes[published])),
        "mean_gap": float((joined[model] - joined[published]).mean()),
        "latest_gap": float(joined[model].iloc[-1] - joined[published].iloc[-1]),
        "latest_kim_wright": float(joined[published].iloc[-1]),
        "observations": float(len(joined)),
    }


def survey_anchor(result: Result) -> dict[str, float]:
    """The level gap against Kim-Wright, with the model's expectations swapped for a survey's.

    Each first-quarter Survey of Professional Forecasters asks for the three-month bill rate
    averaged over the next ten years (``BILL10``). Its responses are due in the middle month
    of the quarter, so each survey is set against that month's curve. Ten-year yield minus
    the survey forecast is a survey-anchored premium, the decomposition Kim-Wright's survey
    anchor pushes towards. Three approximations, each small next to a percentage point: the
    survey forecasts a three-month bill where ACM averages the one-month zero rate, the bill
    is quoted on a discount basis, and the survey premium carries convexity that ACM's
    expectations exclude.
    """
    frame = pd.concat(
        [result.expectations_10y, result.observed_10y, result.term_premium_10y, result.benchmark],
        axis=1,
    )
    frame = frame[frame.index.month % 3 == 2]
    frame = frame.assign(survey=frame.index.to_period("Q").map(result.survey)).dropna()
    if frame.empty:
        raise DataError("no survey date falls inside the panel")
    survey_premium = frame["observed"] - frame["survey"]
    return {
        "surveys": float(len(frame)),
        "first_survey": float(frame.index[0].year),
        "expectations_minus_survey": float((frame["expectations"] - frame["survey"]).mean()),
        "model_gap": float((frame["term_premium"] - frame["kim_wright"]).mean()),
        "survey_gap": float((survey_premium - frame["kim_wright"]).mean()),
        "survey_correlation_levels": float(survey_premium.corr(frame["kim_wright"])),
        "latest_survey": float(frame["survey"].iloc[-1]),
        "latest_survey_expectations": float(frame["expectations"].iloc[-1]),
    }

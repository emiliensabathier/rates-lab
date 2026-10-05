"""Assembly of the final HTML report.

Computes nothing: every number shown here is produced by the modules under `rlab`.
"""

from __future__ import annotations

import html as html_escape

from rlab.pipeline import MATURITY_MONTHS, N_FACTORS, Result, agreement
from rlab.report.charts import (
    benchmark_chart,
    decomposition_chart,
    fit_chart,
    loadings_chart,
    policy_chart,
)

STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
       max-width: 980px; padding: 2rem 1.25rem; color: #16181d; line-height: 1.5; }
h1 { font-size: 1.9rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.25rem; margin-top: 2.5rem; border-bottom: 1px solid #e3e5ea;
     padding-bottom: 0.35rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 0.45rem 0.6rem; border-bottom: 1px solid #eceef2; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid #c9ccd4; }
.note { color: #5b6070; font-size: 0.9rem; }
.caveat { background: #f6f7f9; border-left: 3px solid #c9ccd4; padding: 0.75rem 1rem;
          margin: 1rem 0; }
.headline { font-size: 1.6rem; margin: 0.5rem 0; }
.decomposition { font-size: 1.3rem; font-variant-numeric: tabular-nums; margin: 0.5rem 0; }
svg { max-width: 100%; height: auto; }
"""


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pp(value: float) -> str:
    """A difference between two rates, in percentage points rather than percent."""
    return f"{value * 100:+.2f} pp"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html_escape.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html_escape.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _lead(stats: dict[str, float]) -> str:
    """What to trust, before any number that should not be trusted."""
    return f"""
<p class="headline">The shape of the term premium, not its level</p>
<p><strong>What this measures well:</strong> how the ten-year term premium moves over the
years. Its correlation with the Federal Reserve Board's Kim-Wright estimate is
{stats["correlation_levels"]:.3f} in levels, against {stats["naive_correlation_levels"]:.3f}
for the ten-year yield itself, so the decomposition adds something the yield alone does not
carry.</p>
<p><strong>What it does not:</strong> month-to-month changes, and the level. In monthly
changes the ten-year yield alone tracks Kim-Wright better
({stats["naive_correlation_changes"]:.3f}) than this premium does
({stats["correlation_changes"]:.3f}). The level sits {stats["mean_gap"] * 100:.2f} points above
Kim-Wright on average and {stats["latest_gap"] * 100:.2f} points above it on the latest date.</p>
"""


def _decomposition(result: Result, stats: dict[str, float]) -> str:
    observed = float(result.observed_10y.iloc[-1])
    expectations = float(result.expectations_10y.iloc[-1])
    premium = float(result.term_premium_10y.iloc[-1])
    as_of = result.dates[-1].date().isoformat()
    return f"""
<p class="decomposition">{_pct(observed)} = {_pct(expectations)} expected
+ {_pct(premium)} premium</p>
<p class="note">Ten-year zero rate on {as_of}, split by the ACM decomposition: what the
curve expects the short rate to average over ten years, and what it charges on top for
holding the duration. Kim-Wright puts the premium at {_pct(stats["latest_kim_wright"])} on
the same day. Read the split as indicative: the two models agree the premium is positive and
disagree by {stats["latest_gap"] * 100:.2f} points on how large it is.</p>
"""


def _validation(stats: dict[str, float]) -> str:
    rows = [
        [
            "Correlation with Kim-Wright, levels",
            f"{stats['correlation_levels']:.3f}",
            f"{stats['naive_correlation_levels']:.3f}",
        ],
        [
            "Correlation with Kim-Wright, monthly changes",
            f"{stats['correlation_changes']:.3f}",
            f"{stats['naive_correlation_changes']:.3f}",
        ],
        ["Mean gap, this estimate minus Kim-Wright", _pp(stats["mean_gap"]), ""],
        ["Gap on the latest date", _pp(stats["latest_gap"]), ""],
        ["Overlapping months", f"{int(stats['observations'])}", ""],
    ]
    return _table(["Against the Kim-Wright series", "ACM premium", "Ten-year yield"], rows)


def _fit_summary(result: Result) -> str:
    errors = result.fit_error_bp
    rows = [
        ["Worst maturity, worst month", f"{errors.max():.1f} bp"],
        ["Mean across the whole panel", f"{errors.mean():.2f} bp"],
        ["Worst maturity, latest month", f"{errors[-1].max():.1f} bp"],
    ]
    return _table(["Cross-sectional fit", "Value"], rows)


def _pca_table(result: Result) -> str:
    names = ("Level", "Slope", "Curvature")
    rows = [[name, _pct(share)] for name, share in zip(names, result.pca.explained, strict=False)]
    rows.append(["Together", _pct(float(result.pca.explained.sum()))])
    return _table(["Principal component of monthly changes", "Share of variance"], rows)


def _policy_table(result: Result) -> str:
    rows = [
        [f"{row['start']:.2f}y forward", _pct(row["forward_quoted"])]
        for _, row in result.policy_path.iterrows()
    ]
    return _table(["Three-month rate starting", "Priced"], rows)


def _breakeven_table(result: Result) -> str:
    rows = [
        ["Ten-year spot breakeven", _pct(result.breakeven_spot_10y)],
        ["Five-year, five-year forward breakeven", _pct(result.breakeven_forward_5y5y)],
    ]
    return _table(["Measure", "Value"], rows)


def _refusals(result: Result) -> str:
    if not result.refusals:
        return "<p class='note'>No month was refused.</p>"
    rows = [[date, reason] for date, reason in sorted(result.refusals.items())]
    return _table(["Month", "Why the curve was not built"], rows)


def render(result: Result, generated_on: str) -> str:
    """Return the whole self-contained page."""
    first = result.dates[0].date().isoformat()
    last = result.dates[-1].date().isoformat()
    stats = agreement(result)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>rates-lab: what the yield curve prices</title>
<style>{STYLE}</style></head><body>

<h1>What the yield curve prices</h1>
<p class="note">US Treasury constant-maturity quotes, {first} to {last},
{len(result.dates)} month-end observations, each dated by the day its quotes printed.
Captured {generated_on}; the capture month had not finished and is not used.</p>

{_lead(stats)}

<h2>The latest decomposition</h2>
{_decomposition(result, stats)}

<div class="caveat">
A forward rate is not an expectation. Both this model and Kim-Wright put a positive premium
in the ten-year yield, so quoting the yield as the market's view of the average short rate
quotes some premium as if it were a forecast. How much premium is where the two disagree.
</div>

{decomposition_chart(result.observed_10y, result.expectations_10y)}

<h2>Checked against a published estimate</h2>
<p class="note">The Federal Reserve Board publishes a ten-year term premium from the
Kim-Wright model: a different specification, fitted by Kalman filter rather than by
regression, and anchored to survey forecasts of the short rate. It is compared on the same
day as each curve. Next to each correlation is the naive benchmark: the same correlation with
the observed ten-year yield in place of the model's premium.</p>

{_validation(stats)}

<div class="caveat">
<strong>Levels: the model adds something. Changes: it does not.</strong>
In levels this premium tracks Kim-Wright more closely than the yield does, which is
evidence that the decomposition picks up the slow movements it is supposed to: compression
through the QE decade, reopening after 2022. In monthly changes the yield alone does better.
Kim-Wright puts most of any monthly yield move into its premium, so a high correlation in
changes is mostly the yield, and a model correlation below the yield's own is no evidence
that the model separates premium from expectations month by month.
</div>

<div class="caveat">
<strong>The level is not the claim.</strong>
This estimate runs above Kim-Wright by the mean gap in the table, and the gap is not
constant: it nearly closes in the middle of the sample and is widest at both ends. The
estimator's sampling error does not account for it in any direction: on synthetic panels of
this length that error is wide, about 0.7 points either way, but centred slightly below zero
rather than above (the test suite measures it), so it is a band around the estimate, not an
explanation of the gap. The most likely source is the difference between the models.
Kim-Wright pins its expected path to survey forecasts; ACM has no anchor but the sample, so
its expected path reverts toward the sample's average one-month rate of
{_pct(result.average_short_rate)}. Where surveys expected a different path from that
reversion, the two models split the same yield differently. That is a hypothesis, not tested
here.
</div>

{benchmark_chart(result.term_premium_10y, result.benchmark)}

<h2>Does the model price the curve it was fitted to</h2>
{_fit_summary(result)}
<p class="note">The monthly grid is sampled from a curve interpolated between nine
published pillars, so the cross-section carries nine independent points rather than a
hundred and twenty. The fit below should be read as fitting an interpolation.</p>

{fit_chart(MATURITY_MONTHS, result.panel[-1], result.acm.fitted[-1])}

<h2>What moves the curve</h2>
{_pca_table(result)}
{loadings_chart(MATURITY_MONTHS, result.pca.loadings)}
<p class="note">Principal components of month-on-month changes in the zero curve. The ACM
step uses {N_FACTORS} components of the yield levels as pricing factors, as the method
specifies; the three shown here are the ones with a name and a shape.</p>

<h2>The policy path priced today</h2>
{_policy_table(result)}
{policy_chart(result.policy_path)}
<p class="note">Three-month forward rates read off the latest curve, quoted as simple
annualised rates. This is what the curve prices, not what anyone forecasts, and the premium
above is exactly the reason those are different statements. Between the published pillars
(one, three and six months, one and two years) the path is a property of the interpolation,
a monotone cubic on log discount factors: smooth rather than stepped, but not data.</p>

<h2>Breakeven inflation</h2>
{_breakeven_table(result)}

<div class="caveat">
A breakeven is expected inflation, plus an inflation risk premium, minus a TIPS liquidity
premium. Nothing public separates the three. The figures above are what the market charges,
not what the market expects.
</div>

<h2>Months this page refused to use</h2>
{_refusals(result)}
<p class="note">A month is dropped when its published quotes imply a negative forward rate,
so that no positive, non-increasing discount curve reprices them. A zero forward, as when the
bills printed 0.00%, is accepted. When a month is dropped, the VAR and the return regression
skip the pair of observations that straddles the gap rather than treating a two-month step
as one.</p>

<h2>Method, in one paragraph</h2>
<p class="note">Nine CMT par yields per month are bootstrapped into zero rates honouring the
Treasury's quoting conventions, interpolated by a monotone cubic on log discount factors,
sampled monthly out to ten years, decomposed into principal components, and run through
Adrian-Crump-Moench: a VAR on the factors, a regression of excess holding-period returns on
the innovations and the lagged factors, and the affine pricing recursion run twice, once
under the risk-adjusted dynamics and once with the price of risk switched off. The gap
between the two runs is the premium. Every parameter is estimated on the full sample, so the
premium at a past date uses data from after it: this is a historical decomposition, not one
that could have been computed in real time.</p>

</body></html>
"""

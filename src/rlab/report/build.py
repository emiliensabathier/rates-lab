"""Assembly of the final HTML report.

Computes nothing: every number shown here is produced by the modules under `rlab`.
"""

from __future__ import annotations

import html as html_escape

import numpy as np

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
.headline { font-size: 1.6rem; font-variant-numeric: tabular-nums; margin: 0.5rem 0; }
svg { max-width: 100%; height: auto; }
"""


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html_escape.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html_escape.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _headline(result: Result) -> str:
    observed = float(result.observed_10y.iloc[-1])
    expectations = float(result.expectations_10y.iloc[-1])
    premium = float(result.term_premium_10y.iloc[-1])
    as_of = result.dates[-1].date().isoformat()
    return f"""
<p class="headline">{_pct(observed)} = {_pct(expectations)} expected + {_pct(premium)} premium</p>
<p class="note">Ten-year zero rate as of {as_of}, split by the ACM decomposition.
The first term is what the curve expects the short rate to average over ten years.
The second is what it charges on top for holding the duration.</p>
"""


def _validation(result: Result) -> str:
    stats = agreement(result)
    rows = [
        ["Correlation of levels", f"{stats['correlation_levels']:.3f}"],
        ["Correlation of monthly changes", f"{stats['correlation_changes']:.3f}"],
        ["Mean gap, this estimate minus Kim-Wright", _pct(stats["mean_gap"])],
        ["Overlapping months", f"{int(stats['observations'])}"],
    ]
    return _table(["Against the Kim-Wright series", "Value"], rows)


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
    rows = [
        [name, _pct(share)]
        for name, share in zip(names, result.pca.explained, strict=False)
    ]
    rows.append(["Together", _pct(float(result.pca.explained.sum()))])
    return _table(["Principal component", "Share of variance"], rows)


def _policy_table(result: Result) -> str:
    rows = [
        [f"{row['start']:.2f}y forward", _pct(row["forward_quoted"])]
        for _, row in result.policy_path.iterrows()
    ]
    return _table(["Three-month rate starting", "Priced"], rows)


def _refusals(result: Result) -> str:
    if not result.refusals:
        return "<p class='note'>No month was refused.</p>"
    rows = [[date, reason] for date, reason in sorted(result.refusals.items())]
    return _table(["Month", "Why the curve was not built"], rows)


def render(result: Result, generated_on: str) -> str:
    """Return the whole self-contained page."""
    first = result.dates[0].date().isoformat()
    last = result.dates[-1].date().isoformat()

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>rates-lab: what the yield curve prices</title>
<style>{STYLE}</style></head><body>

<h1>What the yield curve prices</h1>
<p class="note">US Treasury constant-maturity quotes, {first} to {last},
{len(result.dates)} month-end observations. Generated {generated_on}.</p>

{_headline(result)}

<div class="caveat">
A forward rate is not an expectation. The whole point of the decomposition above is that
the two are different, and the difference is the second term. Anyone quoting the ten-year
yield as the market's view of the average short rate is quoting
{_pct(float(result.term_premium_10y.iloc[-1]))} of premium as if it were a forecast.
</div>

{decomposition_chart(result.observed_10y, result.expectations_10y)}

<h2>Checked against a published estimate</h2>
<p class="note">The Federal Reserve Board publishes a ten-year term premium from the
Kim-Wright model: a different specification, fitted by Kalman filter rather than by
regression. Comparing against it is the only external check available on the output of
this package that is not simply its own arithmetic.</p>

{_validation(result)}

<div class="caveat">
<strong>The correlation is the claim. The level is not.</strong>
The two series move together closely, which is evidence that the decomposition is picking
up the thing it is supposed to. They do not agree on how much premium there is: this
estimate runs above Kim-Wright by the mean gap in the table, and the gap widens in the
most recent years. Two known causes, neither of them removed here: OLS on a near-unit-root
VAR under-states persistence, which pushes premium out of the expectations component and
into the residual, and the two models simply differ. Read the level as indicative and the
direction as informative.
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
<p class="note">Estimated with {N_FACTORS} pricing factors in the ACM step; the three shown
here are the ones with a name and a shape.</p>

<h2>The policy path priced today</h2>
{_policy_table(result)}
{policy_chart(result.policy_path)}
<p class="note">Three-month forward rates read off the latest curve, quoted as simple
annualised rates. This is what the curve prices, not what anyone forecasts, and the
premium above is exactly the reason those are different statements.</p>

<h2>Breakeven inflation</h2>
{_table(["Measure", "Value"], [
    ["Ten-year spot breakeven", _pct(result.breakeven_spot_10y)],
    ["Five-year, five-year forward breakeven", _pct(result.breakeven_forward_5y5y)],
])}

<div class="caveat">
A breakeven is expected inflation, plus an inflation risk premium, minus a TIPS liquidity
premium. Nothing public separates the three. The figures above are what the market charges,
not what the market expects.
</div>

<h2>Months this page refused to use</h2>
{_refusals(result)}
<p class="note">A month is dropped when its published quotes do not admit a positive,
decreasing discount curve, which is to say when the front of the bill curve implies a
negative forward rate. Smoothing that away would be inventing a quote nobody published.</p>

<h2>Method, in one paragraph</h2>
<p class="note">Nine CMT par yields per month are bootstrapped into zero rates honouring the
Treasury's own conventions, sampled monthly out to ten years, decomposed into principal
components, and run through Adrian-Crump-Moench: a VAR on the factors, a regression of
excess holding-period returns on the innovations and the lagged factors, and the affine
pricing recursion run twice, once under the estimated risk-neutral dynamics and once with
the price of risk switched off. The gap between the two runs is the premium. Every step
refuses rather than approximates when its inputs do not support it, and every refusal
appears on this page.</p>

</body></html>
"""


def worst_fit_bp(result: Result) -> float:
    """Convenience for callers that only want the headline fit number."""
    return float(np.max(result.fit_error_bp))

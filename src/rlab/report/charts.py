"""Chart rendering to inline SVG.

Charts are embedded directly in the HTML: the report must open offline, with no CDN and no
external asset of any kind.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

FIGSIZE = (9.0, 4.0)
DPI = 110
PERCENT = 100.0


def figure_to_svg(fig: Figure) -> str:
    """Serialize a figure as inline SVG markup, stripped of its XML preamble."""
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    markup = buffer.getvalue()
    return markup[markup.index("<svg") :]


def _new_figure(title: str, ylabel: str) -> tuple[Figure, object]:
    fig = Figure(figsize=FIGSIZE, dpi=DPI)
    axes = fig.add_subplot(111)
    axes.set_title(title)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=0.25)
    return fig, axes


def decomposition_chart(observed: pd.Series, expectations: pd.Series) -> str:
    """The ten-year yield split into what it expects and what it charges.

    The premium is drawn as the band between the two lines rather than as a third line,
    because it is the gap: a reader should not have to hold two levels in their head to
    see it.
    """
    fig, axes = _new_figure(
        "Ten-year yield: expected average short rate, and the premium on top",
        "Percent",
    )
    axes.plot(observed.index, observed * PERCENT, linewidth=1.3, color="#16181d",
              label="Observed ten-year zero")
    axes.plot(expectations.index, expectations * PERCENT, linewidth=1.1, color="#4a6fa5",
              label="Expected average short rate")
    axes.fill_between(observed.index, expectations * PERCENT, observed * PERCENT,
                      alpha=0.25, color="#c0714a", label="Term premium")
    axes.axhline(0.0, color="#9aa0ad", linewidth=0.8)
    axes.legend(loc="upper right", frameon=False)
    return figure_to_svg(fig)


def benchmark_chart(estimated: pd.Series, published: pd.Series) -> str:
    """This estimate against the Federal Reserve Board's Kim-Wright series."""
    fig, axes = _new_figure(
        "Ten-year term premium: this estimate against Kim-Wright", "Percent"
    )
    axes.plot(estimated.index, estimated * PERCENT, linewidth=1.3, color="#c0714a",
              label="ACM, this repository")
    axes.plot(published.index, published * PERCENT, linewidth=1.3, color="#4a6fa5",
              label="Kim-Wright, Federal Reserve Board")
    axes.axhline(0.0, color="#9aa0ad", linewidth=0.8)
    axes.legend(loc="upper right", frameon=False)
    return figure_to_svg(fig)


def fit_chart(maturities_months: np.ndarray, observed: np.ndarray, fitted: np.ndarray) -> str:
    """Today's curve, as quoted and as the affine model prices it."""
    years = maturities_months / 12.0
    fig, axes = _new_figure("Latest curve: observed against the ACM fit", "Percent")
    axes.plot(years, observed * PERCENT, linewidth=1.6, color="#16181d", label="Observed zero")
    axes.plot(years, fitted * PERCENT, linewidth=1.2, color="#c0714a", linestyle="--",
              label="ACM fitted")
    axes.set_xlabel("Maturity, years")
    axes.legend(loc="lower right", frameon=False)
    return figure_to_svg(fig)


def loadings_chart(maturities_months: np.ndarray, loadings: np.ndarray) -> str:
    """The three principal components, in the shape that earns them their names."""
    years = maturities_months / 12.0
    names = ("Level", "Slope", "Curvature")
    fig, axes = _new_figure("Principal components of the zero curve", "Loading")
    for row, name in zip(loadings, names, strict=False):
        axes.plot(years, row, linewidth=1.3, label=name)
    axes.axhline(0.0, color="#9aa0ad", linewidth=0.8)
    axes.set_xlabel("Maturity, years")
    axes.legend(loc="upper right", frameon=False)
    return figure_to_svg(fig)


def policy_chart(path: pd.DataFrame) -> str:
    """The three-month forward path the latest curve prices."""
    fig, axes = _new_figure("Policy path priced by the latest curve", "Percent")
    axes.step(path["start"], path["forward_quoted"] * PERCENT, where="post",
              linewidth=1.4, color="#4a6fa5")
    axes.set_xlabel("Years forward")
    return figure_to_svg(fig)

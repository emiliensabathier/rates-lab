"""Principal components of daily yield-curve changes.

Three components — level, slope, curvature — normally account for almost all of the
variance of a government curve. That is the sanity check of this whole project: if they do
not, the curve built upstream is wrong, and every number derived from it is too.

Written on numpy's SVD rather than pulled from a library, because the sign convention
matters here and libraries do not agree on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlab.errors import ModelError


@dataclass(frozen=True)
class PCAResult:
    """Loadings are (components, tenors); scores are (observations, components)."""

    loadings: np.ndarray
    scores: np.ndarray
    explained: np.ndarray


def decompose(changes: np.ndarray, n_components: int = 3) -> PCAResult:
    """Decompose a panel of curve changes into its principal components."""
    panel = np.asarray(changes, dtype=float)
    if panel.ndim != 2:
        raise ModelError(f"expected a 2-D panel, got shape {panel.shape}")
    if np.any(~np.isfinite(panel)):
        raise ModelError("panel contains missing or non-finite values")
    if n_components > panel.shape[1]:
        raise ModelError(
            f"cannot extract {n_components} components from {panel.shape[1]} tenors"
        )

    centred = panel - panel.mean(axis=0)
    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    variance = singular**2
    explained = variance / variance.sum()

    loadings = right[:n_components]
    # SVD fixes each component only up to sign. Pinning the sign by the row sum makes the
    # first component a level shift with positive loadings on every tenor, which is what the
    # report's chart legend claims it is.
    signs = np.where(loadings.sum(axis=1) < 0, -1.0, 1.0)
    loadings = loadings * signs[:, None]
    scores = centred @ loadings.T

    return PCAResult(loadings=loadings, scores=scores, explained=explained[:n_components])

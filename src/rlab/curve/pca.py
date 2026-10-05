"""Principal components of a yield-curve panel.

The function is agnostic about what it is handed, and the two callers hand it different
things on purpose. The report decomposes monthly *changes*: three components -- level,
slope, curvature -- normally account for almost all of the variance of a government curve's
moves, and that is the sanity check on the curve built upstream. ACM decomposes *levels*,
because its pricing factors are the principal components of the yield panel itself. On
levels the first component also absorbs the persistence of rates, so a share of variance
computed there overstates how much of the curve's movement it explains.

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


def decompose(panel: np.ndarray, n_components: int = 3) -> PCAResult:
    """Decompose a (observations, tenors) panel, of levels or of changes, into components."""
    panel = np.asarray(panel, dtype=float)
    if panel.ndim != 2:
        raise ModelError(f"expected a 2-D panel, got shape {panel.shape}")
    if np.any(~np.isfinite(panel)):
        raise ModelError("panel contains missing or non-finite values")
    if n_components > panel.shape[1]:
        raise ModelError(f"cannot extract {n_components} components from {panel.shape[1]} tenors")

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

import numpy as np
import pytest

from rlab.curve.pca import decompose
from rlab.errors import ModelError


def test_a_pure_level_shock_gives_one_component_holding_all_the_variance():
    rng = np.random.default_rng(0)
    shocks = rng.normal(size=(400, 1))
    changes = shocks @ np.ones((1, 8))
    result = decompose(changes, n_components=3)
    assert result.explained[0] == pytest.approx(1.0, abs=1e-8)


def test_level_plus_slope_gives_two_components_and_the_third_is_empty():
    rng = np.random.default_rng(1)
    tenors = np.linspace(0, 1, 8)
    level = np.ones(8)
    slope = tenors - tenors.mean()
    factors = rng.normal(size=(600, 2)) * np.array([1.0, 0.5])
    changes = factors @ np.vstack([level, slope])
    result = decompose(changes, n_components=3)
    assert result.explained[:2].sum() == pytest.approx(1.0, abs=1e-8)
    assert result.explained[2] == pytest.approx(0.0, abs=1e-10)


def test_loadings_are_sign_normalised_so_the_level_is_positive():
    rng = np.random.default_rng(2)
    changes = rng.normal(size=(300, 1)) @ -np.ones((1, 8))
    result = decompose(changes, n_components=2)
    assert result.loadings[0].sum() > 0


def test_scores_reconstruct_the_input_when_all_components_are_kept():
    rng = np.random.default_rng(3)
    changes = rng.normal(size=(200, 5))
    result = decompose(changes, n_components=5)
    rebuilt = result.scores @ result.loadings + changes.mean(axis=0)
    assert np.allclose(rebuilt, changes, atol=1e-10)


def test_asking_for_more_components_than_tenors_raises():
    with pytest.raises(ModelError, match="components"):
        decompose(np.zeros((10, 3)), n_components=5)


def test_a_panel_with_missing_values_raises_rather_than_dropping_them_quietly():
    changes = np.full((10, 3), 0.1)
    changes[4, 1] = np.nan
    with pytest.raises(ModelError, match="missing"):
        decompose(changes)

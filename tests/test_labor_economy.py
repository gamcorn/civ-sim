"""Tests for the labor economy: soil layers, degradation, production."""
import numpy as np
import pytest
import random as rnd

from civ_sim.config import SimConfig
from civ_sim.world.grid import ResourceGrid


@pytest.fixture
def cfg():
    return SimConfig(width=20, height=20, degradation_rate=0.002, recovery_rate=0.001)


@pytest.fixture
def grid(cfg):
    return ResourceGrid(20, 20, cfg, rnd.Random(42))


def test_grid_has_soil_arrays(grid):
    assert hasattr(grid, "base_soil_fertility")
    assert hasattr(grid, "soil_fertility")
    assert hasattr(grid, "base_mineral_richness")
    assert hasattr(grid, "mineral_richness")
    assert hasattr(grid, "base_forest_density")
    assert hasattr(grid, "forest_density")


def test_soil_arrays_shape(grid):
    assert grid.base_soil_fertility.shape == (20, 20)
    assert grid.soil_fertility.shape == (20, 20)
    assert grid.base_mineral_richness.shape == (20, 20)
    assert grid.mineral_richness.shape == (20, 20)
    assert grid.base_forest_density.shape == (20, 20)
    assert grid.forest_density.shape == (20, 20)


def test_soil_arrays_in_range(grid):
    for arr in (
        grid.base_soil_fertility, grid.soil_fertility,
        grid.base_mineral_richness, grid.mineral_richness,
        grid.base_forest_density, grid.forest_density,
    ):
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 100.0


def test_soil_starts_equal_to_base(grid):
    np.testing.assert_array_equal(grid.soil_fertility, grid.base_soil_fertility)
    np.testing.assert_array_equal(grid.mineral_richness, grid.base_mineral_richness)
    np.testing.assert_array_equal(grid.forest_density, grid.base_forest_density)


def test_avg_soil_fertility_returns_mean_of_owned_tiles(grid, cfg):
    # Claim tiles (5,5) to (7,7) as civ 0 and set known fertility
    for x in range(5, 8):
        for y in range(5, 8):
            grid.ownership[x, y] = 0
            grid.soil_fertility[x, y] = 60.0
    val = grid.avg_soil_fertility(0, 6, 6, 2)
    assert abs(val - 60.0) < 1.0


def test_avg_returns_zero_with_no_owned_tiles(grid):
    # civ 99 owns nothing
    assert grid.avg_soil_fertility(99, 10, 10, 3) == 0.0
    assert grid.avg_mineral_richness(99, 10, 10, 3) == 0.0
    assert grid.avg_forest_density(99, 10, 10, 3) == 0.0


def test_apply_labor_degrades_soil(grid, cfg):
    grid.ownership[10, 10] = 0
    grid.soil_fertility[10, 10] = 80.0
    initial = float(grid.soil_fertility[10, 10])
    grid.apply_labor_degradation(10, 10, 0, 0, 1.0, 0.0, 0.0, cfg)
    assert float(grid.soil_fertility[10, 10]) < initial


def test_apply_labor_no_degradation_when_ratio_zero(grid, cfg):
    grid.ownership[10, 10] = 0
    grid.soil_fertility[10, 10] = 50.0
    grid.base_soil_fertility[10, 10] = 50.0
    grid.apply_labor_degradation(10, 10, 0, 0, 0.0, 0.0, 0.0, cfg)
    # With ratio=0 and tile at base, no change expected
    assert abs(float(grid.soil_fertility[10, 10]) - 50.0) < 0.01


def test_fallow_recovery_moves_toward_base(grid, cfg):
    grid.ownership[10, 10] = 0
    grid.soil_fertility[10, 10] = 20.0
    grid.base_soil_fertility[10, 10] = 80.0
    grid.apply_labor_degradation(10, 10, 0, 0, 0.0, 0.0, 0.0, cfg)
    assert float(grid.soil_fertility[10, 10]) > 20.0


def test_soil_does_not_degrade_below_zero(grid, cfg):
    grid.ownership[10, 10] = 0
    grid.soil_fertility[10, 10] = 0.0
    grid.base_soil_fertility[10, 10] = 0.0
    grid.mineral_richness[10, 10] = 0.0
    grid.base_mineral_richness[10, 10] = 0.0
    grid.forest_density[10, 10] = 0.0
    grid.base_forest_density[10, 10] = 0.0
    grid.apply_labor_degradation(10, 10, 0, 0, 1.0, 1.0, 1.0, cfg)
    assert float(grid.soil_fertility[10, 10]) >= 0.0
    assert float(grid.mineral_richness[10, 10]) >= 0.0
    assert float(grid.forest_density[10, 10]) >= 0.0


from civ_sim.agents.civilization import Civilization, CulturalTraits


def _make_civ():
    return Civilization(0, "Alpha", CulturalTraits())


def test_civ_has_labor_productivity_fields():
    civ = _make_civ()
    assert hasattr(civ, "land_productivity")
    assert hasattr(civ, "mining_efficiency")
    assert hasattr(civ, "forestry_efficiency")
    assert hasattr(civ, "unlocked_actions")


def test_civ_labor_defaults():
    civ = _make_civ()
    assert civ.land_productivity == 0.5
    assert civ.mining_efficiency == 0.5
    assert civ.forestry_efficiency == 0.5
    assert isinstance(civ.unlocked_actions, set)
    assert len(civ.unlocked_actions) == 0

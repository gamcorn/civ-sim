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


from civ_sim.technology.discovery import TECH_TREE, TECH_COSTS, TECH_EFFECTS, TechEngine
from civ_sim.agents.city import CityAgent


def _get_city(mini_model):
    return next(a for a in mini_model.agents if isinstance(a, CityAgent))


def test_mining_in_tech_tree():
    assert "mining" in TECH_TREE
    assert "mining" in TECH_COSTS


def test_forestry_in_tech_tree():
    assert "forestry" in TECH_TREE
    assert "forestry" in TECH_COSTS


def test_agriculture_unlocks_cultivate(mini_model):
    city = _get_city(mini_model)
    city.civ.unlocked_actions.clear()
    mini_model.tech_engine._discover("agriculture", city)
    assert "cultivate" in city.civ.unlocked_actions


def test_mining_unlocks_mine(mini_model):
    city = _get_city(mini_model)
    city.civ.unlocked_actions.clear()
    mini_model.tech_engine._discover("mining", city)
    assert "mine" in city.civ.unlocked_actions


def test_forestry_unlocks_woodcut(mini_model):
    city = _get_city(mini_model)
    city.civ.unlocked_actions.clear()
    mini_model.tech_engine._discover("forestry", city)
    assert "woodcut" in city.civ.unlocked_actions


def test_agriculture_raises_land_productivity(mini_model):
    city = _get_city(mini_model)
    before = city.civ.land_productivity
    mini_model.tech_engine._discover("agriculture", city)
    assert city.civ.land_productivity > before


def test_mining_raises_mining_efficiency(mini_model):
    city = _get_city(mini_model)
    before = city.civ.mining_efficiency
    mini_model.tech_engine._discover("mining", city)
    assert city.civ.mining_efficiency > before


def test_forestry_raises_forestry_efficiency(mini_model):
    city = _get_city(mini_model)
    before = city.civ.forestry_efficiency
    mini_model.tech_engine._discover("forestry", city)
    assert city.civ.forestry_efficiency > before


def test_land_productivity_caps_at_max(mini_model):
    city = _get_city(mini_model)
    city.civ.land_productivity = mini_model.config.efficiency_max - 0.01
    mini_model.tech_engine._discover("agriculture", city)
    assert city.civ.land_productivity <= mini_model.config.efficiency_max


def test_city_has_labor_ratio_fields(mini_model):
    city = _get_city(mini_model)
    assert hasattr(city, "farmer_ratio")
    assert hasattr(city, "miner_ratio")
    assert hasattr(city, "woodcutter_ratio")
    assert city.farmer_ratio == 0.0
    assert city.miner_ratio == 0.0
    assert city.woodcutter_ratio == 0.0


def test_shift_labor_ratio_increases_target(mini_model):
    city = _get_city(mini_model)
    city.farmer_ratio = 0.0
    city._shift_labor_ratio("farmer", 0.1)
    assert abs(city.farmer_ratio - 0.1) < 1e-6


def test_shift_labor_ratio_sum_never_exceeds_one(mini_model):
    city = _get_city(mini_model)
    city.farmer_ratio = 0.5
    city.miner_ratio = 0.4
    city.woodcutter_ratio = 0.0
    city._shift_labor_ratio("woodcutter", 0.2)
    total = city.farmer_ratio + city.miner_ratio + city.woodcutter_ratio
    assert total <= 1.0 + 1e-9


def test_labor_production_scales_with_population(mini_model):
    city = _get_city(mini_model)
    city.civ.discovered_techs.add("agriculture")
    city.civ.unlocked_actions.add("cultivate")
    city.civ.land_productivity = 1.0
    mini_model.grid.ownership[:] = city.civ.civ_id
    mini_model.grid.soil_fertility[:] = 80.0

    city.farmer_ratio = 0.5
    city.population = 200
    city.military = 0
    city.food_stock = 0.0
    city._produce_labor()
    food_large_pop = city.food_stock

    city.farmer_ratio = 0.5
    city.population = 100
    city.military = 0
    city.food_stock = 0.0
    mini_model.grid.soil_fertility[:] = 80.0
    city._produce_labor()
    food_small_pop = city.food_stock

    assert food_large_pop > food_small_pop * 1.5


def test_do_cultivate_decrements_stocks_and_shifts_ratio(mini_model):
    city = _get_city(mini_model)
    city.civ.discovered_techs.add("agriculture")
    city.civ.unlocked_actions.add("cultivate")
    city.wood_stock = 20.0
    city.mineral_stock = 20.0
    city.farmer_ratio = 0.0
    city._do_cultivate()
    assert city.wood_stock < 20.0
    assert city.mineral_stock < 20.0
    assert city.farmer_ratio > 0.0


def test_do_mine_decrements_minerals_and_shifts_ratio(mini_model):
    city = _get_city(mini_model)
    city.civ.discovered_techs.add("mining")
    city.civ.unlocked_actions.add("mine")
    city.mineral_stock = 20.0
    city.miner_ratio = 0.0
    city._do_mine()
    assert city.mineral_stock < 20.0
    assert city.miner_ratio > 0.0


def test_do_woodcut_decrements_wood_and_shifts_ratio(mini_model):
    city = _get_city(mini_model)
    city.civ.discovered_techs.add("forestry")
    city.civ.unlocked_actions.add("woodcut")
    city.wood_stock = 20.0
    city.woodcutter_ratio = 0.0
    city._do_woodcut()
    assert city.wood_stock < 20.0
    assert city.woodcutter_ratio > 0.0


def test_produce_labor_no_output_without_tech(mini_model):
    city = _get_city(mini_model)
    city.civ.discovered_techs.discard("agriculture")
    city.civ.discovered_techs.discard("mining")
    city.civ.discovered_techs.discard("forestry")
    city.civ.unlocked_actions.discard("cultivate")
    city.civ.unlocked_actions.discard("mine")
    city.civ.unlocked_actions.discard("woodcut")
    city.farmer_ratio = 0.8
    city.miner_ratio = 0.1
    city.woodcutter_ratio = 0.1
    city.food_stock = 0.0
    city.mineral_stock = 0.0
    city.wood_stock = 0.0
    mini_model.grid.ownership[:] = city.civ.civ_id
    city._produce_labor()
    assert city.food_stock == 0.0
    assert city.mineral_stock == 0.0
    assert city.wood_stock == 0.0


def test_soil_degrades_after_produce_labor(mini_model):
    city = _get_city(mini_model)
    city.civ.discovered_techs.add("agriculture")
    city.civ.unlocked_actions.add("cultivate")
    city.civ.land_productivity = 1.0
    mini_model.grid.ownership[city.x, city.y] = city.civ.civ_id
    mini_model.grid.soil_fertility[city.x, city.y] = 90.0
    mini_model.grid.base_soil_fertility[city.x, city.y] = 90.0
    city.farmer_ratio = 1.0
    city.miner_ratio = 0.0
    city.woodcutter_ratio = 0.0
    initial = float(mini_model.grid.soil_fertility[city.x, city.y])
    city._produce_labor()
    assert float(mini_model.grid.soil_fertility[city.x, city.y]) < initial


from civ_sim.agents.decisions import (
    CULTIVATE, MINE, WOODCUT, GATHER,
    get_feasible_actions,
)
from tests.conftest import make_mock_city


def test_cultivate_not_feasible_before_agriculture():
    city = make_mock_city(techs=[], wood_stock=50.0, mineral_stock=50.0)
    city.civ.unlocked_actions = set()
    feasible = get_feasible_actions(city)
    assert CULTIVATE not in feasible


def test_cultivate_feasible_after_agriculture():
    city = make_mock_city(techs=["agriculture"], wood_stock=50.0, mineral_stock=50.0)
    city.civ.unlocked_actions = {"cultivate"}
    feasible = get_feasible_actions(city)
    assert CULTIVATE in feasible


def test_mine_not_feasible_before_mining():
    city = make_mock_city(techs=[], mineral_stock=50.0)
    city.civ.unlocked_actions = set()
    feasible = get_feasible_actions(city)
    assert MINE not in feasible


def test_mine_feasible_after_mining():
    city = make_mock_city(techs=["mining"], mineral_stock=50.0)
    city.civ.unlocked_actions = {"mine"}
    feasible = get_feasible_actions(city)
    assert MINE in feasible


def test_woodcut_feasible_after_forestry():
    city = make_mock_city(techs=["forestry"], wood_stock=50.0)
    city.civ.unlocked_actions = {"woodcut"}
    feasible = get_feasible_actions(city)
    assert WOODCUT in feasible


def test_gather_always_feasible():
    city = make_mock_city(techs=[])
    city.civ.unlocked_actions = set()
    feasible = get_feasible_actions(city)
    assert GATHER in feasible


def test_gather_still_feasible_after_agriculture():
    city = make_mock_city(techs=["agriculture"], wood_stock=50.0, mineral_stock=50.0)
    city.civ.unlocked_actions = {"cultivate"}
    feasible = get_feasible_actions(city)
    assert GATHER in feasible


from civ_sim.agents.providers.council_prompts import (
    build_civ_state_snapshot,
    CHIEF_SCHEMA_DICT,
    CHIEF_LITE_SCHEMA_DICT,
    MINISTER_SPECS,
)


def test_state_snapshot_includes_labor_block(mini_model):
    city = _get_city(mini_model)
    city.farmer_ratio = 0.3
    city.miner_ratio = 0.2
    city.woodcutter_ratio = 0.1
    cities = [city]
    snapshot = build_civ_state_snapshot(city.civ, cities, mini_model)
    assert "Labor allocation" in snapshot
    assert "farmers" in snapshot
    assert "miners" in snapshot
    assert "woodcutters" in snapshot


def test_state_snapshot_includes_soil_stats(mini_model):
    city = _get_city(mini_model)
    cities = [city]
    snapshot = build_civ_state_snapshot(city.civ, cities, mini_model)
    assert "soil fertility" in snapshot
    assert "mineral richness" in snapshot
    assert "forest density" in snapshot


def test_chief_schema_includes_new_actions():
    required = CHIEF_SCHEMA_DICT["properties"]["action_weights"]["required"]
    assert "cultivate" in required
    assert "mine" in required
    assert "woodcut" in required


def test_chief_lite_schema_includes_new_actions():
    required = CHIEF_LITE_SCHEMA_DICT["properties"]["action_weights"]["required"]
    assert "cultivate" in required
    assert "mine" in required
    assert "woodcut" in required


def test_economy_minister_has_new_actions():
    economy = next(s for s in MINISTER_SPECS if s["name"] == "Minister of Economy")
    assert "cultivate" in economy["actions"]
    assert "mine" in economy["actions"]
    assert "woodcut" in economy["actions"]

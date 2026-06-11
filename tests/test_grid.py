import pytest

from civ_sim.config import SimConfig
from civ_sim.world.grid import ResourceGrid
from civ_sim.world.resources import ResourceType


@pytest.fixture
def grid():
    cfg = SimConfig(
        width=20,
        height=20,
        resource_max=100.0,
        food_regen=0.04,
        water_regen=0.02,
        wood_regen=0.015,
        mineral_regen=0.0,
        rng_seed=1,
    )
    import random as stdlib_random

    rng = stdlib_random.Random(1)
    return ResourceGrid(20, 20, cfg, rng)


def test_consume_returns_min_of_available_and_requested(grid):
    grid.layers[ResourceType.FOOD].data[5, 5] = 10.0
    consumed = grid.consume(5, 5, ResourceType.FOOD, 6.0)
    assert consumed == pytest.approx(6.0)
    assert grid.get(5, 5, ResourceType.FOOD) == pytest.approx(4.0)


def test_consume_on_empty_tile_returns_zero(grid):
    grid.layers[ResourceType.FOOD].data[3, 3] = 0.0
    consumed = grid.consume(3, 3, ResourceType.FOOD, 10.0)
    assert consumed == pytest.approx(0.0)


def test_consume_does_not_go_negative(grid):
    grid.layers[ResourceType.FOOD].data[2, 2] = 3.0
    grid.consume(2, 2, ResourceType.FOOD, 50.0)
    assert grid.get(2, 2, ResourceType.FOOD) >= 0.0


def test_deposit_adds_to_tile(grid):
    grid.layers[ResourceType.MINERALS].data[1, 1] = 10.0
    grid.deposit(1, 1, ResourceType.MINERALS, 5.0)
    assert grid.get(1, 1, ResourceType.MINERALS) == pytest.approx(15.0)


def test_deposit_clamps_at_resource_max(grid):
    grid.layers[ResourceType.MINERALS].data[1, 1] = 95.0
    grid.deposit(1, 1, ResourceType.MINERALS, 20.0)
    assert grid.get(1, 1, ResourceType.MINERALS) == pytest.approx(100.0)


def test_get_returns_float(grid):
    val = grid.get(0, 0, ResourceType.FOOD)
    assert isinstance(val, float)


def test_claim_sets_ownership(grid):
    grid.claim(4, 4, civ_id=0)
    assert grid.ownership[4, 4] == 0


def test_territory_count_counts_claimed_tiles(grid):
    grid.claim(0, 0, civ_id=1)
    grid.claim(1, 0, civ_id=1)
    grid.claim(2, 0, civ_id=0)
    assert grid.territory_count(1) == 2
    assert grid.territory_count(0) == 1


def test_step_regenerates_food_up_to_cap(grid):
    grid.layers[ResourceType.FOOD].data[:] = 0.0
    grid.step()
    # food_regen=0.04 → 0.04 * 100 = 4 per tile
    assert grid.get(0, 0, ResourceType.FOOD) == pytest.approx(4.0, abs=0.01)


def test_step_does_not_exceed_resource_max(grid):
    grid.layers[ResourceType.FOOD].data[:] = 100.0
    grid.step()
    assert grid.get(0, 0, ResourceType.FOOD) == pytest.approx(100.0)


def test_step_minerals_do_not_regen_when_rate_is_zero(grid):
    grid.layers[ResourceType.MINERALS].data[5, 5] = 20.0
    grid.step()
    assert grid.get(5, 5, ResourceType.MINERALS) == pytest.approx(20.0)

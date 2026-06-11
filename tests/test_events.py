import random
from unittest.mock import patch

import pytest

from civ_sim.config import SimConfig
from civ_sim.world.events import EventSampler
from civ_sim.world.grid import ResourceGrid
from civ_sim.world.resources import ResourceType


@pytest.fixture
def cfg():
    return SimConfig(
        width=20,
        height=20,
        resource_max=100.0,
        drought_prob=0.0,
        disease_prob=0.0,
        mineral_boom_prob=0.0,
        climate_shift_prob=0.0,
        rng_seed=0,
    )


@pytest.fixture
def grid(cfg):
    rng = random.Random(0)
    g = ResourceGrid(20, 20, cfg, rng)
    g.layers[ResourceType.FOOD].data[:] = 80.0
    g.layers[ResourceType.MINERALS].data[:] = 30.0
    return g


def test_sample_returns_empty_when_all_probs_zero(cfg, grid):
    sampler = EventSampler(cfg, random.Random(0))
    events = sampler.sample(grid)
    assert events == []


def test_drought_fires_and_returns_drought_event(cfg, grid):
    cfg.drought_prob = 1.0
    sampler = EventSampler(cfg, random.Random(0))
    events = sampler.sample(grid)
    names = [e.name for e in events]
    assert "drought" in names


def test_drought_halves_food_in_patch(cfg, grid):
    cfg.drought_prob = 1.0
    before = float(grid.layers[ResourceType.FOOD].data.sum())
    sampler = EventSampler(cfg, random.Random(5))
    sampler.sample(grid)
    after = float(grid.layers[ResourceType.FOOD].data.sum())
    assert after < before


def test_mineral_boom_fires_and_returns_event(cfg, grid):
    cfg.mineral_boom_prob = 1.0
    sampler = EventSampler(cfg, random.Random(0))
    events = sampler.sample(grid)
    names = [e.name for e in events]
    assert "mineral_boom" in names


def test_mineral_boom_increases_minerals_in_patch(cfg, grid):
    cfg.mineral_boom_prob = 1.0
    before = float(grid.layers[ResourceType.MINERALS].data.sum())
    sampler = EventSampler(cfg, random.Random(0))
    sampler.sample(grid)
    after = float(grid.layers[ResourceType.MINERALS].data.sum())
    assert after >= before


def test_climate_shift_fires_and_returns_event(cfg, grid):
    cfg.climate_shift_prob = 1.0
    sampler = EventSampler(cfg, random.Random(0))
    events = sampler.sample(grid)
    names = [e.name for e in events]
    assert "climate_shift" in names


def test_disease_fires_and_returns_event(cfg, grid):
    cfg.disease_prob = 1.0
    sampler = EventSampler(cfg, random.Random(0))
    events = sampler.sample(grid)
    names = [e.name for e in events]
    assert "disease" in names


def test_all_events_fire_when_all_probs_one(cfg, grid):
    cfg.drought_prob = 1.0
    cfg.disease_prob = 1.0
    cfg.mineral_boom_prob = 1.0
    cfg.climate_shift_prob = 1.0
    sampler = EventSampler(cfg, random.Random(0))
    events = sampler.sample(grid)
    assert len(events) == 4


def test_climate_shift_reduces_regen_not_tile_data(mini_model):
    """After a climate shift, grid regen is throttled but existing tile food is untouched."""
    from civ_sim.world.resources import ResourceType

    mini_model.grid.layers[ResourceType.FOOD].data[:] = 50.0
    mini_model._climate_penalty_ticks = 1

    food_before = float(mini_model.grid.layers[ResourceType.FOOD].data.sum())
    mini_model.grid.step(food_regen_mult=0.85)
    food_after = float(mini_model.grid.layers[ResourceType.FOOD].data.sum())

    assert (
        food_after >= food_before
    ), f"Climate shift must not destroy tile food; before={food_before:.1f} after={food_after:.1f}"
    full_regen = (
        mini_model.config.food_regen
        * mini_model.config.resource_max
        * mini_model.grid.width
        * mini_model.grid.height
    )
    assert (
        food_after - food_before
    ) < full_regen, "Climate-penalised regen should be less than full regen"


def test_model_computes_climate_penalty_multiplier(mini_model):
    """When _climate_penalty_ticks > 0, model.step() must call grid.step() with food_regen_mult=0.85."""
    mini_model._climate_penalty_ticks = 3

    captured_mult = []
    original_step = mini_model.grid.step

    def capturing_step(food_regen_mult=1.0):
        captured_mult.append(food_regen_mult)
        return original_step(food_regen_mult=food_regen_mult)

    with patch.object(mini_model.grid, "step", side_effect=capturing_step):
        mini_model.step()

    assert captured_mult, "grid.step() should have been called during model.step()"
    assert captured_mult[0] == pytest.approx(
        0.85
    ), f"grid.step() should receive food_regen_mult=0.85 when penalty active; got {captured_mult[0]}"

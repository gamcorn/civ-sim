import random
import pytest
from config import SimConfig
from world.events import EventSampler
from world.grid import ResourceGrid
from world.resources import ResourceType


@pytest.fixture
def cfg():
    return SimConfig(
        width=20, height=20, resource_max=100.0,
        drought_prob=0.0, disease_prob=0.0,
        mineral_boom_prob=0.0, climate_shift_prob=0.0,
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

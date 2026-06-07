import math
import numpy as np
import pytest
from config import SimConfig
from simulation.model import CivModel
from agents.city import CityAgent


@pytest.fixture
def mini_config():
    return SimConfig(
        width=20, height=20, num_civs=2, cities_per_civ=1,
        max_ticks=5, rng_seed=0, db_path=":memory:", visualize=False,
    )


def test_military_decays_each_tick(mini_config):
    """Each call to _consume_resources reduces military by int(military * 0.02)."""
    model = CivModel(mini_config)
    city = next(a for a in model.agents if isinstance(a, CityAgent))
    city.military = 100
    city._consume_resources()
    assert city.military < 100, "Military should decay each tick"
    assert city.military == 98, "int(100 * 0.02) = 2, so 100 - 2 = 98"


def test_small_military_does_not_decay(mini_config):
    """Cities with military < 50 have no decay (int(49*0.02)=0)."""
    model = CivModel(mini_config)
    city = next(a for a in model.agents if isinstance(a, CityAgent))
    city.military = 49
    city._consume_resources()
    assert city.military == 49, "int(49 * 0.02) = 0, no decay for small military"


def test_apply_disease_reduces_all_city_populations(mini_config):
    """_apply_disease hits every city with ~20% population loss."""
    model = CivModel(mini_config)
    cities = [a for a in model.agents if isinstance(a, CityAgent)]
    pops_before = {c.unique_id: c.population for c in cities}

    model._apply_disease()

    for city in cities:
        assert city.population < pops_before[city.unique_id], (
            f"City {city.unique_id} population should have dropped after disease"
        )


def test_border_tiles_revert_with_probability_one(mini_config):
    """With reversion prob=1.0, every tile adjacent to an enemy border reverts to -1."""
    model = CivModel(mini_config)
    # Clear ownership and set up a minimal border: civ 0 at (5,5), civ 1 at (6,5)
    model.grid.ownership[:] = -1
    model.grid.ownership[5, 5] = 0
    model.grid.ownership[6, 5] = 1

    model.config.border_reversion_prob = 1.0
    model._apply_border_reversion()

    assert model.grid.ownership[5, 5] == -1, "Civ-0 border tile should revert"
    assert model.grid.ownership[6, 5] == -1, "Civ-1 border tile should revert"

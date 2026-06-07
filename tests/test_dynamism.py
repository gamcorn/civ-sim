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
    """Each call to _consume_resources reduces military by ceil(military * 0.008)."""
    model = CivModel(mini_config)
    city = next(a for a in model.agents if isinstance(a, CityAgent))
    city.military = 100
    city._consume_resources()
    assert city.military < 100, "Military should decay each tick"
    assert city.military == 99, "ceil(100 * 0.008) = 1, so 100 - 1 = 99"


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

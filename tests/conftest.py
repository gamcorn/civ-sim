import pytest
from unittest.mock import MagicMock
import numpy as np
from civ_sim.config import SimConfig
from civ_sim.simulation.model import CivModel


def make_mock_city(
    pop=200, military=20, food_stock=150.0,
    aggressiveness=0.5, trust=0.5, innovation=0.5,
    tribalism=0.5, risk_tolerance=0.5,
    civ_id=0, civ_name="Alpha",
    enemy_military=20, tick=10,
    techs=None, territory=50,
    wood=30.0, minerals=20.0,
    wood_stock=30.0, mineral_stock=20.0,
    has_unclaimed_neighbor=True,
    has_trade_partner=False,
):
    """Build a minimal mock CityAgent with all attributes providers need."""
    city = MagicMock()

    # Cultural traits
    traits = MagicMock()
    traits.aggressiveness = aggressiveness
    traits.trust = trust
    traits.innovation = innovation
    traits.tribalism = tribalism
    traits.risk_tolerance = risk_tolerance
    traits.as_dict.return_value = {
        "aggressiveness": aggressiveness,
        "trust": trust,
        "innovation": innovation,
        "tribalism": tribalism,
        "risk_tolerance": risk_tolerance,
    }

    # Civilization
    civ = MagicMock()
    civ.civ_id = civ_id
    civ.name = civ_name
    civ.traits = traits
    civ.total_military = military
    civ.discovered_techs = set(techs or [])
    civ.trade_range_bonus = 0
    civ.military_tech_bonus = 0.0

    # Enemy civ
    enemy_civ = MagicMock()
    enemy_civ.civ_id = 1 - civ_id
    enemy_civ.name = "Beta"
    enemy_civ.total_pop = 0
    enemy_civ.total_military = enemy_military
    enemy_civ.city_count = 1
    enemy_civ.tech_level = 0

    # Grid
    grid = MagicMock()
    grid.width = 80
    grid.height = 60
    grid.territory_count.return_value = territory
    ownership = np.full((80, 60), -1, dtype=np.int8)
    # Claim city tile and a few neighbors for civ_id
    ownership[10, 10] = civ_id
    # Ensure unclaimed neighbor exists if requested
    grid.ownership = ownership

    def _get(x, y, rt):
        from civ_sim.world.resources import ResourceType
        if rt == ResourceType.WOOD:
            return wood
        if rt == ResourceType.MINERALS:
            return minerals
        return 50.0  # food default
    grid.get.side_effect = _get

    # Model
    model = MagicMock()
    model.steps = tick
    model.config.resource_max = 100.0
    model.config.harvest_radius = 5
    model.config.fog_of_war = 0.0
    model.config.expand_wood_cost = 5.0
    model.config.research_wood_cost = 8.0
    model.config.research_mineral_cost = 5.0
    model.config.attack_mineral_cost = 3.0
    model.config.initial_pop = 50
    model.config.recruit_pop_cost = 10
    model.config.recruit_mineral_cost = 3.0
    model.grid = grid
    model.civilizations = [civ, enemy_civ]
    # agents_by_type returns empty by default (no trade partners, no attack targets)
    model.agents_by_type = {}

    # City
    city.civ = civ
    city.model = model
    city.population = pop
    city.military = military
    city.food_stock = food_stock
    city.wood_stock = wood_stock
    city.mineral_stock = mineral_stock
    city.x = 10
    city.y = 10
    city._pending_action = None

    return city


@pytest.fixture
def mock_city():
    return make_mock_city()


@pytest.fixture
def mini_config():
    return SimConfig(
        width=20,
        height=20,
        num_civs=2,
        cities_per_civ=1,
        max_ticks=5,
        rng_seed=0,
        db_path=":memory:",
        visualize=False,
    )


@pytest.fixture
def mini_model(mini_config):
    model = CivModel(mini_config)
    yield model
    if hasattr(model, "logger") and model.logger:
        try:
            model.logger.close()
        except Exception:
            pass

import pytest
from unittest.mock import MagicMock
import numpy as np


def make_mock_city(
    pop=200, military=20, food_stock=150.0,
    aggressiveness=0.5, trust=0.5, innovation=0.5,
    tribalism=0.5, risk_tolerance=0.5,
    civ_id=0, civ_name="Alpha",
    enemy_military=20, tick=10,
    techs=None, territory=50,
    wood=30.0, minerals=20.0,
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

    # Enemy civ
    enemy_civ = MagicMock()
    enemy_civ.civ_id = 1 - civ_id
    enemy_civ.total_military = enemy_military

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
        from world.resources import ResourceType
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
    city.x = 10
    city.y = 10
    city._pending_action = None

    return city


@pytest.fixture
def mock_city():
    return make_mock_city()

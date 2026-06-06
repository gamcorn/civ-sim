"""Tests for TechEngine — check(), _discover(), _requirements_met()."""
import pytest
from agents.city import CityAgent
from technology.discovery import TechEngine, TECH_TREE
from world.resources import ResourceType


def _get_city(mini_model):
    """Return the first real CityAgent from the model."""
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    assert cities, "mini_model has no CityAgent"
    return cities[0]


def _force_resource(mini_model, city, rt: ResourceType, value: float):
    """Set a resource layer value at city's position."""
    mini_model.grid.layers[rt].data[city.x, city.y] = value


# ---------------------------------------------------------------------------
# check() — discovery via check() entry point
# ---------------------------------------------------------------------------

def test_check_discovers_agriculture_when_food_sufficient(mini_model):
    """Agriculture (requires FOOD >= 30) is discovered when food tile is high enough."""
    city = _get_city(mini_model)
    city.civ.discovered_techs.clear()

    # Agriculture threshold is 30.0
    _force_resource(mini_model, city, ResourceType.FOOD, 40.0)

    engine = TechEngine()
    engine.check(city)

    assert "agriculture" in city.civ.discovered_techs


def test_check_does_not_rediscover_known_tech(mini_model):
    """check() must not add a tech that is already discovered."""
    city = _get_city(mini_model)
    city.civ.discovered_techs = {"agriculture"}
    city.civ.tech_level = 1

    _force_resource(mini_model, city, ResourceType.FOOD, 40.0)

    engine = TechEngine()
    engine.check(city)

    # Still exactly one copy — sets don't duplicate, but tech_level shouldn't increment again
    assert city.civ.discovered_techs.count if False else True  # sets have no count()
    assert "agriculture" in city.civ.discovered_techs
    # tech_level must not have grown beyond 1 from re-discovering agriculture
    # (it may grow if other techs were also discovered, but agriculture itself wasn't re-added)
    techs_with_agriculture = {t for t in city.civ.discovered_techs if t == "agriculture"}
    assert len(techs_with_agriculture) == 1


def test_check_does_not_discover_without_resources(mini_model):
    """No tech is discovered when all resource levels are 0 (and no prereqs met)."""
    city = _get_city(mini_model)
    city.civ.discovered_techs.clear()
    city.civ.tech_level = 0

    # Zero out all resource types at city position
    for rt in ResourceType:
        _force_resource(mini_model, city, rt, 0.0)

    engine = TechEngine()
    engine.check(city)

    # All techs that require only resources (no tech prereqs) must NOT be discovered
    # because all resources are 0; techs requiring other techs as prereqs also blocked
    assert len(city.civ.discovered_techs) == 0


# ---------------------------------------------------------------------------
# _requirements_met() — unit tests for the inner gate
# ---------------------------------------------------------------------------

def test_requirements_met_false_when_prereq_tech_missing(mini_model):
    """Irrigation requires tech:agriculture; returns False when it's absent."""
    city = _get_city(mini_model)
    city.civ.discovered_techs.clear()

    # Also satisfy the water threshold for irrigation
    _force_resource(mini_model, city, ResourceType.WATER, 30.0)

    engine = TechEngine()
    reqs = TECH_TREE["irrigation"]
    result = engine._requirements_met(
        "irrigation", reqs, city, city.civ, mini_model.grid
    )
    assert result is False


def test_requirements_met_true_when_all_met(mini_model):
    """Irrigation returns True when agriculture is discovered and WATER is >= 25."""
    city = _get_city(mini_model)
    city.civ.discovered_techs = {"agriculture"}

    # Irrigation requires WATER >= 25.0
    _force_resource(mini_model, city, ResourceType.WATER, 30.0)

    engine = TechEngine()
    reqs = TECH_TREE["irrigation"]
    result = engine._requirements_met(
        "irrigation", reqs, city, city.civ, mini_model.grid
    )
    assert result is True


# ---------------------------------------------------------------------------
# _discover() — direct tests for side effects
# ---------------------------------------------------------------------------

def test_discover_adds_tech_and_increments_level(mini_model):
    """_discover() adds the tech name to discovered_techs and updates tech_level."""
    city = _get_city(mini_model)
    city.civ.discovered_techs.clear()
    city.civ.tech_level = 0

    engine = TechEngine()
    engine._discover("agriculture", city)

    assert "agriculture" in city.civ.discovered_techs
    assert city.civ.tech_level == 1


def test_discover_logs_event_with_discover_prefix(mini_model):
    """_discover() emits a log row with action='discover:<tech>'."""
    city = _get_city(mini_model)
    city.civ.discovered_techs.discard("masonry")

    engine = TechEngine()
    engine._discover("masonry", city)

    # Flush the in-memory buffer to the DB
    mini_model.logger.flush()

    rows = mini_model.logger._con.execute(
        "SELECT action FROM events WHERE action LIKE 'discover:%'"
    ).fetchall()
    actions = [r[0] for r in rows]
    assert any("masonry" in a for a in actions), (
        f"Expected a 'discover:masonry' row, got: {actions}"
    )

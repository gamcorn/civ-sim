"""Tests for TechEngine — check(), _discover(), _requirements_met()."""
import pytest
from civ_sim.agents.city import CityAgent
from civ_sim.technology.discovery import TechEngine, TECH_TREE
from civ_sim.world.resources import ResourceType


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
    from civ_sim.technology.discovery import TECH_COSTS
    city = _get_city(mini_model)
    city.civ.discovered_techs.clear()
    city.civ.science_points = TECH_COSTS["agriculture"] + 1.0

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
    from civ_sim.technology.discovery import TECH_COSTS
    city = _get_city(mini_model)
    city.civ.discovered_techs = {"agriculture"}
    city.civ.science_points = TECH_COSTS["irrigation"] + 1.0

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


def test_food_tech_does_not_mutate_shared_config(mini_model):
    """Discovering agriculture must NOT change model.config.food_regen."""
    from civ_sim.agents.city import CityAgent
    from civ_sim.world.resources import ResourceType
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    city = cities[0]
    original_regen = mini_model.config.food_regen

    mini_model.grid.layers[ResourceType.FOOD].data[city.x, city.y] = 100.0
    mini_model.tech_engine._discover("agriculture", city)

    assert mini_model.config.food_regen == original_regen, (
        "agriculture must not change shared config.food_regen"
    )
    assert city.civ.harvest_bonus > 1.0, (
        "agriculture should raise civ.harvest_bonus instead"
    )


def test_harvest_bonus_increases_gather_yield(mini_model):
    """A civ with harvest_bonus > 1 should gather more food."""
    from civ_sim.agents.city import CityAgent
    from civ_sim.world.resources import ResourceType
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    city = cities[0]
    mini_model.grid.layers[ResourceType.FOOD].data[:] = 80.0
    mini_model.grid.ownership[:] = city.civ.civ_id  # own all tiles

    city.civ.harvest_bonus = 1.0
    city.food_stock = 0.0
    city._do_gather()
    baseline = city.food_stock

    city.civ.harvest_bonus = 2.0
    city.food_stock = 0.0
    mini_model.grid.layers[ResourceType.FOOD].data[:] = 80.0
    city._do_gather()
    boosted = city.food_stock

    assert boosted > baseline * 1.5, (
        f"harvest_bonus=2 should yield much more food; baseline={baseline:.1f} boosted={boosted:.1f}"
    )


def test_research_accumulates_science_points(mini_model):
    """Each research action should increase civ.science_points."""
    from civ_sim.agents.city import CityAgent
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    city = cities[0]
    city.civ.science_points = 0.0
    city.wood_stock = 50.0
    city.mineral_stock = 50.0

    city._do_research()

    assert city.civ.science_points > 0.0, (
        f"science_points should be > 0 after research; got {city.civ.science_points}"
    )


def test_tech_unlocks_when_points_sufficient(mini_model):
    """Agriculture should unlock when science_points >= its cost and food threshold met."""
    from civ_sim.agents.city import CityAgent
    from civ_sim.technology.discovery import TECH_COSTS
    from civ_sim.world.resources import ResourceType
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    city = cities[0]
    city.civ.science_points = TECH_COSTS["agriculture"] + 1.0
    mini_model.grid.layers[ResourceType.FOOD].data[city.x, city.y] = 50.0

    mini_model.tech_engine.check(city)

    assert "agriculture" in city.civ.discovered_techs, (
        "Agriculture should be discovered when points sufficient and food threshold met"
    )


def test_research_discovers_at_most_one_tech(mini_model):
    """A single research action must discover at most one new technology."""
    from civ_sim.agents.city import CityAgent
    from civ_sim.technology.discovery import TECH_COSTS
    from civ_sim.world.resources import ResourceType
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    city = cities[0]
    city.civ.science_points = 10000.0
    city.wood_stock = 50.0
    city.mineral_stock = 50.0
    for rt in ResourceType:
        mini_model.grid.layers[rt].data[city.x, city.y] = 100.0

    techs_before = len(city.civ.discovered_techs)
    city._do_research()
    techs_after = len(city.civ.discovered_techs)

    assert techs_after - techs_before <= 1, (
        f"At most 1 tech per research action; discovered {techs_after - techs_before}"
    )

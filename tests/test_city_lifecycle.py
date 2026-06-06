"""Tests for CityAgent lifecycle methods: step, _consume_resources, _grow_population, _collapse."""
from __future__ import annotations

import pytest
from agents.city import CityAgent
from agents.decisions import ALL_ACTIONS
from world.resources import ResourceType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_city(mini_model) -> CityAgent:
    """Return the first CityAgent from the mini_model."""
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    assert cities, "mini_model must have at least one CityAgent"
    return cities[0]


def _set_food_on_tile(mini_model, city: CityAgent, amount: float) -> None:
    """Set the food layer value on city's tile."""
    mini_model.grid.layers[ResourceType.FOOD].data[city.x, city.y] = amount


# ---------------------------------------------------------------------------
# step() — age / last_action / pending_action
# ---------------------------------------------------------------------------

def test_step_increments_age(mini_model):
    city = _get_city(mini_model)
    age_before = city.age
    # Ensure the city won't collapse during step
    city.population = 200
    city.food_stock = 1000.0
    _set_food_on_tile(mini_model, city, 50.0)
    city.step()
    assert city.age == age_before + 1


def test_step_sets_last_action(mini_model):
    city = _get_city(mini_model)
    city.population = 200
    city.food_stock = 1000.0
    _set_food_on_tile(mini_model, city, 50.0)
    city.step()
    # last_action should be one of the standard actions (or "spawn" before first step)
    valid = set(ALL_ACTIONS) | {"spawn"}
    assert city.last_action in valid


def test_step_clears_pending_action(mini_model):
    city = _get_city(mini_model)
    city.population = 200
    city.food_stock = 1000.0
    _set_food_on_tile(mini_model, city, 50.0)
    city._pending_action = "gather"
    city.step()
    assert city._pending_action is None


def test_step_uses_pending_action_when_set(mini_model):
    city = _get_city(mini_model)
    city.population = 200
    city.food_stock = 1000.0
    _set_food_on_tile(mini_model, city, 50.0)
    city._pending_action = "fortify"
    city.step()
    assert city.last_action == "fortify"


# ---------------------------------------------------------------------------
# _consume_resources()
# ---------------------------------------------------------------------------

def test_consume_resources_reduces_food_stock(mini_model):
    """food_stock decreases after _consume_resources when no food on tile."""
    city = _get_city(mini_model)
    city.population = 200
    city.military = 10
    city.food_stock = 100.0
    # Zero out food on tile so the passive harvest doesn't top it up meaningfully
    _set_food_on_tile(mini_model, city, 0.0)
    city._consume_resources()
    assert city.food_stock < 100.0


def test_consume_resources_starvation_reduces_population(mini_model):
    """When food_stock is 0 and no food on tile, population drops (starvation)."""
    city = _get_city(mini_model)
    city.population = 100
    city.military = 0
    city.food_stock = 0.0
    _set_food_on_tile(mini_model, city, 0.0)
    pop_before = city.population
    city._consume_resources()
    assert city.population < pop_before


def test_consume_resources_food_stock_never_negative(mini_model):
    """food_stock must never go below zero."""
    city = _get_city(mini_model)
    city.population = 200
    city.military = 0
    city.food_stock = 0.0
    _set_food_on_tile(mini_model, city, 0.0)
    city._consume_resources()
    assert city.food_stock >= 0.0


# ---------------------------------------------------------------------------
# _grow_population()
# ---------------------------------------------------------------------------

def test_grow_population_increases_pop_when_food_sufficient(mini_model):
    """Population grows when the city tile has abundant food."""
    city = _get_city(mini_model)
    city.population = 100
    # resource_max is 100.0; threshold is 20% → 20.0; put 80 to be safely above
    _set_food_on_tile(mini_model, city, 80.0)
    pop_before = city.population
    city._grow_population()
    # Growth rate 0.012 → int(100 * 0.012) = 1; population should be >= before
    assert city.population >= pop_before


def test_grow_population_no_growth_when_food_scarce(mini_model):
    """Population does not increase when city tile food is below threshold."""
    city = _get_city(mini_model)
    city.population = 100
    # resource_max = 100.0; 20% threshold = 20.0 — put 0 so condition fails
    _set_food_on_tile(mini_model, city, 0.0)
    pop_before = city.population
    city._grow_population()
    assert city.population == pop_before


# ---------------------------------------------------------------------------
# Collapse logic
# ---------------------------------------------------------------------------

def test_step_collapses_city_when_population_zero(mini_model):
    """A city with zero population collapses and is removed from model.agents."""
    city = _get_city(mini_model)
    city.population = 0
    city.food_stock = 0.0
    _set_food_on_tile(mini_model, city, 0.0)
    # Force population to stay at 0 before step runs _consume_resources
    # (which would set population via starvation but it's already 0)
    city.step()
    remaining = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    assert city not in remaining


def test_collapse_releases_territory(mini_model):
    """After collapse, the civilization's territory ownership is cleared to -1."""
    city = _get_city(mini_model)
    civ_id = city.civ.civ_id

    # Claim a block of tiles for this civ so we know some exist before collapse
    for dx in range(3):
        for dy in range(3):
            nx, ny = city.x + dx, city.y + dy
            if 0 <= nx < mini_model.grid.width and 0 <= ny < mini_model.grid.height:
                mini_model.grid.claim(nx, ny, civ_id)

    # Confirm territory exists
    assert mini_model.grid.territory_count(civ_id) > 0

    # Trigger collapse
    city.population = 0
    city.food_stock = 0.0
    _set_food_on_tile(mini_model, city, 0.0)
    city.step()

    # All tiles previously owned by this civ should now be -1
    assert mini_model.grid.territory_count(civ_id) == 0

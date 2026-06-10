"""Integration tests for CityAgent action methods (_do_*).

Uses the ``mini_model`` fixture (real CivModel, 20×20 grid, 2 civs, in-memory
DuckDB) so every test exercises real grid state, not mocks.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from civ_sim.agents.city import CityAgent
from civ_sim.world.resources import ResourceType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cities(model) -> list[CityAgent]:
    """Return all CityAgent instances in the model."""
    return [a for a in model.agents if isinstance(a, CityAgent)]


# ---------------------------------------------------------------------------
# _do_gather
# ---------------------------------------------------------------------------

def test_do_gather_increases_food_stock(mini_model):
    """Gathering from claimed tiles in harvest radius should increase food_stock."""
    cities = _get_cities(mini_model)
    assert cities, "mini_model must have at least one city"
    city = cities[0]

    # Deposit food on the city tile itself (already claimed) so consume has something
    mini_model.grid.deposit(city.x, city.y, ResourceType.FOOD, 50.0)

    food_before = city.food_stock
    city._do_gather()
    assert city.food_stock > food_before, (
        f"food_stock should increase after _do_gather; was {food_before}, now {city.food_stock}"
    )


# ---------------------------------------------------------------------------
# _do_expand
# ---------------------------------------------------------------------------

def test_do_expand_claims_an_unclaimed_tile(mini_model):
    """_do_expand should claim one unclaimed tile within radius 3."""
    cities = _get_cities(mini_model)
    city = cities[0]
    grid = mini_model.grid

    # Ensure at least one tile near city is unclaimed
    # (on a 20×20 grid with 1 city per civ, there's plenty of unclaimed space)
    before = grid.territory_count(city.civ.civ_id)
    city._do_expand()
    after = grid.territory_count(city.civ.civ_id)
    assert after > before, (
        f"territory should grow after _do_expand; was {before}, now {after}"
    )


def test_do_expand_deducts_wood_stock(mini_model):
    """_do_expand deducts expand_wood_cost from wood_stock when a tile is claimed."""
    cities = _get_cities(mini_model)
    city = cities[0]
    cfg = mini_model.config

    city.wood_stock = 50.0
    wood_before = city.wood_stock
    territory_before = mini_model.grid.territory_count(city.civ.civ_id)

    city._do_expand()

    assert mini_model.grid.territory_count(city.civ.civ_id) > territory_before, (
        "expect unclaimed tiles on 20x20 grid"
    )
    assert city.wood_stock == pytest.approx(wood_before - cfg.expand_wood_cost)


def test_do_expand_no_op_when_all_tiles_claimed(mini_model):
    """If every tile in radius 3 is already owned, territory count stays the same."""
    cities = _get_cities(mini_model)
    city = cities[0]
    grid = mini_model.grid

    # Claim every tile within radius 3 for this civ
    r = 3
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            nx, ny = city.x + dx, city.y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                grid.claim(nx, ny, city.civ.civ_id)

    before = grid.territory_count(city.civ.civ_id)
    city._do_expand()
    after = grid.territory_count(city.civ.civ_id)
    assert after == before, (
        f"territory should not change when no unclaimed tiles exist; was {before}, now {after}"
    )


def test_do_expand_no_deduction_when_no_tile(mini_model):
    """_do_expand does not deduct wood_stock if no unclaimed tile is reachable."""
    cities = _get_cities(mini_model)
    city = cities[0]
    grid = mini_model.grid

    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx, ny = city.x + dx, city.y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                grid.claim(nx, ny, city.civ.civ_id)

    city.wood_stock = 50.0
    wood_before = city.wood_stock
    city._do_expand()
    assert city.wood_stock == wood_before


# ---------------------------------------------------------------------------
# _do_fortify
# ---------------------------------------------------------------------------

def test_do_fortify_increases_military(mini_model):
    """Fortifying with stockpiled minerals and wood increases military."""
    cities = _get_cities(mini_model)
    city = cities[0]

    # Cities start with initial_wood_stock and initial_mineral_stock from config,
    # both above the fortify cost thresholds.
    mil_before = city.military
    city._do_fortify()
    assert city.military >= mil_before, (
        f"military should not decrease after _do_fortify; was {mil_before}, now {city.military}"
    )
    assert city.military > mil_before, (
        f"military should increase when stockpiles are available; was {mil_before}, now {city.military}"
    )


def test_do_fortify_proportional_with_low_stock(mini_model):
    """Military gain equals int((consumed_m + consumed_w) / 2) with partial stocks."""
    cities = _get_cities(mini_model)
    city = cities[0]

    city.mineral_stock = 4.0   # less than fortify_mineral_cost (8.0)
    city.wood_stock = 2.0      # less than fortify_wood_cost (4.0)
    mil_before = city.military

    city._do_fortify()

    expected_gain = int((4.0 + 2.0) / 2)  # 3
    assert city.military == mil_before + expected_gain


# ---------------------------------------------------------------------------
# _do_attack
# ---------------------------------------------------------------------------

def test_do_attack_on_win_reduces_enemy_military(mini_model):
    """With overwhelming military advantage the attacker wins and enemy takes losses."""
    cities = _get_cities(mini_model)
    assert len(cities) >= 2, "mini_model must have 2 cities for attack tests"

    # Pick attacker and defender from different civs
    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    # Place them close enough (within 10 tiles Manhattan distance)
    defender.x = attacker.x + 5
    defender.y = attacker.y
    defender.cell = mini_model.grid.cell(defender.x, defender.y)

    # Give attacker overwhelming military, defender minimal
    attacker.military = 1000
    defender.military = 1

    enemy_mil_before = defender.military

    # Force win by patching model.random.random to return 0.0 (< any win_prob)
    with patch.object(mini_model.random, "random", return_value=0.0):
        attacker._do_attack()

    assert defender.military < enemy_mil_before, (
        f"enemy military should decrease on attacker win; was {enemy_mil_before}, now {defender.military}"
    )


def test_do_attack_on_loss_reduces_attacker_military(mini_model):
    """When attacker loses the battle, its own military is reduced."""
    cities = _get_cities(mini_model)
    assert len(cities) >= 2, "mini_model must have 2 cities for attack tests"

    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    # Place them close enough
    defender.x = attacker.x + 5
    defender.y = attacker.y
    defender.cell = mini_model.grid.cell(defender.x, defender.y)

    attacker.military = 10
    defender.military = 10

    attacker_mil_before = attacker.military

    # Force loss by patching model.random.random to return 1.0 (> any win_prob)
    with patch.object(mini_model.random, "random", return_value=1.0):
        attacker._do_attack()

    assert attacker.military < attacker_mil_before, (
        f"attacker military should decrease on loss; was {attacker_mil_before}, now {attacker.military}"
    )


# ---------------------------------------------------------------------------
# _do_trade
# ---------------------------------------------------------------------------

def test_do_trade_no_op_when_no_nearby_city(mini_model):
    """With no other cities reachable, _do_trade should not change food_stock."""
    cities = _get_cities(mini_model)
    city = cities[0]

    # Put plenty of food on the tile so the surplus check passes
    mini_model.grid.deposit(city.x, city.y, ResourceType.FOOD, 80.0)

    food_before = city.food_stock

    # Patch _agents_by_type so agents_by_type returns only this city (no trade partners)
    with patch.object(
        type(mini_model),
        "agents_by_type",
        new_callable=lambda: property(lambda self: {CityAgent: [city]}),
    ):
        city._do_trade()

    assert city.food_stock == food_before, (
        f"food_stock should be unchanged when no trade partner exists; "
        f"was {food_before}, now {city.food_stock}"
    )


# ---------------------------------------------------------------------------
# _do_research
# ---------------------------------------------------------------------------

def test_do_research_calls_tech_engine(mini_model):
    """_do_research should delegate to model.tech_engine.check(city)."""
    cities = _get_cities(mini_model)
    city = cities[0]

    mock_check = MagicMock()
    with patch.object(mini_model.tech_engine, "check", mock_check):
        city._do_research()

    mock_check.assert_called_once_with(city)


def test_do_research_deducts_wood_and_mineral_stock(mini_model):
    """_do_research deducts wood_stock and mineral_stock as research costs."""
    cities = _get_cities(mini_model)
    city = cities[0]
    city.wood_stock = 50.0
    city.mineral_stock = 50.0
    wood_before = city.wood_stock
    mineral_before = city.mineral_stock

    with patch.object(mini_model.tech_engine, "check", MagicMock()):
        city._do_research()

    assert city.wood_stock < wood_before
    assert city.mineral_stock < mineral_before


# ---------------------------------------------------------------------------
# _do_attack — stockpile interactions
# ---------------------------------------------------------------------------

def test_do_attack_deducts_mineral_stock(mini_model):
    """Attack always deducts mineral_stock as ammo cost, even with no target in range."""
    cities = _get_cities(mini_model)
    attacker = cities[0]
    attacker.mineral_stock = 50.0
    mineral_before = attacker.mineral_stock
    attacker._do_attack()
    assert attacker.mineral_stock < mineral_before


def test_do_attack_on_win_raids_target_stocks(mini_model):
    """On attacker victory, the defender's wood and mineral stocks are pillaged."""
    cities = _get_cities(mini_model)
    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    defender.x = attacker.x + 5
    defender.y = attacker.y
    defender.cell = mini_model.grid.cell(defender.x, defender.y)
    attacker.military = 1000
    defender.military = 1
    defender.wood_stock = 100.0
    defender.mineral_stock = 80.0

    wood_before = defender.wood_stock
    mineral_before = defender.mineral_stock

    with patch.object(mini_model.random, "random", return_value=0.0):
        attacker._do_attack()

    assert defender.wood_stock < wood_before
    assert defender.mineral_stock < mineral_before


def test_do_attack_win_fortified_city_less_pillage(mini_model):
    """A defender with high military loses fewer stockpile resources than an undefended one."""
    cities = _get_cities(mini_model)
    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    defender.x = attacker.x + 5
    defender.y = attacker.y
    defender.cell = mini_model.grid.cell(defender.x, defender.y)
    attacker.military = 1000

    # Undefended: low military → damage_factor ≈ 1.0 → high pillage
    defender.military = 1
    defender.wood_stock = 100.0
    with patch.object(mini_model.random, "random", return_value=0.0):
        attacker._do_attack()
    wood_lost_undefended = 100.0 - defender.wood_stock

    # Fortified: high military → damage_factor = 0.2 → low pillage
    defender.military = 100  # max_defense_military default
    defender.wood_stock = 100.0
    with patch.object(mini_model.random, "random", return_value=0.0):
        attacker._do_attack()
    wood_lost_fortified = 100.0 - defender.wood_stock

    assert wood_lost_fortified < wood_lost_undefended


# ---------------------------------------------------------------------------
# _capture_city — stock transfer and reconstruction costs
# ---------------------------------------------------------------------------

def test_capture_city_transfers_stocks_to_attacker(mini_model):
    """_capture_city zeroes target's stocks and transfers them to the attacker."""
    cities = _get_cities(mini_model)
    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    defender.food_stock = 50.0
    defender.wood_stock = 40.0
    defender.mineral_stock = 30.0
    attacker.food_stock = 0.0
    attacker.wood_stock = 0.0
    attacker.mineral_stock = 0.0

    attacker._capture_city(defender, damage_factor=0.0)  # zero cost to reconstruct

    assert defender.food_stock == 0.0
    assert defender.wood_stock == 0.0
    assert defender.mineral_stock == 0.0
    assert attacker.food_stock == pytest.approx(50.0)
    assert attacker.wood_stock == pytest.approx(40.0)
    assert attacker.mineral_stock == pytest.approx(30.0)


def test_capture_city_deducts_reconstruction_costs(mini_model):
    """_capture_city deducts reconstruction wood and mineral from the attacker."""
    cities = _get_cities(mini_model)
    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)
    cfg = mini_model.config

    defender.food_stock = 0.0
    defender.wood_stock = 0.0
    defender.mineral_stock = 0.0
    attacker.food_stock = 0.0
    attacker.wood_stock = 100.0
    attacker.mineral_stock = 100.0

    attacker._capture_city(defender, damage_factor=1.0)

    assert attacker.wood_stock == pytest.approx(100.0 - cfg.capture_reconstruct_wood)
    assert attacker.mineral_stock == pytest.approx(100.0 - cfg.capture_reconstruct_mineral)


def test_capture_city_reconstruction_cheaper_for_fortified(mini_model):
    """Capturing with damage_factor=0.2 costs less reconstruction than damage_factor=1.0."""
    cities = _get_cities(mini_model)
    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    defender.food_stock = 0.0
    defender.wood_stock = 0.0
    defender.mineral_stock = 0.0

    attacker.wood_stock = 100.0
    attacker.mineral_stock = 100.0
    attacker._capture_city(defender, damage_factor=1.0)
    wood_after_full = attacker.wood_stock

    attacker.wood_stock = 100.0
    attacker.mineral_stock = 100.0
    attacker._capture_city(defender, damage_factor=0.2)
    wood_after_fortified = attacker.wood_stock

    assert wood_after_fortified > wood_after_full


# ---------------------------------------------------------------------------
# _do_gather — stockpile harvesting
# ---------------------------------------------------------------------------

def test_do_gather_increases_wood_and_mineral_stock(mini_model):
    """_do_gather harvests wood and minerals from claimed tiles in radius."""
    cities = _get_cities(mini_model)
    city = cities[0]
    grid = mini_model.grid

    grid.deposit(city.x, city.y, ResourceType.WOOD, 50.0)
    grid.deposit(city.x, city.y, ResourceType.MINERALS, 50.0)
    grid.claim(city.x, city.y, city.civ.civ_id)

    wood_before = city.wood_stock
    mineral_before = city.mineral_stock
    city._do_gather()

    assert city.wood_stock > wood_before
    assert city.mineral_stock > mineral_before

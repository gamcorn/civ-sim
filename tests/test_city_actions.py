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


# ---------------------------------------------------------------------------
# _do_fortify
# ---------------------------------------------------------------------------

def test_do_fortify_increases_fortification(mini_model):
    """Fortifying with minerals + wood increases city.fortification, not military."""
    cities = _get_cities(mini_model)
    city = cities[0]
    city.mineral_stock = 50.0
    city.wood_stock = 50.0

    fort_before = city.fortification
    mil_before = city.military
    city._do_fortify()

    assert city.fortification > fort_before, (
        f"fortification should increase after _do_fortify; was {fort_before:.1f}"
    )
    assert city.military == mil_before, (
        f"military must NOT change on _do_fortify; was {mil_before}, now {city.military}"
    )


def test_fortification_reduces_pillage_damage(mini_config):
    """Attacker pillages less from a city with high fortification vs zero."""
    from civ_sim.simulation.model import CivModel

    def _run(fortification_level: float) -> float:
        m = CivModel(mini_config)
        cities = [a for a in m.agents if isinstance(a, CityAgent)]
        attacker, defender = cities[0], cities[1]
        attacker.military = 1000
        attacker.mineral_stock = 50.0
        defender.military = 1
        defender.fortification = fortification_level
        defender.food_stock = 200.0
        m.random.seed(0)
        food_before = defender.food_stock
        attacker._do_attack()
        return food_before - defender.food_stock

    loss_no_fort   = _run(0.0)
    loss_with_fort = _run(mini_config.max_fortification)

    assert loss_with_fort < loss_no_fort, (
        f"Fortification should reduce pillage; no_fort_loss={loss_no_fort:.1f} fort_loss={loss_with_fort:.1f}"
    )


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


def test_attacker_loses_troops_on_successful_attack(mini_model):
    """The attacker must take casualties even when it wins."""
    from civ_sim.agents.city import CityAgent
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    assert len(cities) >= 2, "Need 2 cities for combat test"
    attacker = cities[0]
    defender = cities[1]
    # Overwhelming force guarantees a win
    attacker.military = 200
    defender.military = 1
    attacker.mineral_stock = 50.0

    mil_before = attacker.military
    mini_model.random.seed(0)
    attacker._do_attack()

    assert attacker.military < mil_before, (
        f"Winning attacker should lose some troops; had {mil_before}, now has {attacker.military}"
    )


def test_trade_transfers_stockpiles_not_tile_resources(mini_model):
    """Trade must move food_stock to receiver and get mineral_stock back."""
    from civ_sim.agents.city import CityAgent
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    assert len(cities) >= 2
    sender = cities[0]
    receiver = cities[1]

    # Sender has food surplus; receiver has mineral surplus
    daily_need = sender.population * mini_model.config.food_per_person
    sender.food_stock = daily_need * 30   # large surplus (well above 10-tick buffer)
    sender.mineral_stock = 0.0

    receiver.food_stock = 0.0
    receiver.mineral_stock = 200.0

    sender_food_before = sender.food_stock
    receiver_food_before = receiver.food_stock
    sender_mineral_before = sender.mineral_stock
    receiver_mineral_before = receiver.mineral_stock

    sender._do_trade()

    assert sender.food_stock < sender_food_before, "Sender should have sent food from stockpile"
    assert receiver.food_stock > receiver_food_before, "Receiver should have received food in stockpile"
    assert sender.mineral_stock > sender_mineral_before, "Sender should have received minerals"
    assert receiver.mineral_stock < receiver_mineral_before, "Receiver should have paid minerals"


def test_trade_aborts_when_receiver_has_no_mineral_surplus(mini_model):
    """Trade should not execute if the receiver cannot pay minerals."""
    from civ_sim.agents.city import CityAgent
    cities = [a for a in mini_model.agents if isinstance(a, CityAgent)]
    sender = cities[0]
    receiver = cities[1]

    daily_need = sender.population * mini_model.config.food_per_person
    sender.food_stock = daily_need * 30   # sender has surplus
    receiver.mineral_stock = 0.0          # receiver is broke

    sender_food_before = sender.food_stock
    sender._do_trade()

    assert sender.food_stock == sender_food_before, (
        "Trade should abort when receiver has no mineral surplus"
    )


def test_gather_output_scales_with_population(mini_model):
    """A large-pop city must harvest more than a tiny-pop city given the same territory."""
    from civ_sim.world.resources import ResourceType
    cities = _get_cities(mini_model)
    city = cities[0]

    # Own every tile in harvest radius so neither city is tile-limited
    r = mini_model.config.harvest_radius
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            nx, ny = city.x + dx, city.y + dy
            if 0 <= nx < mini_model.grid.width and 0 <= ny < mini_model.grid.height:
                mini_model.grid.claim(nx, ny, city.civ.civ_id)
                mini_model.grid.layers[ResourceType.FOOD].data[nx, ny] = 80.0
                mini_model.grid.layers[ResourceType.WOOD].data[nx, ny] = 80.0
                mini_model.grid.layers[ResourceType.MINERALS].data[nx, ny] = 80.0

    # Small pop
    city.population = 5
    city.food_stock = 0.0
    city.wood_stock = 0.0
    city.mineral_stock = 0.0
    city._do_gather()
    small_food = city.food_stock
    small_wood = city.wood_stock

    # Restore tile resources
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            nx, ny = city.x + dx, city.y + dy
            if 0 <= nx < mini_model.grid.width and 0 <= ny < mini_model.grid.height:
                mini_model.grid.layers[ResourceType.FOOD].data[nx, ny] = 80.0
                mini_model.grid.layers[ResourceType.WOOD].data[nx, ny] = 80.0
                mini_model.grid.layers[ResourceType.WATER].data[nx, ny] = 80.0
                mini_model.grid.layers[ResourceType.MINERALS].data[nx, ny] = 80.0

    # Large pop
    city.population = 500
    city.food_stock = 0.0
    city.wood_stock = 0.0
    city.mineral_stock = 0.0
    city._do_gather()
    large_food = city.food_stock
    large_wood = city.wood_stock

    assert large_food > small_food * 2, (
        f"Large-pop city should harvest much more food; small={small_food:.1f} large={large_food:.1f}"
    )
    assert large_wood > small_wood * 2, (
        f"Large-pop city should harvest much more wood; small={small_wood:.1f} large={large_wood:.1f}"
    )


def test_gather_zero_pop_yields_nothing(mini_model):
    """A city with 0 population should harvest nothing (no labor available)."""
    from civ_sim.world.resources import ResourceType
    cities = _get_cities(mini_model)
    city = cities[0]

    r = mini_model.config.harvest_radius
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            nx, ny = city.x + dx, city.y + dy
            if 0 <= nx < mini_model.grid.width and 0 <= ny < mini_model.grid.height:
                mini_model.grid.claim(nx, ny, city.civ.civ_id)
                mini_model.grid.layers[ResourceType.FOOD].data[nx, ny] = 80.0

    city.population = 0
    city.food_stock = 0.0
    city._do_gather()
    assert city.food_stock == 0.0, "Zero-pop city must not harvest any food"


# ---------------------------------------------------------------------------
# _do_recruit
# ---------------------------------------------------------------------------

def test_do_recruit_increases_military_and_reduces_population(mini_model):
    """Recruiting must increase military and reduce population by the drafted amount."""
    cities = _get_cities(mini_model)
    city = cities[0]
    city.population = 200
    city.mineral_stock = 50.0

    pop_before = city.population
    mil_before = city.military
    city._do_recruit()

    assert city.military > mil_before, (
        f"military should increase after recruit; was {mil_before}, now {city.military}"
    )
    assert city.population < pop_before, (
        f"population should decrease after recruit; was {pop_before}, now {city.population}"
    )


def test_do_recruit_does_nothing_when_population_too_low(mini_model):
    """Recruit must not execute when population is at the initial_pop floor."""
    cities = _get_cities(mini_model)
    city = cities[0]
    city.population = mini_model.config.initial_pop  # at the floor
    city.mineral_stock = 50.0

    pop_before = city.population
    mil_before = city.military
    city._do_recruit()

    assert city.military == mil_before, "Military must not change when population is at floor"
    assert city.population == pop_before, "Population must not change when at floor"

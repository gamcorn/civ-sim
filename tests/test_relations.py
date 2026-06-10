"""Tests for the civilization relations score system."""
from __future__ import annotations
import pytest
from civ_sim.agents.city import CityAgent


def _get_cities(model):
    return [a for a in model.agents if isinstance(a, CityAgent)]


# --- model.get_relation / update_relation ---

def test_get_relation_returns_zero_for_unknown_pair(mini_model):
    civs = mini_model.civilizations
    assert mini_model.get_relation(civs[0].civ_id, civs[1].civ_id) == 0.0


def test_update_relation_stores_and_retrieves_value(mini_model):
    civs = mini_model.civilizations
    mini_model.update_relation(civs[0].civ_id, civs[1].civ_id, 0.3)
    assert abs(mini_model.get_relation(civs[0].civ_id, civs[1].civ_id) - 0.3) < 1e-9


def test_get_relation_is_symmetric(mini_model):
    civs = mini_model.civilizations
    mini_model.update_relation(civs[0].civ_id, civs[1].civ_id, 0.4)
    assert mini_model.get_relation(civs[1].civ_id, civs[0].civ_id) == mini_model.get_relation(civs[0].civ_id, civs[1].civ_id)


def test_update_relation_clamps_to_positive_one(mini_model):
    civs = mini_model.civilizations
    mini_model.update_relation(civs[0].civ_id, civs[1].civ_id, 5.0)
    assert mini_model.get_relation(civs[0].civ_id, civs[1].civ_id) == 1.0


def test_update_relation_clamps_to_negative_one(mini_model):
    civs = mini_model.civilizations
    mini_model.update_relation(civs[0].civ_id, civs[1].civ_id, -5.0)
    assert mini_model.get_relation(civs[0].civ_id, civs[1].civ_id) == -1.0


# --- trade wires relations ---

def test_trade_increases_relations(mini_model):
    cities = _get_cities(mini_model)
    assert len(cities) >= 2
    sender = cities[0]
    receiver = cities[1]

    # Set up a trade-able surplus
    daily_need = sender.population * mini_model.config.food_per_person
    sender.food_stock = daily_need * 30
    receiver.mineral_stock = 200.0

    rel_before = mini_model.get_relation(sender.civ.civ_id, receiver.civ.civ_id)
    sender._do_trade()
    rel_after = mini_model.get_relation(sender.civ.civ_id, receiver.civ.civ_id)

    assert rel_after > rel_before, f"Trade should increase relations; before={rel_before}, after={rel_after}"


# --- attack wires relations ---

def test_attack_decreases_relations(mini_model):
    from unittest.mock import patch
    cities = _get_cities(mini_model)
    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    defender.x = attacker.x + 5
    defender.y = attacker.y
    defender.cell = mini_model.grid.cell(defender.x, defender.y)
    attacker.military = 500
    attacker.mineral_stock = 50.0

    rel_before = mini_model.get_relation(attacker.civ.civ_id, defender.civ.civ_id)
    attacker._do_attack()
    rel_after = mini_model.get_relation(attacker.civ.civ_id, defender.civ.civ_id)

    assert rel_after < rel_before, f"Attack should decrease relations; before={rel_before}, after={rel_after}"


# --- trade blocked by low relations ---

def test_trade_blocked_when_relations_below_threshold(mini_model):
    cities = _get_cities(mini_model)
    sender = cities[0]
    receiver = next(c for c in cities if c.civ.civ_id != sender.civ.civ_id)

    # Drive relations well below threshold
    mini_model.update_relation(sender.civ.civ_id, receiver.civ.civ_id, -1.0)

    daily_need = sender.population * mini_model.config.food_per_person
    sender.food_stock = daily_need * 30
    receiver.mineral_stock = 200.0

    food_before = sender.food_stock
    sender._do_trade()
    assert sender.food_stock == food_before, "Trade should be blocked when relations are very negative"


# --- relation decay ---

def test_relation_decay_moves_positive_score_toward_zero(mini_model):
    civs = mini_model.civilizations
    mini_model.update_relation(civs[0].civ_id, civs[1].civ_id, 0.5)

    rel_before = mini_model.get_relation(civs[0].civ_id, civs[1].civ_id)
    mini_model.step()
    rel_after = mini_model.get_relation(civs[0].civ_id, civs[1].civ_id)

    assert rel_after < rel_before, f"Positive relation should decay toward 0; before={rel_before}, after={rel_after}"
    assert rel_after >= 0.0, "Positive relation must not decay below 0"


def test_relation_decay_moves_negative_score_toward_zero(mini_model):
    civs = mini_model.civilizations
    mini_model.update_relation(civs[0].civ_id, civs[1].civ_id, -0.5)

    rel_before = mini_model.get_relation(civs[0].civ_id, civs[1].civ_id)
    mini_model.step()
    rel_after = mini_model.get_relation(civs[0].civ_id, civs[1].civ_id)

    assert rel_after > rel_before, f"Negative relation should decay toward 0; before={rel_before}, after={rel_after}"
    assert rel_after <= 0.0, "Negative relation must not decay above 0"

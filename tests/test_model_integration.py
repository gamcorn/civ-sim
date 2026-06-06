"""Integration tests for CivModel (simulation/model.py)."""
from __future__ import annotations

import pytest
from config import SimConfig
from simulation.model import CivModel
from agents.city import CityAgent


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------

def test_constructor_places_correct_number_of_cities(mini_model):
    """With num_civs=2 and cities_per_civ=1, exactly 2 CityAgents should exist."""
    city_count = sum(1 for a in mini_model.agents if isinstance(a, CityAgent))
    assert city_count == 2


def test_constructor_creates_two_civilizations(mini_model):
    """CivModel should create exactly 2 Civilization objects."""
    assert len(mini_model.civilizations) == 2


# ---------------------------------------------------------------------------
# Step tests
# ---------------------------------------------------------------------------

def test_step_increments_steps(mini_model):
    """Each call to model.step() should increment model.steps by 1."""
    before = mini_model.steps
    mini_model.step()
    assert mini_model.steps == before + 1


def test_multi_tick_run_completes_without_exception(mini_config):
    """Running 5 ticks on a fresh model should not raise any exception."""
    model = CivModel(mini_config)
    try:
        for _ in range(5):
            if model.running:
                model.step()
    finally:
        try:
            model.logger.close()
        except Exception:
            pass


def test_model_stops_at_max_ticks(mini_config):
    """Model should set running=False once max_ticks is reached."""
    mini_config.max_ticks = 3
    model = CivModel(mini_config)
    try:
        for _ in range(5):
            if model.running:
                model.step()
        assert model.running is False
    finally:
        try:
            model.logger.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# History tests
# ---------------------------------------------------------------------------

def test_history_grows_by_one_entry_per_step(mini_model):
    """Each step() should append exactly one entry to history['tick']."""
    before = len(mini_model.history["tick"])
    mini_model.step()
    assert len(mini_model.history["tick"]) == before + 1


def test_history_tracks_pop_for_both_civs(mini_model):
    """After step(), history should contain 'pop_0' and 'pop_1' with at least 1 entry."""
    mini_model.step()
    assert "pop_0" in mini_model.history
    assert "pop_1" in mini_model.history
    assert len(mini_model.history["pop_0"]) >= 1
    assert len(mini_model.history["pop_1"]) >= 1


# ---------------------------------------------------------------------------
# Aggregate tests
# ---------------------------------------------------------------------------

def test_civilization_aggregates_updated_after_step(mini_model):
    """After step(), each civilization's total_pop and total_military should be >= 0."""
    mini_model.step()
    for civ in mini_model.civilizations:
        assert civ.total_pop >= 0
        assert civ.total_military >= 0


# ---------------------------------------------------------------------------
# Stop-condition tests
# ---------------------------------------------------------------------------

def test_model_stops_when_one_civ_eliminated(mini_model):
    """If all cities of one civilization are eliminated, model.running becomes False."""
    # Find the cities belonging to civilization at index 1
    target_civ = mini_model.civilizations[1]
    target_cities = [
        a for a in mini_model.agents
        if isinstance(a, CityAgent) and a.civ is target_civ
    ]
    assert target_cities, "Need at least one city in civ 1 to eliminate"

    # Drive population and food to 0 so _consume_resources triggers _collapse
    for city in target_cities:
        city.population = 0
        city.food_stock = 0.0

    # One step should detect the eliminated civ
    mini_model.step()

    alive_civs = [c for c in mini_model.civilizations if c.alive]
    assert len(alive_civs) <= 1
    assert mini_model.running is False

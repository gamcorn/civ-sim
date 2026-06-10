# tests/providers/test_council_prompts.py
import pytest
from unittest.mock import MagicMock
from tests.conftest import make_mock_city
from agents.providers.council_prompts import (
    MINISTER_SPECS,
    build_sector_persona,
    build_chief_persona,
    build_civ_state_snapshot,
    build_sector_user_message,
)


def make_traits(aggressiveness=0.5, trust=0.5, innovation=0.5, tribalism=0.5, risk_tolerance=0.5):
    t = MagicMock()
    t.aggressiveness = aggressiveness
    t.trust = trust
    t.innovation = innovation
    t.tribalism = tribalism
    t.risk_tolerance = risk_tolerance
    return t


def test_minister_specs_has_four_sectors():
    assert len(MINISTER_SPECS) == 4
    roles = [s["name"] for s in MINISTER_SPECS]
    assert "Minister of War" in roles
    assert "Minister of Economy" in roles
    assert "Minister of Science" in roles
    assert "Minister of Expansion" in roles


def test_sector_persona_hawkish_when_high_aggressiveness():
    traits = make_traits(aggressiveness=0.9)
    war_spec = next(s for s in MINISTER_SPECS if s["name"] == "Minister of War")
    persona = build_sector_persona(war_spec, traits)
    assert "strongly" in persona


def test_sector_persona_cautious_when_low_aggressiveness():
    traits = make_traits(aggressiveness=0.1)
    war_spec = next(s for s in MINISTER_SPECS if s["name"] == "Minister of War")
    persona = build_sector_persona(war_spec, traits)
    assert "cautiously" in persona


def test_chief_persona_bold_when_high_risk_tolerance():
    traits = make_traits(risk_tolerance=0.9)
    persona = build_chief_persona(traits)
    assert "bold" in persona


def test_build_civ_state_snapshot_contains_key_fields():
    city = make_mock_city(pop=200, military=30, food_stock=100.0, tick=15)
    snapshot = build_civ_state_snapshot(city.civ, [city], city.model)
    assert "Turn: 15" in snapshot
    assert "200" in snapshot   # population
    assert "30" in snapshot    # military


def test_build_civ_state_snapshot_contains_enemy_intel():
    city = make_mock_city(pop=200, military=30, food_stock=100.0, tick=15)
    snapshot = build_civ_state_snapshot(city.civ, [city], city.model)
    assert "Intelligence Report:" in snapshot
    assert "Beta" in snapshot


def test_build_civ_state_snapshot_fog_marks_approximate():
    city = make_mock_city(pop=200, military=30, food_stock=100.0, tick=15)
    city.model.random.uniform.return_value = 1.2
    snapshot = build_civ_state_snapshot(city.civ, [city], city.model, fog_of_war=0.5)
    assert "fog=50%" in snapshot
    assert "~" in snapshot


def test_build_sector_user_message_includes_state():
    city = make_mock_city()
    snapshot = build_civ_state_snapshot(city.civ, [city], city.model)
    spec = MINISTER_SPECS[0]
    msg = build_sector_user_message(spec, snapshot)
    assert "Civilization State" in msg
    assert spec["actions"][0] in msg

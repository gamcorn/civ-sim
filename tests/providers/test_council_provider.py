# tests/providers/test_council_provider.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from civ_sim.agents.decisions import ALL_ACTIONS
from tests.conftest import make_mock_city


def make_provider_config(**kwargs):
    from civ_sim.config import ProviderConfig

    defaults = dict(
        type="council",
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        timeout=5.0,
        directive_period=10,
        max_rounds=1,
        emergency_triggers=True,
        emergency_cooldown_ticks=3,
        sector_model="",
        chief_model="",
    )
    defaults.update(kwargs)
    return ProviderConfig(**defaults)


def make_directive(issued_at_tick=0, valid_for_ticks=10, emergency=False):
    from civ_sim.agents.providers.council_provider import StrategicDirective

    return StrategicDirective(
        era_goal="Hold steady",
        action_weights={a: 0.0 for a in ALL_ACTIONS},
        reasoning="balanced",
        issued_at_tick=issued_at_tick,
        valid_for_ticks=valid_for_ticks,
        emergency=emergency,
    )


# -- _should_run_council --


def test_should_run_council_when_no_directive():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config())
    city = make_mock_city()
    assert p._should_run_council(city.civ, [city], 0) is True


def test_should_run_council_on_cadence():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config(directive_period=10))
    p._directive = make_directive(issued_at_tick=0)
    p._last_council_tick = 0
    city = make_mock_city(tick=10)
    assert p._should_run_council(city.civ, [city], 10) is True


def test_should_not_run_council_mid_period():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config(directive_period=10))
    p._directive = make_directive(issued_at_tick=0)
    p._last_council_tick = 0
    city = make_mock_city(tick=5, food_stock=200.0)
    # Ensure no emergency triggers fire: food high, enemy weak
    city.civ.total_military = 100
    city.civ._pop_at_last_directive = 200
    city.civ.total_pop = 200
    city.civ._techs_at_last_directive = 0
    city.civ._city_count_at_last_directive = 1
    assert p._should_run_council(city.civ, [city], 5) is False


# -- _check_emergencies --


def test_emergency_food_crisis():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config())
    p._directive = make_directive()
    p._last_council_tick = 0
    city = make_mock_city(food_stock=5.0, tick=5)  # < 10.0
    city.civ._city_count_at_last_directive = 1
    city.civ._pop_at_last_directive = 200
    city.civ.total_pop = 200
    city.civ._techs_at_last_directive = 0
    assert p._check_emergencies(city.civ, [city]) is True


def test_emergency_military_threat():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config())
    city = make_mock_city(food_stock=200.0, tick=5)
    city.civ.total_military = 10
    city.civ._city_count_at_last_directive = 1
    city.civ._pop_at_last_directive = 200
    city.civ.total_pop = 200
    city.civ._techs_at_last_directive = 0
    # Enemy military set to 30 (> 2 * 10 = 20)
    city.model.civilizations[1].total_military = 30
    assert p._check_emergencies(city.civ, [city]) is True


def test_emergency_city_lost():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config())
    city = make_mock_city(food_stock=200.0, tick=5)
    city.civ.total_military = 100
    city.civ._pop_at_last_directive = 200
    city.civ.total_pop = 200
    city.civ._techs_at_last_directive = 0
    city.civ._city_count_at_last_directive = 3  # had 3 cities
    # now only 1 city passed in → city lost
    assert p._check_emergencies(city.civ, [city]) is True


def test_emergency_cooldown_prevents_re_trigger():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(
        make_provider_config(directive_period=10, emergency_cooldown_ticks=3)
    )
    p._directive = make_directive()
    p._last_council_tick = 0
    p._last_emergency_tick = 4  # recent emergency at tick 4
    city = make_mock_city(food_stock=5.0, tick=5)  # food crisis would trigger
    city.civ._city_count_at_last_directive = 1
    city.civ._pop_at_last_directive = 200
    city.civ.total_pop = 200
    city.civ._techs_at_last_directive = 0
    # tick 5 - last_emergency 4 = 1 < cooldown 3 → should not trigger
    assert p._should_run_council(city.civ, [city], 5) is False


# -- _apply_directive --


def test_apply_directive_selects_max_weighted_feasible_action():
    from civ_sim.agents.providers.council_provider import (
        CouncilProvider,
        StrategicDirective,
    )

    p = CouncilProvider(make_provider_config())
    # Give a huge boost to fortify (always feasible)
    p._directive = StrategicDirective(
        era_goal="Fortify",
        action_weights={a: -1.0 for a in ALL_ACTIONS} | {"fortify": 2.0},
        reasoning="defensive",
        issued_at_tick=0,
        valid_for_ticks=10,
    )
    city = make_mock_city()
    result = p._apply_directive(city)
    assert result == "fortify"


# -- choose_actions_batch --


@pytest.mark.asyncio
async def test_choose_actions_batch_uses_existing_directive():
    """Directive within period → no council call."""
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config(directive_period=10))
    p._directive = make_directive(issued_at_tick=5)
    p._last_council_tick = 5
    p._client = MagicMock()
    city = make_mock_city(tick=8, food_stock=200.0)
    city.civ.total_military = 100
    city.civ._pop_at_last_directive = 200
    city.civ.total_pop = 200
    city.civ._techs_at_last_directive = 0
    city.civ._city_count_at_last_directive = 1

    result = await p.choose_actions_batch([city])

    assert len(result) == 1
    assert result[0] in ALL_ACTIONS
    p._client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_choose_actions_batch_falls_back_when_no_directive_and_llm_fails():
    """No directive + LLM error → rule_based fallback."""
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config())
    p._client = MagicMock()
    p._client.chat.completions.create = AsyncMock(side_effect=Exception("LLM down"))
    city = make_mock_city()
    city.model.logger = MagicMock()

    result = await p.choose_actions_batch([city])

    assert len(result) == 1
    assert result[0] in ALL_ACTIONS


@pytest.mark.asyncio
async def test_choose_actions_batch_empty_cities():
    from civ_sim.agents.providers.council_provider import CouncilProvider

    p = CouncilProvider(make_provider_config())
    result = await p.choose_actions_batch([])
    assert result == []

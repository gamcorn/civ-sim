# tests/providers/test_council_ministers.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock


def make_mock_llm_response(content: str):
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


def make_async_client(content: str):
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        return_value=make_mock_llm_response(content)
    )
    return client


@pytest.mark.asyncio
async def test_call_sector_minister_returns_structured_output():
    from agents.providers.council_ministers import call_sector_minister
    from agents.providers.council_prompts import MINISTER_SPECS

    content = json.dumps({
        "analysis": "Enemy is weak",
        "recommendation": "Attack east flank",
        "weight_requests": {"attack": 0.8, "fortify": 0.2},
        "disagreement_level": 0.1,
    })
    traits = MagicMock()
    traits.aggressiveness = 0.8
    client = make_async_client(content)

    result = await call_sector_minister(
        MINISTER_SPECS[0], "state...", traits, client, "test-model", 5.0
    )

    assert result["name"] == "Minister of War"
    assert result["recommendation"] == "Attack east flank"
    assert result["weight_requests"]["attack"] == 0.8


@pytest.mark.asyncio
async def test_call_sector_minister_fallback_on_bad_json():
    from agents.providers.council_ministers import call_sector_minister
    from agents.providers.council_prompts import MINISTER_SPECS

    traits = MagicMock()
    traits.aggressiveness = 0.5
    client = make_async_client("not json at all")

    result = await call_sector_minister(
        MINISTER_SPECS[0], "state...", traits, client, "test-model", 5.0
    )

    assert result["name"] == "Minister of War"
    assert result["weight_requests"] == {}
    assert result["disagreement_level"] == 0.0


@pytest.mark.asyncio
async def test_call_sector_minister_fallback_on_timeout():
    from agents.providers.council_ministers import call_sector_minister
    from agents.providers.council_prompts import MINISTER_SPECS
    import asyncio

    traits = MagicMock()
    traits.aggressiveness = 0.5
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=asyncio.TimeoutError())

    result = await call_sector_minister(
        MINISTER_SPECS[0], "state...", traits, client, "test-model", 0.001
    )

    assert result["weight_requests"] == {}


@pytest.mark.asyncio
async def test_call_budget_minister_returns_structured_output():
    from agents.providers.council_ministers import call_budget_minister

    content = json.dumps({
        "veto": False,
        "veto_reason": None,
        "approved_weights": {"attack": 0.6},
    })
    traits = MagicMock()
    traits.risk_tolerance = 0.5
    client = make_async_client(content)

    result = await call_budget_minister("state...", [], traits, client, "test-model", 5.0)

    assert result["veto"] is False
    assert result["approved_weights"]["attack"] == 0.6


@pytest.mark.asyncio
async def test_call_chief_returns_parsed_directive():
    from agents.providers.council_ministers import call_chief
    from agents.decisions import ALL_ACTIONS

    content = json.dumps({
        "era_goal": "Dominate the east",
        "action_weights": {a: 0.0 for a in ALL_ACTIONS} | {"attack": 0.9},
        "reasoning": "We are stronger",
        "veto_overridden": False,
        "veto_override_justification": None,
    })
    traits = MagicMock()
    traits.risk_tolerance = 0.8
    client = make_async_client(content)

    result = await call_chief("state...", [], {}, traits, client, "test-model", 5.0)

    assert result is not None
    assert result["era_goal"] == "Dominate the east"
    assert result["action_weights"]["attack"] == 0.9
    assert all(k in result["action_weights"] for k in ALL_ACTIONS)


@pytest.mark.asyncio
async def test_call_chief_clamps_weights():
    from agents.providers.council_ministers import call_chief
    from agents.decisions import ALL_ACTIONS

    content = json.dumps({
        "era_goal": "Test",
        "action_weights": {a: 5.0 for a in ALL_ACTIONS},
        "reasoning": "extreme",
        "veto_overridden": False,
        "veto_override_justification": None,
    })
    traits = MagicMock()
    traits.risk_tolerance = 0.5
    client = make_async_client(content)

    result = await call_chief("state...", [], {}, traits, client, "test-model", 5.0)

    assert all(v <= 1.0 for v in result["action_weights"].values())
    assert all(v >= -1.0 for v in result["action_weights"].values())


@pytest.mark.asyncio
async def test_call_chief_returns_none_on_bad_json():
    from agents.providers.council_ministers import call_chief

    traits = MagicMock()
    traits.risk_tolerance = 0.5
    client = make_async_client("not json")

    result = await call_chief("state...", [], {}, traits, client, "test-model", 5.0)
    assert result is None

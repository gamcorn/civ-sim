import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import make_mock_city


def _make_completion(content: str):
    """Build a minimal mock openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def test_returns_llm_action_when_valid():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider

    cfg = ProviderConfig(type="openai_compatible", model="test-model")
    provider = OpenAICompatibleProvider(cfg)

    city = make_mock_city()
    mock_create = AsyncMock(return_value=_make_completion("expand"))

    with patch.object(provider._client.chat.completions, "create", mock_create):
        results = asyncio.run(provider.choose_actions_batch([city]))

    assert results[0] == "expand"


def test_falls_back_on_timeout():
    import asyncio as _asyncio
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider

    cfg = ProviderConfig(type="openai_compatible", model="test-model", timeout=0.001)
    provider = OpenAICompatibleProvider(cfg)

    city = make_mock_city()

    async def _slow(*a, **kw):
        await _asyncio.sleep(10)

    with patch.object(provider._client.chat.completions, "create", _slow):
        results = asyncio.run(provider.choose_actions_batch([city]))

    # Must return a valid action (rule-based fallback), not raise
    from civ_sim.agents.decisions import ALL_ACTIONS
    assert results[0] in ALL_ACTIONS


def test_falls_back_on_hallucinated_response():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider

    cfg = ProviderConfig(type="openai_compatible", model="test-model")
    provider = OpenAICompatibleProvider(cfg)

    city = make_mock_city()
    mock_create = AsyncMock(return_value=_make_completion("I would recommend a diplomatic approach"))

    with patch.object(provider._client.chat.completions, "create", mock_create):
        results = asyncio.run(provider.choose_actions_batch([city]))

    from civ_sim.agents.decisions import ALL_ACTIONS
    assert results[0] in ALL_ACTIONS


def test_falls_back_on_api_exception():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider

    cfg = ProviderConfig(type="openai_compatible", model="test-model")
    provider = OpenAICompatibleProvider(cfg)

    city = make_mock_city()
    mock_create = AsyncMock(side_effect=Exception("connection refused"))

    with patch.object(provider._client.chat.completions, "create", mock_create):
        results = asyncio.run(provider.choose_actions_batch([city]))

    from civ_sim.agents.decisions import ALL_ACTIONS
    assert results[0] in ALL_ACTIONS


def test_batch_of_three_cities():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider

    cfg = ProviderConfig(type="openai_compatible", model="test-model")
    provider = OpenAICompatibleProvider(cfg)

    cities = [make_mock_city(), make_mock_city(), make_mock_city()]
    mock_create = AsyncMock(return_value=_make_completion("research"))

    with patch.object(provider._client.chat.completions, "create", mock_create):
        results = asyncio.run(provider.choose_actions_batch(cities))

    assert len(results) == 3
    assert mock_create.call_count == 3

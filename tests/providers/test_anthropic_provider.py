import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from tests.conftest import make_mock_city


def _make_anthropic_response(content: str):
    block = MagicMock()
    block.text = content
    response = MagicMock()
    response.content = [block]
    return response


def test_anthropic_returns_llm_action_when_valid():
    from config import ProviderConfig
    from agents.providers.anthropic_provider import AnthropicProvider

    cfg = ProviderConfig(type="anthropic", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(cfg)

    city = make_mock_city()
    mock_create = AsyncMock(return_value=_make_anthropic_response("research"))

    with patch.object(provider._client.messages, "create", mock_create):
        results = asyncio.run(provider.choose_actions_batch([city]))

    assert results[0] == "research"


def test_anthropic_falls_back_on_exception():
    from config import ProviderConfig
    from agents.providers.anthropic_provider import AnthropicProvider

    cfg = ProviderConfig(type="anthropic", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(cfg)

    city = make_mock_city()
    mock_create = AsyncMock(side_effect=Exception("rate limit"))

    with patch.object(provider._client.messages, "create", mock_create):
        results = asyncio.run(provider.choose_actions_batch([city]))

    from agents.decisions import ALL_ACTIONS
    assert results[0] in ALL_ACTIONS


def test_anthropic_falls_back_on_hallucination():
    from config import ProviderConfig
    from agents.providers.anthropic_provider import AnthropicProvider

    cfg = ProviderConfig(type="anthropic", model="claude-haiku-4-5-20251001")
    provider = AnthropicProvider(cfg)

    city = make_mock_city()
    mock_create = AsyncMock(return_value=_make_anthropic_response("Let me think about this carefully..."))

    with patch.object(provider._client.messages, "create", mock_create):
        results = asyncio.run(provider.choose_actions_batch([city]))

    from agents.decisions import ALL_ACTIONS
    assert results[0] in ALL_ACTIONS

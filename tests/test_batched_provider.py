from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider
from civ_sim.config import ProviderConfig
from tests.conftest import make_mock_city


def _provider(use_completions_api=False, max_concurrent=64):
    cfg = ProviderConfig(
        type="openai_compatible",
        model="test-model",
        use_completions_api=use_completions_api,
        max_concurrent=max_concurrent,
        timeout=5.0,
    )
    return OpenAICompatibleProvider(cfg)


def _make_completion_response(texts):
    resp = MagicMock()
    resp.choices = [MagicMock(text=t) for t in texts]
    return resp


def _make_chat_response(text):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    return resp


@pytest.mark.asyncio
async def test_choose_actions_batch_chat_returns_one_action_per_city():
    provider = _provider(use_completions_api=False)
    cities = [make_mock_city() for _ in range(3)]
    with patch.object(
        provider._client.chat.completions,
        "create",
        new=AsyncMock(return_value=_make_chat_response("gather")),
    ):
        results = await provider.choose_actions_batch(cities)
    assert len(results) == 3
    assert all(
        r in ["gather", "fortify", "trade", "expand", "attack", "research"]
        for r in results
    )


@pytest.mark.asyncio
async def test_choose_actions_batch_chat_respects_semaphore():
    provider = _provider(use_completions_api=False, max_concurrent=2)
    cities = [make_mock_city() for _ in range(5)]
    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        return _make_chat_response("gather")

    with patch.object(
        provider._client.chat.completions,
        "create",
        new=AsyncMock(side_effect=fake_create),
    ):
        results = await provider.choose_actions_batch(cities)
    assert len(results) == 5
    assert call_count == 5


@pytest.mark.asyncio
async def test_choose_actions_batch_completions_sends_one_request():
    provider = _provider(use_completions_api=True)
    cities = [make_mock_city() for _ in range(4)]
    fake_resp = _make_completion_response(["gather", "attack", "fortify", "gather"])

    with patch.object(
        provider._client.completions, "create", new=AsyncMock(return_value=fake_resp)
    ) as mock_create:
        await provider.choose_actions_batch(cities)

    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    assert isinstance(call_kwargs["prompt"], list)
    assert len(call_kwargs["prompt"]) == 4


@pytest.mark.asyncio
async def test_choose_actions_batch_completions_returns_valid_actions():
    provider = _provider(use_completions_api=True)
    cities = [make_mock_city() for _ in range(3)]
    fake_resp = _make_completion_response(["gather", "INVALID_ACTION", "fortify"])

    with patch.object(
        provider._client.completions, "create", new=AsyncMock(return_value=fake_resp)
    ):
        results = await provider.choose_actions_batch(cities)

    assert results[0] == "gather"
    assert results[1] in ["gather", "fortify", "trade", "expand", "attack", "research"]
    assert results[2] == "fortify"


@pytest.mark.asyncio
async def test_choose_actions_batch_completions_falls_back_on_exception():
    provider = _provider(use_completions_api=True)
    cities = [make_mock_city() for _ in range(2)]

    with patch.object(
        provider._client.completions,
        "create",
        new=AsyncMock(side_effect=Exception("vLLM down")),
    ):
        results = await provider.choose_actions_batch(cities)

    assert len(results) == 2
    assert all(
        r in ["gather", "fortify", "trade", "expand", "attack", "research"]
        for r in results
    )

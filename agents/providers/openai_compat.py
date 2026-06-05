from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING

import openai

from agents.providers.base import DecisionProvider
from agents.providers.prompt import SYSTEM_PROMPT, build_prompt, parse_response
from agents.decisions import choose_action, get_feasible_actions

if TYPE_CHECKING:
    from agents.city import CityAgent
    from config import ProviderConfig


class OpenAICompatibleProvider(DecisionProvider):
    """Async batch provider for any OpenAI-compatible endpoint (vLLM, Ollama, NIM, OpenAI)."""

    def __init__(self, config: "ProviderConfig"):
        self._config = config
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        tasks = [self._call_one(city) for city in cities]
        return list(await asyncio.gather(*tasks))

    async def _call_one(self, city: "CityAgent") -> str:
        feasible = get_feasible_actions(city)
        fallback = choose_action(city)
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._config.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(city, feasible)},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                ),
                timeout=self._config.timeout,
            )
            raw = response.choices[0].message.content or ""
            return parse_response(raw, feasible, fallback)
        except Exception:
            return fallback

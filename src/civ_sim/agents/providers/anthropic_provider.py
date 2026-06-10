from __future__ import annotations
import asyncio
from typing import TYPE_CHECKING

import anthropic

from civ_sim.agents.providers.base import DecisionProvider
from civ_sim.agents.providers.prompt import SYSTEM_PROMPT, build_prompt, parse_response
from civ_sim.agents.decisions import choose_action, get_feasible_actions

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent
    from civ_sim.config import ProviderConfig


class AnthropicProvider(DecisionProvider):
    """Async batch provider for the Anthropic API."""

    def __init__(self, config: "ProviderConfig"):
        self._config = config
        # base_url is passed through so Claude-compatible proxies work.
        # If using provider=anthropic with the default localhost URL, calls will
        # fail immediately and the per-city fallback will kick in.
        self._client = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        tasks = [self._call_one(city) for city in cities]
        return list(await asyncio.gather(*tasks))

    async def _call_one(self, city: "CityAgent") -> str:
        feasible = get_feasible_actions(city)
        fallback = choose_action(city)
        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._config.model,
                    max_tokens=self._config.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": build_prompt(city, feasible)},
                    ],
                    temperature=self._config.temperature,
                ),
                timeout=self._config.timeout,
            )
            raw = response.content[0].text if response.content else ""
            return parse_response(raw, feasible, fallback)
        except Exception:
            return fallback

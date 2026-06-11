from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import openai

from civ_sim.agents.decisions import choose_action, get_feasible_actions
from civ_sim.agents.providers.base import DecisionProvider
from civ_sim.agents.providers.prompt import SYSTEM_PROMPT, build_prompt, parse_response

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent
    from civ_sim.config import ProviderConfig

# Default chat template for Llama-3.1-Instruct / vLLM.
# Override via ProviderConfig.prompt_template for other model families.
_LLAMA_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


class OpenAICompatibleProvider(DecisionProvider):
    """Async batch provider for any OpenAI-compatible endpoint (vLLM, Ollama, NIM, OpenAI).

    Two execution paths selected by ProviderConfig.use_completions_api:

    False (default) — concurrent chat.completions calls, limited by
        max_concurrent semaphore to avoid flooding the server.

    True (DGX Spark recommended) — single /v1/completions call with all
        city prompts packed into one request. vLLM processes them as a
        true batch. Reduces HTTP round-trips from N → 1 per tick.
    """

    def __init__(self, config: "ProviderConfig"):
        self._config = config
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        if self._config.use_completions_api:
            return await self._call_batch_completions(cities)
        sem = asyncio.Semaphore(self._config.max_concurrent)

        async def _limited(city: "CityAgent") -> str:
            async with sem:
                return await self._call_one(city)

        return list(await asyncio.gather(*[_limited(c) for c in cities]))

    # ------------------------------------------------------------------
    # Chat API path (default)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Completions batch path (DGX Spark)
    # ------------------------------------------------------------------

    async def _call_batch_completions(self, cities: list["CityAgent"]) -> list[str]:
        """Send all city prompts in a single /v1/completions call.

        vLLM's completions endpoint accepts prompt as a list and returns one
        choice per element, processed as a true GPU batch. This eliminates N-1
        HTTP round-trips compared to the chat path.
        """
        feasibles = [get_feasible_actions(c) for c in cities]
        fallbacks = [choose_action(c) for c in cities]
        template = self._config.prompt_template or _LLAMA_TEMPLATE
        prompts = [
            template.format(system=SYSTEM_PROMPT, user=build_prompt(c, feas))
            for c, feas in zip(cities, feasibles)
        ]
        try:
            response = await asyncio.wait_for(
                self._client.completions.create(
                    model=self._config.model,
                    prompt=prompts,
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    stop=["\n", "<|eot_id|>"],
                ),
                timeout=self._config.timeout,
            )
            return [
                parse_response(choice.text, feas, fb)
                for choice, feas, fb in zip(response.choices, feasibles, fallbacks)
            ]
        except Exception:
            return fallbacks

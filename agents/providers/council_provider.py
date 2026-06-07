# agents/providers/council_provider.py
from __future__ import annotations
import asyncio
import dataclasses
from typing import TYPE_CHECKING

import openai

from agents.providers.base import DecisionProvider
from agents.decisions import ALL_ACTIONS, GATHER, get_action_scores, get_feasible_actions
from agents.providers.council_ministers import (
    call_sector_minister, call_budget_minister, call_chief,
)
from agents.providers.council_prompts import MINISTER_SPECS, build_civ_state_snapshot

if TYPE_CHECKING:
    from agents.city import CityAgent
    from agents.civilization import Civilization
    from config import ProviderConfig


@dataclasses.dataclass
class StrategicDirective:
    era_goal: str
    action_weights: dict[str, float]
    reasoning: str
    issued_at_tick: int
    valid_for_ticks: int
    emergency: bool = False


class CouncilProvider(DecisionProvider):
    def __init__(self, config: "ProviderConfig") -> None:
        self._config = config
        self._directive: StrategicDirective | None = None
        self._last_council_tick: int = -999
        self._last_emergency_tick: int = -999
        self._client = openai.AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        from agents.providers.rule_based import RuleBasedProvider as _RBP
        self._fallback = _RBP()

    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        if not cities:
            return []
        civ = cities[0].civ
        tick = cities[0].model.steps

        if self._should_run_council(civ, cities, tick):
            await self._run_council(civ, cities, tick)

        if self._directive is None:
            return await self._fallback.choose_actions_batch(cities)

        return [self._apply_directive(city) for city in cities]

    # ------------------------------------------------------------------

    def _should_run_council(
        self, civ: "Civilization", cities: list["CityAgent"], tick: int
    ) -> bool:
        if self._directive is None:
            return True
        if tick - self._last_council_tick >= self._config.directive_period:
            return True
        if not self._config.emergency_triggers:
            return False
        if tick - self._last_emergency_tick < self._config.emergency_cooldown_ticks:
            return False
        return self._check_emergencies(civ, cities)

    def _check_emergencies(
        self, civ: "Civilization", cities: list["CityAgent"]
    ) -> bool:
        model = cities[0].model
        # Food crisis
        if cities and sum(c.food_stock for c in cities) / len(cities) < 10.0:
            return True
        # Military threat
        enemy_mil = sum(
            c.total_military for c in model.civilizations if c.civ_id != civ.civ_id
        )
        if enemy_mil > 2 * max(1, civ.total_military):
            return True
        # City lost since last directive
        if (
            civ._city_count_at_last_directive > 0
            and len(cities) < civ._city_count_at_last_directive
        ):
            return True
        # Population collapse
        if civ._pop_at_last_directive > 0 and civ.total_pop < 0.8 * civ._pop_at_last_directive:
            return True
        # Tech unlock
        if len(civ.discovered_techs) > civ._techs_at_last_directive:
            return True
        return False

    async def _run_council(
        self, civ: "Civilization", cities: list["CityAgent"], tick: int
    ) -> None:
        is_emergency = (
            self._directive is not None
            and tick - self._last_council_tick < self._config.directive_period
        )
        model = cities[0].model
        state_snapshot = build_civ_state_snapshot(civ, cities, model)
        sector_model = self._config.sector_model or self._config.model
        chief_model = self._config.chief_model or self._config.model
        timeout = self._config.timeout

        sector_outputs: list[dict] = []
        for _round in range(1, self._config.max_rounds + 1):
            prev = (
                [f"{o['name']}: {o.get('recommendation', '')}" for o in sector_outputs]
                if sector_outputs else None
            )
            results = await asyncio.gather(*[
                call_sector_minister(
                    spec, state_snapshot, civ.traits,
                    self._client, sector_model, timeout, prev,
                )
                for spec in MINISTER_SPECS
            ], return_exceptions=True)
            sector_outputs = [r for r in results if isinstance(r, dict)]

        budget_output = await call_budget_minister(
            state_snapshot, sector_outputs, civ.traits,
            self._client, sector_model, timeout,
        )
        chief_output = await call_chief(
            state_snapshot, sector_outputs, budget_output, civ.traits,
            self._client, chief_model, timeout,
            round_num=self._config.max_rounds,
            max_rounds=self._config.max_rounds,
        )

        if chief_output is None:
            self._last_council_tick = tick
            return

        directive = StrategicDirective(
            era_goal=chief_output.get("era_goal", ""),
            action_weights={a: float(chief_output["action_weights"].get(a, 0.0)) for a in ALL_ACTIONS},
            reasoning=chief_output.get("reasoning", ""),
            issued_at_tick=tick,
            valid_for_ticks=self._config.directive_period,
            emergency=is_emergency,
        )
        self._directive = directive
        self._last_council_tick = tick
        if is_emergency:
            self._last_emergency_tick = tick

        civ._pop_at_last_directive = civ.total_pop
        civ._techs_at_last_directive = len(civ.discovered_techs)
        civ._city_count_at_last_directive = len(cities)

        if hasattr(model, "logger"):
            model.logger.log_directive(tick, civ.civ_id, directive)

    def _apply_directive(self, city: "CityAgent") -> str:
        scores = get_action_scores(city)
        for action, weight in self._directive.action_weights.items():
            if action in scores:
                scores[action] += weight
        feasible = get_feasible_actions(city)
        if not feasible:
            return GATHER
        return max(feasible, key=lambda a: scores.get(a, 0.0))

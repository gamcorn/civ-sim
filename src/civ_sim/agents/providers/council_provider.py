# agents/providers/council_provider.py
from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING

import openai

from civ_sim.agents.decisions import (
    ALL_ACTIONS,
    GATHER,
    get_action_scores,
    get_feasible_actions,
)
from civ_sim.agents.providers.base import DecisionProvider
from civ_sim.agents.providers.council_ministers import (
    call_budget_minister,
    call_chief,
    call_chief_lite,
    call_sector_minister,
)
from civ_sim.agents.providers.council_prompts import (
    MINISTER_SPECS,
    build_civ_state_snapshot,
)

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent
    from civ_sim.agents.civilization import Civilization
    from civ_sim.config import ProviderConfig


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
        from civ_sim.agents.providers.rule_based import RuleBasedProvider as _RBP

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
        if (
            civ._pop_at_last_directive > 0
            and civ.total_pop < 0.8 * civ._pop_at_last_directive
        ):
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
        fog = getattr(model.config, "fog_of_war", 0.0)
        state_snapshot = build_civ_state_snapshot(civ, cities, model, fog_of_war=fog)
        sector_model = self._config.sector_model or self._config.model
        chief_model = self._config.chief_model or self._config.model
        timeout = self._config.timeout
        chief_timeout = self._config.chief_timeout or timeout * 3
        gj = self._config.guided_json

        # ── council_off: single chief call, skip minister debate ───────
        if self._config.council_off:
            chief_output = await call_chief_lite(
                state_snapshot,
                civ.traits,
                self._client,
                chief_model,
                chief_timeout,
                guided_json=gj,
            )
            self._last_council_tick = tick
            if chief_output is None:
                if hasattr(model, "logger"):
                    model.logger.log_directive(tick, civ.civ_id, None, success=False)
                    model.logger.log_council_session(
                        tick,
                        civ.civ_id,
                        emergency=is_emergency,
                        council_off=True,
                        state_snapshot=state_snapshot,
                        sector_outputs=[],
                        budget_output=None,
                        chief_output=None,
                        success=False,
                    )
                return
            directive = StrategicDirective(
                era_goal="",
                action_weights={
                    a: float(chief_output["action_weights"].get(a, 0.0))
                    for a in ALL_ACTIONS
                },
                reasoning="",
                issued_at_tick=tick,
                valid_for_ticks=self._config.directive_period,
                emergency=is_emergency,
            )
            self._directive = directive
            if hasattr(model, "logger"):
                model.logger.log_directive(tick, civ.civ_id, directive)
                model.logger.log_council_session(
                    tick,
                    civ.civ_id,
                    emergency=is_emergency,
                    council_off=True,
                    state_snapshot=state_snapshot,
                    sector_outputs=[],
                    budget_output=None,
                    chief_output=chief_output,
                    success=True,
                )
            self._update_civ_snapshot(civ, cities, tick, is_emergency)
            return

        # ── Full council: sector ministers → budget → chief ────────────
        sector_outputs: list[dict] = []
        for _round in range(1, self._config.max_rounds + 1):
            prev = (
                [f"{o['name']}: {o.get('recommendation', '')}" for o in sector_outputs]
                if sector_outputs
                else None
            )
            results = await asyncio.gather(
                *[
                    call_sector_minister(
                        spec,
                        state_snapshot,
                        civ.traits,
                        self._client,
                        sector_model,
                        timeout,
                        prev,
                        gj,
                    )
                    for spec in MINISTER_SPECS
                ],
                return_exceptions=True,
            )
            sector_outputs = [r for r in results if isinstance(r, dict)]

        budget_output = await call_budget_minister(
            state_snapshot,
            sector_outputs,
            civ.traits,
            self._client,
            sector_model,
            timeout,
            gj,
        )
        chief_output = await call_chief(
            state_snapshot,
            sector_outputs,
            budget_output,
            civ.traits,
            self._client,
            chief_model,
            chief_timeout,
            round_num=self._config.max_rounds,
            max_rounds=self._config.max_rounds,
            guided_json=gj,
        )

        self._last_council_tick = tick

        if chief_output is None:
            # Try to salvage a directive from sector weight_requests
            chief_output = self._synthesize_from_sectors(sector_outputs)

        if chief_output is None:
            if hasattr(model, "logger"):
                model.logger.log_directive(tick, civ.civ_id, None, success=False)
                model.logger.log_council_session(
                    tick,
                    civ.civ_id,
                    emergency=is_emergency,
                    council_off=False,
                    state_snapshot=state_snapshot,
                    sector_outputs=sector_outputs,
                    budget_output=budget_output,
                    chief_output=None,
                    success=False,
                )
            return

        directive = StrategicDirective(
            era_goal=chief_output.get("era_goal", ""),
            action_weights={
                a: float(chief_output["action_weights"].get(a, 0.0))
                for a in ALL_ACTIONS
            },
            reasoning=chief_output.get("reasoning", ""),
            issued_at_tick=tick,
            valid_for_ticks=self._config.directive_period,
            emergency=is_emergency,
        )
        self._directive = directive
        if is_emergency:
            self._last_emergency_tick = tick

        self._update_civ_snapshot(civ, cities, tick, is_emergency)

        if hasattr(model, "logger"):
            model.logger.log_directive(tick, civ.civ_id, directive)
            model.logger.log_council_session(
                tick,
                civ.civ_id,
                emergency=is_emergency,
                council_off=False,
                state_snapshot=state_snapshot,
                sector_outputs=sector_outputs,
                budget_output=budget_output,
                chief_output=chief_output,
                success=True,
            )

    def _synthesize_from_sectors(self, sector_outputs: list[dict]) -> dict | None:
        """Build a minimal chief output from sector weight_requests when chief fails."""
        merged: dict[str, float] = {}
        count = 0
        for o in sector_outputs:
            for action, w in o.get("weight_requests", {}).items():
                if action in ALL_ACTIONS:
                    try:
                        merged[action] = merged.get(action, 0.0) + float(w)
                        count += 1
                    except (TypeError, ValueError):
                        pass
        if not merged:
            return None
        n = max(1, len(sector_outputs))
        return {
            "era_goal": "",
            "action_weights": {
                a: max(-1.0, min(1.0, merged.get(a, 0.0) / n)) for a in ALL_ACTIONS
            },
            "reasoning": "synthesized from sector weights (chief failed)",
        }

    def _update_civ_snapshot(
        self,
        civ: "Civilization",
        cities: list["CityAgent"],
        tick: int,
        is_emergency: bool,
    ) -> None:
        if is_emergency:
            self._last_emergency_tick = tick
        civ._pop_at_last_directive = civ.total_pop
        civ._techs_at_last_directive = len(civ.discovered_techs)
        civ._city_count_at_last_directive = len(cities)

    def _apply_directive(self, city: "CityAgent") -> str:
        assert self._directive is not None
        scores = get_action_scores(city)
        for action, weight in self._directive.action_weights.items():
            if action in scores:
                scores[action] += weight
        feasible = get_feasible_actions(city)
        if not feasible:
            return GATHER
        return max(feasible, key=lambda a: scores.get(a, 0.0))

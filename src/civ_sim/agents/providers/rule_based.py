from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from civ_sim.agents.decisions import choose_action
from civ_sim.agents.providers.base import DecisionProvider

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent

logger = logging.getLogger(__name__)


class RuleBasedProvider(DecisionProvider):
    """Wraps the weighted-scoring decision engine. No async I/O."""

    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        logger.debug("RuleBasedProvider: choosing actions for %d cities", len(cities))
        return [choose_action(city) for city in cities]

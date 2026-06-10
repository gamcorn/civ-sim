from __future__ import annotations
from typing import TYPE_CHECKING

from agents.providers.base import DecisionProvider
from agents.decisions import choose_action

if TYPE_CHECKING:
    from agents.city import CityAgent


class RuleBasedProvider(DecisionProvider):
    """Wraps the existing weighted-scoring decision engine. No async I/O."""

    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        return [choose_action(city) for city in cities]

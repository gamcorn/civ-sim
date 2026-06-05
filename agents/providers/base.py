from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.city import CityAgent


class DecisionProvider(ABC):
    """Abstract base for all city decision backends."""

    @abstractmethod
    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        """Return one action string per city. Must never raise."""

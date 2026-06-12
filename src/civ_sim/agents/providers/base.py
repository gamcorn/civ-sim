from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent

logger = logging.getLogger(__name__)


class DecisionProvider(ABC):
    """Abstract base for all city decision backends."""

    @abstractmethod
    async def choose_actions_batch(self, cities: list["CityAgent"]) -> list[str]:
        """Return one action string per city. Must never raise."""

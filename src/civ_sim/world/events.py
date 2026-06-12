from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np

from civ_sim.world.resources import ResourceType

if TYPE_CHECKING:
    from civ_sim.config import SimConfig
    from civ_sim.world.grid import ResourceGrid

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EnvEvent:
    name: str
    description: str
    transmission_rate: float = 1.0  # for disease events: β in [0.1, 3.0]


class EventSampler:
    """Samples environmental events each tick and applies them to the resource grid."""

    def __init__(self, config: SimConfig, rng):
        self.config = config
        self._rng = rng

    def sample(self, grid: ResourceGrid) -> list[EnvEvent]:
        fired: list[EnvEvent] = []
        r = self._rng

        if r.random() < self.config.drought_prob:
            event = self._drought(grid)
            logger.info("Environmental event: %s", event.description)
            fired.append(event)

        occupation = float((grid.ownership >= 0).sum()) / (grid.width * grid.height)
        disease_prob = self.config.disease_prob * (
            1.0 + occupation * self.config.disease_land_scale
        )
        if r.random() < disease_prob:
            beta = round(r.uniform(0.1, 3.0), 2)
            pct = round(occupation * 100, 1)
            event = EnvEvent(
                "disease",
                f"Epidemic β={beta} (land use {pct}%) sweeps populations",
                transmission_rate=beta,
            )
            logger.warning("Environmental event: %s", event.description)
            fired.append(event)

        if r.random() < self.config.mineral_boom_prob:
            event = self._mineral_boom(grid)
            logger.info("Environmental event: %s", event.description)
            fired.append(event)

        if r.random() < self.config.climate_shift_prob:
            event = self._climate_shift()
            logger.info("Environmental event: %s", event.description)
            fired.append(event)

        return fired

    def _drought(self, grid: ResourceGrid) -> EnvEvent:
        cx = self._rng.randint(0, grid.width - 1)
        cy = self._rng.randint(0, grid.height - 1)
        x0, x1 = max(0, cx - 5), min(grid.width, cx + 5)
        y0, y1 = max(0, cy - 5), min(grid.height, cy + 5)
        grid.layers[ResourceType.FOOD].data[x0:x1, y0:y1] *= 0.5
        return EnvEvent("drought", f"Drought at ({cx},{cy}) halves food in region")

    def _mineral_boom(self, grid: ResourceGrid) -> EnvEvent:
        cx = self._rng.randint(0, grid.width - 1)
        cy = self._rng.randint(0, grid.height - 1)
        x0, x1 = max(0, cx - 3), min(grid.width, cx + 3)
        y0, y1 = max(0, cy - 3), min(grid.height, cy + 3)
        grid.layers[ResourceType.MINERALS].data[x0:x1, y0:y1] = np.minimum(
            grid.layers[ResourceType.MINERALS].data[x0:x1, y0:y1] * 3,
            grid.config.resource_max,
        )
        return EnvEvent("mineral_boom", f"Mineral vein exposed at ({cx},{cy})")

    def _climate_shift(self) -> EnvEvent:
        return EnvEvent(
            "climate_shift", "Climate shift: food regeneration reduced globally"
        )

from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from world.grid import ResourceGrid
    from config import SimConfig


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
            fired.append(self._drought(grid))

        occupation = float((grid.ownership >= 0).sum()) / (grid.width * grid.height)
        disease_prob = self.config.disease_prob * (1.0 + occupation * self.config.disease_land_scale)
        if r.random() < disease_prob:
            beta = round(r.uniform(0.1, 3.0), 2)
            pct = round(occupation * 100, 1)
            fired.append(EnvEvent(
                "disease",
                f"Epidemic β={beta} (land use {pct}%) sweeps populations",
                transmission_rate=beta,
            ))

        if r.random() < self.config.mineral_boom_prob:
            fired.append(self._mineral_boom(grid))

        if r.random() < self.config.climate_shift_prob:
            fired.append(self._climate_shift())

        return fired

    # ------------------------------------------------------------------

    def _drought(self, grid: ResourceGrid) -> EnvEvent:
        # Halve food in a random 10×10 patch
        cx = self._rng.randint(0, grid.width - 1)
        cy = self._rng.randint(0, grid.height - 1)
        x0, x1 = max(0, cx - 5), min(grid.width, cx + 5)
        y0, y1 = max(0, cy - 5), min(grid.height, cy + 5)
        from world.resources import ResourceType
        grid.layers[ResourceType.FOOD].data[x0:x1, y0:y1] *= 0.5
        return EnvEvent("drought", f"Drought at ({cx},{cy}) halves food in region")

    def _mineral_boom(self, grid: ResourceGrid) -> EnvEvent:
        cx = self._rng.randint(0, grid.width - 1)
        cy = self._rng.randint(0, grid.height - 1)
        x0, x1 = max(0, cx - 3), min(grid.width, cx + 3)
        y0, y1 = max(0, cy - 3), min(grid.height, cy + 3)
        from world.resources import ResourceType
        import numpy as np
        grid.layers[ResourceType.MINERALS].data[x0:x1, y0:y1] = np.minimum(
            grid.layers[ResourceType.MINERALS].data[x0:x1, y0:y1] * 3,
            grid.config.resource_max,
        )
        return EnvEvent("mineral_boom", f"Mineral vein exposed at ({cx},{cy})")

    def _climate_shift(self) -> EnvEvent:
        # Climate shifts affect regen rates — handled by model reducing food_regen for a few ticks
        return EnvEvent("climate_shift", "Climate shift: food regeneration reduced globally")

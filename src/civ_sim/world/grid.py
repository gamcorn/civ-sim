import random as _random_module

import numpy as np
import opensimplex
from mesa.discrete_space import OrthogonalMooreGrid, PropertyLayer

from .resources import ResourceType, NOISE_SCALE


def _load_backend(name: str):
    """Return the array module for the requested backend.

    Supported: 'numpy' (always available), 'cupy' (requires GPU + cupy install).
    Raises ValueError for unknown values so misconfiguration is caught early.
    """
    if name == "numpy":
        return np
    if name == "cupy":
        try:
            import cupy
            return cupy
        except ImportError as exc:
            raise ImportError(
                "CuPy is not installed. Install it for your CUDA version: "
                "pip install cupy-cuda12x"
            ) from exc
    raise ValueError(
        f"Unknown grid_backend {name!r}. Choose 'numpy' or 'cupy'."
    )


class ResourceGrid:
    """2D world: Mesa spatial grid + per-resource property layers + ownership map.

    All array operations route through self.xp (the backend module).
    Switching to CuPy requires only changing SimConfig.grid_backend to 'cupy'
    — no call-site edits needed.
    """

    def __init__(self, width: int, height: int, config, random):
        self.width = width
        self.height = height
        self.config = config

        backend_name = getattr(config, "grid_backend", "numpy")
        self.xp = _load_backend(backend_name)

        self.mesa_grid = OrthogonalMooreGrid(
            [width, height], torus=False, random=random
        )

        self.layers: dict[ResourceType, PropertyLayer] = {}
        for rt in ResourceType:
            data = self._perlin_map(rt, random)
            layer = PropertyLayer.from_data(rt.value, data)
            layer.data = self.xp.asarray(layer.data)
            self.layers[rt] = layer

        self.ownership = self.xp.full((width, height), -1, dtype=self.xp.int8)

        # Permanent soil-quality layers (Perlin-seeded, independent from resource buffers)
        # Use a dedicated sub-RNG seeded deterministically from config so we
        # consume zero draws from the shared random sequence (preserving
        # reproducibility of all downstream RNG-dependent code).
        soil_seed = getattr(config, "rng_seed", 0) ^ 0xDEADBEEF
        soil_rng = _random_module.Random(soil_seed)
        self.base_soil_fertility   = self._perlin_map_raw(0.05, soil_rng)
        self.soil_fertility        = self.base_soil_fertility.copy()
        self.base_mineral_richness = self._perlin_map_raw(0.09, soil_rng)
        self.mineral_richness      = self.base_mineral_richness.copy()
        self.base_forest_density   = self._perlin_map_raw(0.04, soil_rng)
        self.forest_density        = self.base_forest_density.copy()

    def _perlin_map(self, rt: ResourceType, random) -> np.ndarray:
        # Always generate on CPU with numpy (opensimplex is CPU-only)
        seed = random.randint(0, 2**30)
        opensimplex.seed(seed)
        scale = NOISE_SCALE[rt]
        xs = np.arange(self.width) * scale
        ys = np.arange(self.height) * scale
        raw = opensimplex.noise2array(xs, ys).T
        return ((raw + 1.0) / 2.0 * self.config.resource_max).astype(np.float32)

    def _perlin_map_raw(self, scale: float, random) -> np.ndarray:
        seed = random.randint(0, 2**30)
        opensimplex.seed(seed)
        xs = np.arange(self.width) * scale
        ys = np.arange(self.height) * scale
        raw = opensimplex.noise2array(xs, ys).T
        return ((raw + 1.0) / 2.0 * self.config.resource_max).astype(np.float32)

    def step(self, food_regen_mult: float = 1.0):
        regen_map = {
            ResourceType.FOOD:     self.config.food_regen * food_regen_mult,
            ResourceType.WATER:    self.config.water_regen,
            ResourceType.WOOD:     self.config.wood_regen,
            ResourceType.MINERALS: self.config.mineral_regen,
        }
        for rt, rate in regen_map.items():
            if rate > 0:
                d = self.layers[rt].data
                d += rate * self.config.resource_max
                self.xp.clip(d, 0, self.config.resource_max, out=d)

    def consume(self, x: int, y: int, rt: ResourceType, amount: float) -> float:
        d = self.layers[rt].data
        available = float(d[x, y])
        consumed = min(available, amount)
        d[x, y] = available - consumed
        return consumed

    def deposit(self, x: int, y: int, rt: ResourceType, amount: float):
        d = self.layers[rt].data
        d[x, y] = min(float(d[x, y]) + amount, self.config.resource_max)

    def get(self, x: int, y: int, rt: ResourceType) -> float:
        return float(self.layers[rt].data[x, y])

    def cell(self, x: int, y: int):
        return self.mesa_grid[(x, y)]

    def claim(self, x: int, y: int, civ_id: int):
        self.ownership[x, y] = civ_id

    def territory_count(self, civ_id: int) -> int:
        return int(self.xp.sum(self.ownership == civ_id))

    def _owned_tiles_in_radius(
        self, civ_id: int, cx: int, cy: int, radius: int
    ) -> list[tuple[int, int]]:
        tiles = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if int(self.ownership[nx, ny]) == civ_id:
                        tiles.append((nx, ny))
        return tiles

    def avg_soil_fertility(self, civ_id: int, cx: int, cy: int, radius: int) -> float:
        tiles = self._owned_tiles_in_radius(civ_id, cx, cy, radius)
        if not tiles:
            return 0.0
        return float(np.mean([self.soil_fertility[x, y] for x, y in tiles]))

    def avg_mineral_richness(self, civ_id: int, cx: int, cy: int, radius: int) -> float:
        tiles = self._owned_tiles_in_radius(civ_id, cx, cy, radius)
        if not tiles:
            return 0.0
        return float(np.mean([self.mineral_richness[x, y] for x, y in tiles]))

    def avg_forest_density(self, civ_id: int, cx: int, cy: int, radius: int) -> float:
        tiles = self._owned_tiles_in_radius(civ_id, cx, cy, radius)
        if not tiles:
            return 0.0
        return float(np.mean([self.forest_density[x, y] for x, y in tiles]))

    def apply_labor_degradation(
        self,
        cx: int, cy: int, radius: int, civ_id: int,
        farmer_ratio: float, miner_ratio: float, woodcutter_ratio: float,
        config,
    ) -> None:
        dr = config.degradation_rate
        rr = config.recovery_rate
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if int(self.ownership[nx, ny]) == civ_id:
                        self.soil_fertility[nx, ny] = max(
                            0.0,
                            float(self.soil_fertility[nx, ny]) - dr * farmer_ratio,
                        )
                        self.mineral_richness[nx, ny] = max(
                            0.0,
                            float(self.mineral_richness[nx, ny]) - dr * miner_ratio,
                        )
                        self.forest_density[nx, ny] = max(
                            0.0,
                            float(self.forest_density[nx, ny]) - dr * woodcutter_ratio,
                        )
                        # Fallow recovery toward base
                        self.soil_fertility[nx, ny] += rr * (
                            float(self.base_soil_fertility[nx, ny]) - float(self.soil_fertility[nx, ny])
                        )
                        self.mineral_richness[nx, ny] += rr * (
                            float(self.base_mineral_richness[nx, ny]) - float(self.mineral_richness[nx, ny])
                        )
                        self.forest_density[nx, ny] += rr * (
                            float(self.base_forest_density[nx, ny]) - float(self.forest_density[nx, ny])
                        )

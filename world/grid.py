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

    def _perlin_map(self, rt: ResourceType, random) -> np.ndarray:
        # Always generate on CPU with numpy (opensimplex is CPU-only)
        seed = random.randint(0, 2**30)
        opensimplex.seed(seed)
        scale = NOISE_SCALE[rt]
        xs = np.arange(self.width) * scale
        ys = np.arange(self.height) * scale
        raw = opensimplex.noise2array(xs, ys).T
        return ((raw + 1.0) / 2.0 * self.config.resource_max).astype(np.float32)

    def step(self):
        regen_map = {
            ResourceType.FOOD:     self.config.food_regen,
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

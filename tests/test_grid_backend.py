import random
import pytest
import numpy as np
from civ_sim.config import SimConfig
from civ_sim.world.grid import ResourceGrid
from civ_sim.world.resources import ResourceType


@pytest.fixture
def numpy_grid():
    cfg = SimConfig(width=10, height=10, resource_max=100.0,
                    food_regen=0.04, mineral_regen=0.0,
                    grid_backend="numpy", rng_seed=0)
    return ResourceGrid(10, 10, cfg, random.Random(0))


def test_backend_attribute_is_numpy_module(numpy_grid):
    import numpy as np_mod
    assert numpy_grid.xp is np_mod


def test_ownership_is_numpy_array(numpy_grid):
    assert isinstance(numpy_grid.ownership, np.ndarray)


def test_layer_data_is_numpy_array(numpy_grid):
    layer = numpy_grid.layers[ResourceType.FOOD]
    assert isinstance(layer.data, np.ndarray)


def test_consume_still_works_with_explicit_backend(numpy_grid):
    numpy_grid.layers[ResourceType.FOOD].data[5, 5] = 20.0
    consumed = numpy_grid.consume(5, 5, ResourceType.FOOD, 10.0)
    assert consumed == pytest.approx(10.0)
    assert numpy_grid.get(5, 5, ResourceType.FOOD) == pytest.approx(10.0)


def test_step_regen_still_works_with_explicit_backend(numpy_grid):
    numpy_grid.layers[ResourceType.FOOD].data[:] = 0.0
    numpy_grid.step()
    assert numpy_grid.get(0, 0, ResourceType.FOOD) == pytest.approx(4.0, abs=0.01)


def test_territory_count_still_works_with_explicit_backend(numpy_grid):
    numpy_grid.claim(0, 0, civ_id=0)
    numpy_grid.claim(1, 0, civ_id=0)
    assert numpy_grid.territory_count(0) == 2


def test_unknown_backend_raises():
    cfg = SimConfig(width=5, height=5, grid_backend="badbackend", rng_seed=0)
    with pytest.raises(ValueError, match="grid_backend"):
        ResourceGrid(5, 5, cfg, random.Random(0))

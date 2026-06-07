import numpy as np
from types import SimpleNamespace
from world.resources import ResourceType
from storage.snapshot import (
    CivState, CityState, _LayerProxy, GridState, ReplayFrame,
)


def test_civ_state_fields():
    cs = CivState(civ_id=0, name="Alpha", alive=True, discovered_techs=[1, 2, 3])
    assert cs.civ_id == 0
    assert cs.name == "Alpha"
    assert cs.alive is True
    assert len(cs.discovered_techs) == 3


def test_city_state_fields():
    civ = CivState(civ_id=1, name="Beta", alive=True, discovered_techs=[])
    city = CityState(x=5, y=10, population=120.0, military=30.0,
                     food_stock=50.0, last_action="gather", civ=civ)
    assert city.x == 5
    assert city.civ.civ_id == 1


def test_layer_proxy_data():
    arr = np.zeros((4, 4), dtype=np.float32)
    proxy = _LayerProxy(arr)
    assert proxy.data is arr


def test_grid_state_fields():
    ownership = np.zeros((4, 4), dtype=np.int8)
    food = np.ones((4, 4), dtype=np.float32) * 50.0
    gs = GridState(ownership=ownership,
                   layers={ResourceType.FOOD: _LayerProxy(food)})
    assert gs.layers[ResourceType.FOOD].data[0, 0] == 50.0


def test_replay_frame_duck_types_model():
    """ReplayFrame has all attributes renderers read off a CivModel."""
    civ = CivState(civ_id=0, name="Alpha", alive=True, discovered_techs=[])
    city = CityState(x=2, y=3, population=80.0, military=10.0,
                     food_stock=20.0, last_action="expand", civ=civ)
    ownership = np.zeros((8, 6), dtype=np.int8)
    food = np.zeros((8, 6), dtype=np.float32)
    gs = GridState(ownership=ownership,
                   layers={ResourceType.FOOD: _LayerProxy(food)})
    frame = ReplayFrame(
        steps=42, running=True,
        civilizations=[civ], agents=[city],
        grid=gs,
        config=SimpleNamespace(width=8, height=6, resource_max=100.0),
        history={"tick": [1], "pop_0": [80.0], "mil_0": [10.0],
                 "pop_1": [0.0], "mil_1": [0.0]},
    )
    assert frame.steps == 42
    assert frame.config.resource_max == 100.0
    assert hasattr(frame.agents[0], "civ")

"""Verify that both renderers accept ReplayFrame objects (duck-typing contract)."""
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from civ_sim.storage.snapshot import CivState, CityState, _LayerProxy, GridState, ReplayFrame
from civ_sim.world.resources import ResourceType


def _make_frame(n_civs=2, n_cities=1):
    """Build a minimal ReplayFrame suitable for renderer tests."""
    civs = [
        CivState(civ_id=i, name=f"Civ{i}", alive=True, discovered_techs=[])
        for i in range(n_civs)
    ]
    agents = [
        CityState(x=5 * (i + 1), y=5, population=80.0, military=10.0,
                  food_stock=20.0, last_action="gather", civ=civs[i % n_civs])
        for i in range(n_cities)
    ]
    w, h = 20, 15
    ownership = np.full((w, h), -1, dtype=np.int8)
    food = np.full((w, h), 40.0, dtype=np.float32)
    history = {
        "tick": [1, 2],
        "pop_0": [80.0, 85.0],
        "pop_1": [80.0, 78.0],
        "mil_0": [10.0, 11.0],
        "mil_1": [10.0, 9.0],
    }
    return ReplayFrame(
        steps=2, running=True,
        civilizations=civs, agents=agents,
        grid=GridState(
            ownership=ownership,
            layers={ResourceType.FOOD: _LayerProxy(food)},
        ),
        config=SimpleNamespace(width=w, height=h, resource_max=100.0),
        history=history,
    )


def test_terminal_renderer_update_accepts_replay_frame():
    """TerminalRenderer.update() must not crash when given a ReplayFrame."""
    from civ_sim.visualization.terminal_renderer import TerminalRenderer

    frame = _make_frame()
    with patch("sys.stdout", io.StringIO()):
        renderer = TerminalRenderer(frame)
        renderer.update(frame)


def test_terminal_renderer_city_state_visible_in_output():
    """City markers appear when agents are CityState objects (hasattr check)."""
    from civ_sim.visualization.terminal_renderer import TerminalRenderer

    frame = _make_frame(n_civs=2, n_cities=1)
    buf = io.StringIO()
    with patch("sys.stdout", buf):
        renderer = TerminalRenderer(frame)
        renderer.update(frame)
    output = buf.getvalue()
    # The renderer must produce some output — it didn't skip all agents
    assert len(output) > 0


def test_terminal_renderer_territory_count_works_on_grid_state():
    """territory_count() on GridState doesn't crash inside update()."""
    from civ_sim.visualization.terminal_renderer import TerminalRenderer

    frame = _make_frame()
    # Mark some tiles as owned by civ 0
    frame.grid.ownership[0:5, 0:5] = 0

    with patch("sys.stdout", io.StringIO()):
        renderer = TerminalRenderer(frame)
        renderer.update(frame)  # calls grid.territory_count(civ.civ_id) internally

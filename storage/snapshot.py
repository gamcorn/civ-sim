from __future__ import annotations
import bisect
import json
from dataclasses import dataclass
from types import SimpleNamespace

import duckdb
import numpy as np

# ---------------------------------------------------------------------------
# DuckDB schema
# ---------------------------------------------------------------------------

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS snapshots (
    tick         INTEGER,
    seed         INTEGER,
    ownership    BLOB,
    food         BLOB,
    width        INTEGER,
    height       INTEGER,
    resource_max REAL,
    cities       VARCHAR,
    civs         VARCHAR
);
"""

# ---------------------------------------------------------------------------
# Replay frame types — duck-typed to match CivModel's interface
# ---------------------------------------------------------------------------


@dataclass
class CivState:
    civ_id: int
    name: str
    alive: bool
    discovered_techs: list  # only len() is used by renderers


@dataclass
class CityState:
    x: int
    y: int
    population: float
    military: float
    food_stock: float
    last_action: str
    civ: CivState


class _LayerProxy:
    """Wraps a numpy array so grid.layers[ResourceType.FOOD].data works."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = data


@dataclass
class GridState:
    ownership: np.ndarray
    layers: dict  # {ResourceType.FOOD: _LayerProxy}


@dataclass
class ReplayFrame:
    steps: int
    running: bool
    civilizations: list  # list[CivState]
    agents: list         # list[CityState]
    grid: GridState
    config: SimpleNamespace   # .width .height .resource_max
    history: dict             # {"tick", "pop_0", "pop_1", "mil_0", "mil_1"}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class SnapshotWriter:
    """Writes periodic world snapshots to the snapshots table."""

    def __init__(self, db_path: str, seed: int) -> None:
        self._seed = seed
        self._con = duckdb.connect(db_path)
        self._con.execute(_CREATE_SQL)

    def write(self, tick: int, grid, agents, civilizations) -> None:
        from agents.city import CityAgent
        from world.resources import ResourceType

        ownership = np.asarray(grid.ownership).astype(np.int8)
        food = np.asarray(grid.layers[ResourceType.FOOD].data).astype(np.float32)

        cities_data = [
            {
                "agent_id": str(a.unique_id),
                "civ_id": a.civ.civ_id,
                "x": a.x, "y": a.y,
                "pop": float(a.population),
                "military": float(a.military),
                "food_stock": float(a.food_stock),
                "last_action": a.last_action or "",
            }
            for a in agents
            if isinstance(a, CityAgent)
        ]
        civs_data = [
            {
                "civ_id": c.civ_id,
                "name": c.name,
                "alive": c.alive,
                "tech_level": c.tech_level,
                "n_techs": len(c.discovered_techs),
            }
            for c in civilizations
        ]

        self._con.execute(
            "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?)",
            (
                tick, self._seed,
                ownership.tobytes(), food.tobytes(),
                int(ownership.shape[0]), int(ownership.shape[1]),
                float(grid.config.resource_max),
                json.dumps(cities_data), json.dumps(civs_data),
            ),
        )

    def close(self) -> None:
        self._con.close()

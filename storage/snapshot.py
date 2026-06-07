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

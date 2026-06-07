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


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class SnapshotReader:
    """Reads snapshots from DuckDB and reconstructs ReplayFrame objects."""

    def __init__(self, db_path: str) -> None:
        self._con = duckdb.connect(db_path, read_only=True)
        self._history = self._load_history()

    def ticks(self) -> list[int]:
        return [r[0] for r in self._con.execute(
            "SELECT tick FROM snapshots ORDER BY tick"
        ).fetchall()]

    def load(self, tick: int) -> ReplayFrame:
        from world.resources import ResourceType

        row = self._con.execute(
            "SELECT ownership, food, width, height, resource_max, cities, civs "
            "FROM snapshots WHERE tick = ?",
            [tick],
        ).fetchone()

        if row is None:
            raise KeyError(f"No snapshot found for tick {tick}")

        w, h = row[2], row[3]
        resource_max = float(row[4])

        ownership = np.frombuffer(row[0], dtype=np.int8).reshape(w, h).copy()
        food = np.frombuffer(row[1], dtype=np.float32).reshape(w, h).copy()

        civ_rows = json.loads(row[6])
        civ_map = {
            c["civ_id"]: CivState(
                civ_id=c["civ_id"],
                name=c["name"],
                alive=c["alive"],
                discovered_techs=[None] * c["n_techs"],
            )
            for c in civ_rows
        }

        city_states = [
            CityState(
                x=city["x"], y=city["y"],
                population=city["pop"],
                military=city["military"],
                food_stock=city["food_stock"],
                last_action=city["last_action"],
                civ=civ_map[city["civ_id"]],
            )
            for city in json.loads(row[5])
        ]

        return ReplayFrame(
            steps=tick,
            running=True,
            civilizations=list(civ_map.values()),
            agents=city_states,
            grid=GridState(
                ownership=ownership,
                layers={ResourceType.FOOD: _LayerProxy(food)},
            ),
            config=SimpleNamespace(width=w, height=h, resource_max=resource_max),
            history=self._history_to(tick),
        )

    def _load_history(self) -> dict:
        empty: dict = {"tick": [], "pop_0": [], "pop_1": [], "mil_0": [], "mil_1": []}
        try:
            rows = self._con.execute("""
                SELECT tick,
                    SUM(CASE WHEN civ_id = 0 THEN pop      ELSE 0 END),
                    SUM(CASE WHEN civ_id = 1 THEN pop      ELSE 0 END),
                    SUM(CASE WHEN civ_id = 0 THEN military ELSE 0 END),
                    SUM(CASE WHEN civ_id = 1 THEN military ELSE 0 END)
                FROM events
                GROUP BY tick
                ORDER BY tick
            """).fetchall()
        except duckdb.CatalogException:
            return empty
        h: dict = {"tick": [], "pop_0": [], "pop_1": [], "mil_0": [], "mil_1": []}
        for r in rows:
            h["tick"].append(r[0])
            h["pop_0"].append(r[1] or 0.0)
            h["pop_1"].append(r[2] or 0.0)
            h["mil_0"].append(r[3] or 0.0)
            h["mil_1"].append(r[4] or 0.0)
        return h

    def _history_to(self, tick: int) -> dict:
        idx = bisect.bisect_right(self._history["tick"], tick)
        return {k: v[:idx] for k, v in self._history.items()}

    def close(self) -> None:
        self._con.close()

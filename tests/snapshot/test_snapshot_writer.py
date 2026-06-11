import os
import tempfile
from unittest.mock import MagicMock

import duckdb
import numpy as np

from civ_sim.storage.snapshot import SnapshotWriter
from civ_sim.world.resources import ResourceType


def _make_mock_grid(width=8, height=6):
    grid = MagicMock()
    grid.ownership = np.zeros((width, height), dtype=np.int8)
    food_layer = MagicMock()
    food_layer.data = np.ones((width, height), dtype=np.float32) * 42.0
    grid.layers = {ResourceType.FOOD: food_layer}
    grid.width = width
    grid.height = height
    grid.config = MagicMock()
    grid.config.resource_max = 100.0
    return grid


def _make_mock_city(civ_id=0, x=2, y=3):
    from civ_sim.agents.city import CityAgent

    city = MagicMock(spec=CityAgent)
    city.unique_id = f"city-{civ_id}-{x}-{y}"
    civ = MagicMock()
    civ.civ_id = civ_id
    city.civ = civ
    city.x = x
    city.y = y
    city.population = 100.0
    city.military = 20.0
    city.food_stock = 30.0
    city.wood_stock = 20.0
    city.mineral_stock = 15.0
    city.last_action = "gather"
    return city


def _make_mock_civ(civ_id=0):
    civ = MagicMock()
    civ.civ_id = civ_id
    civ.name = "Alpha"
    civ.alive = True
    civ.tech_level = 1
    civ.discovered_techs = ["agri"]
    return civ


def test_writer_creates_snapshots_table():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    try:
        os.unlink(db_path)  # Remove empty file so DuckDB can create fresh DB
        writer = SnapshotWriter(db_path, seed=42)
        writer.close()
        con = duckdb.connect(db_path)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        assert "snapshots" in tables
        con.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_writer_inserts_row():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    try:
        os.unlink(db_path)  # Remove empty file so DuckDB can create fresh DB
        writer = SnapshotWriter(db_path, seed=42)
        grid = _make_mock_grid()
        city = _make_mock_city()
        civ = _make_mock_civ()
        writer.write(tick=10, grid=grid, agents=[city], civilizations=[civ])
        writer.close()

        con = duckdb.connect(db_path)
        rows = con.execute("SELECT tick, seed, width, height FROM snapshots").fetchall()
        assert len(rows) == 1
        assert rows[0] == (10, 42, 8, 6)
        con.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_writer_multiple_rows():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    try:
        os.unlink(db_path)
        writer = SnapshotWriter(db_path, seed=9)
        grid = _make_mock_grid()
        for tick in (10, 20, 30):
            writer.write(tick=tick, grid=grid, agents=[], civilizations=[])
        writer.close()

        con = duckdb.connect(db_path)
        ticks = [
            r[0]
            for r in con.execute("SELECT tick FROM snapshots ORDER BY tick").fetchall()
        ]
        con.close()
        assert ticks == [10, 20, 30]
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_writer_serialises_ownership_blob():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    try:
        os.unlink(db_path)
        writer = SnapshotWriter(db_path, seed=2)
        grid = _make_mock_grid()
        grid.ownership[0, 0] = 1  # mark one tile as civ 1
        writer.write(tick=1, grid=grid, agents=[], civilizations=[])
        writer.close()

        con = duckdb.connect(db_path)
        row = con.execute("SELECT ownership, width, height FROM snapshots").fetchone()
        ownership = np.frombuffer(row[0], dtype=np.int8).reshape(row[1], row[2])
        assert ownership[0, 0] == 1
        assert ownership[1, 1] == 0
        con.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_writer_serialises_food_blob():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    try:
        os.unlink(db_path)  # Remove empty file so DuckDB can create fresh DB
        writer = SnapshotWriter(db_path, seed=1)
        grid = _make_mock_grid()
        writer.write(tick=1, grid=grid, agents=[], civilizations=[])
        writer.close()

        con = duckdb.connect(db_path)
        row = con.execute("SELECT food, width, height FROM snapshots").fetchone()
        food = np.frombuffer(row[0], dtype=np.float32).reshape(row[1], row[2])
        assert food[0, 0] == 42.0
        con.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

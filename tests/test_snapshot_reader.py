import tempfile, os
import numpy as np
import duckdb
from storage.snapshot import SnapshotWriter, SnapshotReader, ReplayFrame
from world.resources import ResourceType
from unittest.mock import MagicMock


def _write_two_snapshots(db_path):
    """Helper: write snapshots at tick 10 and 20 with no cities/civs."""
    writer = SnapshotWriter(db_path, seed=7)
    for tick in (10, 20):
        grid = MagicMock()
        grid.ownership = np.zeros((8, 6), dtype=np.int8)
        food_layer = MagicMock()
        food_layer.data = np.full((8, 6), tick, dtype=np.float32)
        grid.layers = {ResourceType.FOOD: food_layer}
        grid.config.resource_max = 100.0
        writer.write(tick=tick, grid=grid, agents=[], civilizations=[])
    writer.close()


def test_reader_ticks():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        _write_two_snapshots(db_path)
        reader = SnapshotReader(db_path)
        assert reader.ticks() == [10, 20]
        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_reader_load_returns_replay_frame():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        _write_two_snapshots(db_path)
        reader = SnapshotReader(db_path)
        frame = reader.load(10)
        assert isinstance(frame, ReplayFrame)
        assert frame.steps == 10
        assert frame.config.width == 8
        assert frame.config.height == 6
        assert frame.config.resource_max == 100.0
        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_reader_load_food_values():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        _write_two_snapshots(db_path)
        reader = SnapshotReader(db_path)
        frame = reader.load(20)
        food = frame.grid.layers[ResourceType.FOOD].data
        assert food[0, 0] == 20.0
        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_reader_history_sliced():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        _write_two_snapshots(db_path)
        reader = SnapshotReader(db_path)
        # No events table → history dicts are empty lists
        frame = reader.load(10)
        assert isinstance(frame.history, dict)
        assert "tick" in frame.history
        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

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


def test_reader_load_missing_tick_raises():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        _write_two_snapshots(db_path)
        reader = SnapshotReader(db_path)
        try:
            reader.load(999)
            assert False, "Expected KeyError"
        except KeyError:
            pass
        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_reader_ticks_empty_on_no_snapshots_table():
    import duckdb as _duckdb
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        con = _duckdb.connect(db_path)
        con.execute("CREATE TABLE events (tick INTEGER)")
        con.close()
        reader = SnapshotReader(db_path)
        assert reader.ticks() == []
        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_reader_load_city_civ_round_trip():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        writer = SnapshotWriter(db_path, seed=3)
        grid = MagicMock()
        grid.ownership = np.zeros((8, 6), dtype=np.int8)
        food_layer = MagicMock()
        food_layer.data = np.zeros((8, 6), dtype=np.float32)
        grid.layers = {ResourceType.FOOD: food_layer}
        grid.config.resource_max = 100.0

        from agents.city import CityAgent
        city = MagicMock(spec=CityAgent)
        city.unique_id = "u1"
        civ_mock = MagicMock()
        civ_mock.civ_id = 0
        city.civ = civ_mock
        city.x, city.y = 3, 4
        city.population = 75.0
        city.military = 15.0
        city.food_stock = 25.0
        city.last_action = "expand"

        civ_obj = MagicMock()
        civ_obj.civ_id = 0
        civ_obj.name = "Alpha"
        civ_obj.alive = True
        civ_obj.tech_level = 2
        civ_obj.discovered_techs = ["agri", "iron"]

        writer.write(tick=5, grid=grid, agents=[city], civilizations=[civ_obj])
        writer.close()

        reader = SnapshotReader(db_path)
        frame = reader.load(5)

        assert len(frame.agents) == 1
        c = frame.agents[0]
        assert c.x == 3
        assert c.y == 4
        assert c.population == 75.0
        assert c.military == 15.0
        assert c.food_stock == 25.0
        assert c.last_action == "expand"
        assert c.civ.civ_id == 0

        assert len(frame.civilizations) == 1
        cv = frame.civilizations[0]
        assert cv.name == "Alpha"
        assert cv.alive is True
        assert len(cv.discovered_techs) == 2

        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_reader_load_ownership_round_trip():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        writer = SnapshotWriter(db_path, seed=5)
        ownership = np.array([[0, 1], [1, -1]], dtype=np.int8)
        grid = MagicMock()
        grid.ownership = ownership
        food_layer = MagicMock()
        food_layer.data = np.zeros((2, 2), dtype=np.float32)
        grid.layers = {ResourceType.FOOD: food_layer}
        grid.config.resource_max = 100.0
        writer.write(tick=1, grid=grid, agents=[], civilizations=[])
        writer.close()

        reader = SnapshotReader(db_path)
        frame = reader.load(1)
        np.testing.assert_array_equal(frame.grid.ownership, ownership)
        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_reader_history_sliced_with_events():
    import duckdb as _duckdb
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=True) as f:
        db_path = f.name
    try:
        # Write snapshots at ticks 10 and 20
        _write_two_snapshots(db_path)

        # Inject synthetic events at ticks 5, 10, 15, 20
        con = _duckdb.connect(db_path)
        con.execute("""
            CREATE TABLE events (
                tick INTEGER, seed INTEGER, agent_id VARCHAR,
                civ_id INTEGER, action VARCHAR,
                pop REAL, military REAL,
                tech_level INTEGER, territory INTEGER, env_event VARCHAR
            )
        """)
        for tick in (5, 10, 15, 20):
            con.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
                (tick, 7, "a1", 0, "gather", float(tick * 10), float(tick), 1, 5, ""),
            )
        con.close()

        reader = SnapshotReader(db_path)

        # Frame at tick 10 should only see events at ticks <= 10
        frame10 = reader.load(10)
        assert frame10.history["tick"] == [5, 10]
        assert frame10.history["pop_0"] == [50.0, 100.0]

        # Frame at tick 20 should see all four ticks
        frame20 = reader.load(20)
        assert frame20.history["tick"] == [5, 10, 15, 20]

        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

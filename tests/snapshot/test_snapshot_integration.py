import tempfile, os
import duckdb
from civ_sim.config import SimConfig
from civ_sim.simulation.model import CivModel


def test_snapshot_written_every_n_ticks():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    os.unlink(db_path)  # Remove empty file so DuckDB can create fresh DB
    try:
        cfg = SimConfig(
            rng_seed=42, max_ticks=10, width=20, height=15,
            cities_per_civ=1, snapshot_interval=5, db_path=db_path, visualize=False,
        )
        model = CivModel(cfg)
        for _ in range(10):
            if model.running:
                model.step()

        con = duckdb.connect(db_path)
        ticks = [r[0] for r in con.execute(
            "SELECT tick FROM snapshots ORDER BY tick"
        ).fetchall()]
        con.close()
        # Must have snapped at multiples of 5: 5 and 10 (or fewer if sim ended early)
        assert len(ticks) >= 1
        assert all(t % 5 == 0 for t in ticks)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_no_snapshot_when_interval_zero():
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    os.unlink(db_path)  # Remove empty file so DuckDB can create fresh DB
    try:
        cfg = SimConfig(
            rng_seed=42, max_ticks=5, width=20, height=15,
            cities_per_civ=1, snapshot_interval=0, db_path=db_path, visualize=False,
        )
        model = CivModel(cfg)
        for _ in range(5):
            if model.running:
                model.step()

        con = duckdb.connect(db_path)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        con.close()
        assert "snapshots" not in tables
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

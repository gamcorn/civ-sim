import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from config import SimConfig


@pytest.fixture
def base_cfg():
    return SimConfig(
        width=20, height=20, num_civs=2, cities_per_civ=1,
        max_ticks=3, rng_seed=0, db_path=":memory:", visualize=False,
    )


def _fake_result(seed):
    return {
        "seed": seed, "ticks": 3, "winner": "Alpha",
        "traits_0": {"aggressiveness": 0.5, "trust": 0.5,
                     "innovation": 0.5, "tribalism": 0.5, "risk_tolerance": 0.5},
        "traits_1": {"aggressiveness": 0.5, "trust": 0.5,
                     "innovation": 0.5, "tribalism": 0.5, "risk_tolerance": 0.5},
        "techs_0": [], "techs_1": [],
    }


def test_run_single_returns_summary_keys(base_cfg):
    from simulation.runner import run_single
    result = run_single(base_cfg)
    assert set(result.keys()) >= {"seed", "ticks", "winner", "traits_0", "traits_1",
                                   "techs_0", "techs_1"}
    assert result["seed"] == base_cfg.rng_seed
    assert isinstance(result["ticks"], int)


def test_run_single_uses_memory_db_by_default(base_cfg):
    base_cfg.db_path = ":memory:"
    from simulation.runner import run_single
    result = run_single(base_cfg)
    assert result is not None


def test_run_sweep_writes_to_output_db(base_cfg):
    n = 4
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        out_db = f.name
    try:
        fake_refs = [f"ref_{i}" for i in range(n)]

        def fake_wait(refs, num_returns, timeout):
            if refs:
                return [refs[0]], refs[1:]
            return [], []

        with patch("simulation.runner.ray") as mock_ray:
            mock_ray.init = MagicMock()
            mock_ray.remote = MagicMock(return_value=MagicMock(
                remote=MagicMock(side_effect=lambda **kw: f"ref_{kw['seed']}")
            ))
            mock_ray.wait.side_effect = fake_wait
            mock_ray.get = MagicMock(side_effect=lambda ref: _fake_result(int(ref[-1])))
            mock_ray.shutdown = MagicMock()

            from simulation.runner import run_sweep
            run_sweep(n_runs=n, base_config=base_cfg, output_db=out_db, num_workers=0)

        import duckdb
        con = duckdb.connect(out_db, read_only=True)
        count = con.execute("SELECT count(*) FROM sweep_results").fetchone()[0]
        con.close()
        assert count == n
    finally:
        os.unlink(out_db)


def test_sweep_worker_uses_memory_db(base_cfg):
    from simulation.runner import _sweep_worker
    result = _sweep_worker(seed=7, base_config=base_cfg)
    assert result["seed"] == 7
    assert not os.path.exists("/tmp/civ_sweep_7.duckdb")

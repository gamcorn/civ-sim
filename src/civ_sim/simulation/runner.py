from __future__ import annotations

import copy
import importlib
import json
import logging
import types
from pathlib import Path

try:
    ray: types.ModuleType | None = importlib.import_module("ray")
except ModuleNotFoundError:
    ray = None

from civ_sim.config import SimConfig
from civ_sim.simulation.model import CivModel

logger = logging.getLogger(__name__)


def run_single(config: SimConfig, renderer=None) -> dict:
    """Run one simulation. Returns a summary dict."""
    model = CivModel(config)

    while model.running:
        model.step()  # ty: ignore[missing-argument]  # cascades from Mesa PEP-695 generics issue
        if renderer is not None and config.visualize:
            renderer.update(model)

    winner = None
    alive = [c for c in model.civilizations if c.alive]
    if len(alive) == 1:
        winner = alive[0].name
    elif len(alive) > 1:
        winner = max(alive, key=lambda c: c.total_pop).name + " (by pop)"

    return {
        "seed": config.rng_seed,
        "ticks": model.steps,
        "winner": winner,
        "traits_0": model.civilizations[0].traits.as_dict(),
        "traits_1": model.civilizations[1].traits.as_dict(),
        "techs_0": list(model.civilizations[0].discovered_techs),
        "techs_1": list(model.civilizations[1].discovered_techs),
    }


def _sweep_worker(seed: int, base_config: SimConfig) -> dict:
    """Single-run worker for Ray sweep. Uses in-memory DuckDB — no temp files."""
    cfg = copy.copy(base_config)
    cfg.rng_seed = seed
    cfg.visualize = False
    cfg.db_path = ":memory:"
    return run_single(cfg)


def run_sweep(
    n_runs: int, base_config: SimConfig, output_db: str, num_workers: int = 0
) -> None:
    """Distributed parameter sweep using Ray.

    Workers are pinned to one CPU core each (num_cpus=1) to prevent
    oversubscription on a DGX Spark with many cores. Results are streamed
    into output_db as each worker finishes rather than buffered in memory.

    Args:
        n_runs: number of simulations to run.
        base_config: template config; rng_seed is overridden per worker.
        output_db: DuckDB file to write aggregate results into.
        num_workers: Ray CPU cap (0 = auto, use all available cores).
    """
    import duckdb

    if ray is None:
        raise ImportError("Ray is required for sweep runs: pip install ray")

    logger.info("Sweep starting: %d runs, output=%s", n_runs, output_db)

    effective_workers = num_workers or getattr(base_config, "num_sweep_workers", 0)
    ray.init(
        ignore_reinit_error=True,
        num_cpus=effective_workers if effective_workers > 0 else None,
    )

    remote_worker = ray.remote(num_cpus=1)(_sweep_worker)

    # Remove a 0-byte placeholder (e.g. from tempfile) so DuckDB can create a
    # fresh valid database file at that path.
    if (
        output_db != ":memory:"
        and Path(output_db).exists()
        and Path(output_db).stat().st_size == 0
    ):
        Path(output_db).unlink()

    con = duckdb.connect(output_db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sweep_results (
            seed      INTEGER,
            ticks     INTEGER,
            winner    VARCHAR,
            traits_0  VARCHAR,
            traits_1  VARCHAR,
            techs_0   VARCHAR,
            techs_1   VARCHAR
        )
    """)

    futures = [remote_worker.remote(i, base_config) for i in range(n_runs)]

    remaining = list(futures)
    completed = 0
    errors = 0

    while remaining:
        done, remaining = ray.wait(remaining, num_returns=1, timeout=60.0)
        if not done:
            errors += 1
            logger.warning("Worker timed out after 60 s; skipping")
            remaining = remaining[1:]
            continue
        for ref in done:
            try:
                r = ray.get(ref)
                con.execute(
                    "INSERT INTO sweep_results VALUES (?,?,?,?,?,?,?)",
                    (
                        r["seed"],
                        r["ticks"],
                        r["winner"],
                        json.dumps(r["traits_0"]),
                        json.dumps(r["traits_1"]),
                        json.dumps(r["techs_0"]),
                        json.dumps(r["techs_1"]),
                    ),
                )
                completed += 1
                print(
                    f"\r  {completed}/{n_runs} runs complete"
                    f"{f'  ({errors} errors)' if errors else ''}",
                    end="",
                    flush=True,
                )
            except Exception as exc:
                errors += 1
                logger.warning("Worker error: %s", exc)

    con.close()
    ray.shutdown()
    print(
        f"\nSweep complete: {completed} runs written to {output_db}"
        + (f" ({errors} errors skipped)" if errors else "")
    )

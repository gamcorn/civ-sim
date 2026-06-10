# tests/test_logger.py
import json
import pytest
import duckdb
from civ_sim.storage.logger import EventLogger


@pytest.fixture
def mem_logger():
    logger = EventLogger(":memory:", seed=0, flush_interval=1)
    yield logger
    logger.close()


def test_log_directive_inserts_row(mem_logger):
    from civ_sim.agents.providers.council_provider import StrategicDirective
    d = StrategicDirective(
        era_goal="Expand east",
        action_weights={"gather": 0.0, "trade": 0.0, "expand": 0.8, "fortify": 0.0, "attack": 0.0, "research": 0.0},
        reasoning="Land pressure is high",
        issued_at_tick=5,
        valid_for_ticks=10,
        emergency=False,
    )
    mem_logger.log_directive(tick=5, civ_id=0, directive=d)
    rows = mem_logger._con.execute("SELECT * FROM directives").fetchall()
    assert len(rows) == 1
    tick, civ_id, era_goal, weights_json, reasoning, emergency, issued_at_tick, success = rows[0]
    assert tick == 5
    assert civ_id == 0
    assert era_goal == "Expand east"
    weights = json.loads(weights_json)
    assert weights["expand"] == 0.8
    assert emergency is False
    assert success is True


def test_log_directive_emergency_flag(mem_logger):
    from civ_sim.agents.providers.council_provider import StrategicDirective
    d = StrategicDirective(
        era_goal="Defend now",
        action_weights={"gather": 0.0, "trade": 0.0, "expand": 0.0, "fortify": 0.8, "attack": 0.0, "research": 0.0},
        reasoning="Under attack",
        issued_at_tick=12,
        valid_for_ticks=10,
        emergency=True,
    )
    mem_logger.log_directive(tick=12, civ_id=1, directive=d)
    row = mem_logger._con.execute("SELECT emergency FROM directives").fetchone()
    assert row[0] is True

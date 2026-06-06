import pytest
from storage.logger import EventLogger


@pytest.fixture
def logger():
    lg = EventLogger(db_path=":memory:", seed=42, flush_interval=5)
    yield lg
    try:
        lg.close()
    except Exception:
        pass


def _row(tick=1, agent_id="a1", civ_id=0, action="gather",
         pop=100, military=10, tech_level=0, territory=5, env_event=""):
    return dict(tick=tick, agent_id=agent_id, civ_id=civ_id, action=action,
                pop=pop, military=military, tech_level=tech_level,
                territory=territory, env_event=env_event)


def test_log_event_buffers_without_flushing():
    lg = EventLogger(":memory:", seed=1, flush_interval=10)
    for i in range(4):
        lg.log_event(**_row(tick=i))
    assert len(lg._buffer) == 4
    lg.close()


def test_flush_writes_to_db_and_clears_buffer(logger):
    for i in range(5):
        logger.log_event(**_row(tick=i))
    # flush_interval=5 triggers auto-flush on 5th insert
    assert len(logger._buffer) == 0
    rows = logger._con.execute("SELECT count(*) FROM events").fetchone()[0]
    assert rows == 5


def test_manual_flush_writes_partial_buffer(logger):
    logger.log_event(**_row(tick=1))
    logger.log_event(**_row(tick=2))
    logger.flush()
    assert len(logger._buffer) == 0
    rows = logger._con.execute("SELECT count(*) FROM events").fetchone()[0]
    assert rows == 2


def test_close_flushes_remaining_rows():
    lg = EventLogger(":memory:", seed=7, flush_interval=100)
    lg.log_event(**_row(tick=1))
    lg.log_event(**_row(tick=2))
    lg.close()
    assert len(lg._buffer) == 0


def test_rows_have_correct_seed_and_action():
    lg = EventLogger(":memory:", seed=99, flush_interval=1)
    lg.log_event(**_row(tick=3, action="attack"))
    row = lg._con.execute("SELECT seed, action FROM events").fetchone()
    assert row[0] == 99
    assert row[1] == "attack"
    lg.close()


def test_flush_on_empty_buffer_does_not_raise(logger):
    logger.flush()  # should be a no-op

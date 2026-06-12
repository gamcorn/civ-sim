from __future__ import annotations

import json
import logging

import duckdb

logger = logging.getLogger(__name__)

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    tick        INTEGER,
    seed        INTEGER,
    agent_id    VARCHAR,
    civ_id      INTEGER,
    action      VARCHAR,
    pop         INTEGER,
    military    INTEGER,
    tech_level  INTEGER,
    territory   INTEGER,
    env_event   VARCHAR
);
"""

CREATE_DIRECTIVES_SQL = """
CREATE TABLE IF NOT EXISTS directives (
    tick              INTEGER,
    civ_id            INTEGER,
    era_goal          VARCHAR,
    action_weights_json VARCHAR,
    reasoning         VARCHAR,
    emergency         BOOLEAN,
    issued_at_tick    INTEGER,
    success           BOOLEAN
);
"""

CREATE_COUNCIL_SQL = """
CREATE TABLE IF NOT EXISTS council_sessions (
    tick                INTEGER,
    seed                INTEGER,
    civ_id              INTEGER,
    emergency           BOOLEAN,
    council_off         BOOLEAN,
    state_snapshot      VARCHAR,
    sector_outputs_json VARCHAR,
    budget_output_json  VARCHAR,
    chief_output_json   VARCHAR,
    success             BOOLEAN
);
"""


class EventLogger:
    def __init__(self, db_path: str, seed: int, flush_interval: int = 10) -> None:
        self.seed = seed
        self.flush_interval = flush_interval
        self._buffer: list[tuple] = []
        self._con = duckdb.connect(db_path)
        self._con.execute(CREATE_SQL)
        self._con.execute(CREATE_DIRECTIVES_SQL)
        self._con.execute(CREATE_COUNCIL_SQL)
        logger.info("EventLogger connected: db=%s seed=%d", db_path, seed)

    def log_event(
        self,
        tick: int,
        agent_id: str,
        civ_id: int,
        action: str,
        pop: int,
        military: int,
        tech_level: int,
        territory: int,
        env_event: str,
    ) -> None:
        self._buffer.append(
            (
                tick,
                self.seed,
                agent_id,
                civ_id,
                action,
                pop,
                military,
                tech_level,
                territory,
                env_event,
            )
        )
        if len(self._buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        n = len(self._buffer)
        self._con.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
            self._buffer,
        )
        self._buffer.clear()
        logger.debug("Flushed %d events to DB", n)

    def close(self) -> None:
        self.flush()
        self._con.close()
        logger.info("EventLogger closed")

    def log_council_session(
        self,
        tick: int,
        civ_id: int,
        *,
        emergency: bool,
        council_off: bool,
        state_snapshot: str,
        sector_outputs: list[dict],
        budget_output: dict | None,
        chief_output: dict | None,
        success: bool,
    ) -> None:
        self._con.execute(
            "INSERT INTO council_sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                tick,
                self.seed,
                civ_id,
                emergency,
                council_off,
                state_snapshot,
                json.dumps(sector_outputs),
                json.dumps(budget_output) if budget_output is not None else None,
                json.dumps(chief_output) if chief_output is not None else None,
                success,
            ],
        )

    def log_directive(
        self, tick: int, civ_id: int, directive, success: bool = True
    ) -> None:
        if directive is None:
            self._con.execute(
                "INSERT INTO directives VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [tick, civ_id, None, None, None, False, tick, False],
            )
        else:
            self._con.execute(
                "INSERT INTO directives VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    tick,
                    civ_id,
                    directive.era_goal,
                    json.dumps(directive.action_weights),
                    directive.reasoning,
                    directive.emergency,
                    directive.issued_at_tick,
                    success,
                ],
            )

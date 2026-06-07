from __future__ import annotations
import duckdb


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
    issued_at_tick    INTEGER
);
"""


class EventLogger:
    def __init__(self, db_path: str, seed: int, flush_interval: int = 10):
        self.seed = seed
        self.flush_interval = flush_interval
        self._buffer: list[tuple] = []
        self._con = duckdb.connect(db_path)
        self._con.execute(CREATE_SQL)
        self._con.execute(CREATE_DIRECTIVES_SQL)

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
        self._buffer.append((
            tick, self.seed, agent_id, civ_id,
            action, pop, military, tech_level, territory, env_event,
        ))
        if len(self._buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self._con.executemany(
            "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
            self._buffer,
        )
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._con.close()

    def log_directive(self, tick: int, civ_id: int, directive) -> None:
        import json
        self._con.execute(
            "INSERT INTO directives VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                tick, civ_id, directive.era_goal,
                json.dumps(directive.action_weights),
                directive.reasoning, directive.emergency,
                directive.issued_at_tick,
            ],
        )

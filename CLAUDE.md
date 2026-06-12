# Civ-Sim — Developer Guide

## Project Overview

Research-grade agent-based civilization simulator. Two civilizations compete across a 2D resource grid; each civilization is composed of city-level agents that make strategic decisions. Results log to DuckDB for statistical analysis; live matplotlib animation shows the world in real time.

**Goal:** Study emergent cooperation, war causation, and dominant cultural strategies across 10k+ reproducible runs.

## Run Commands

```bash
# Headless run (rule-based, default)
.venv/bin/python -m civ_sim --seed 42 --ticks 500 --no-visualize

# Live visualization
.venv/bin/python -m civ_sim --seed 42 --ticks 500

# LLM provider — all civs on local vLLM
.venv/bin/python -m civ_sim --ticks 200 --no-visualize \
  --provider openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY

# Per-civ provider via YAML (LLM vs rule-based head-to-head)
.venv/bin/python -m civ_sim --ticks 200 --no-visualize --config examples/llm_vs_rule.yaml

# Anthropic API
.venv/bin/python -m civ_sim --ticks 100 --no-visualize \
  --provider anthropic \
  --model claude-haiku-4-5-20251001 \
  --api-key $ANTHROPIC_API_KEY

# Ray parameter sweep
.venv/bin/python -m civ_sim --sweep --n-runs 100 --output sweep.duckdb

# Tests
.venv/bin/pytest tests/ -v
```

## Stack

| Component | Library |
|---|---|
| Agent framework | Mesa 3.5.1 |
| Grid / resources | NumPy + Mesa PropertyLayer |
| Terrain generation | opensimplex (Perlin noise) |
| Event logging | DuckDB |
| Parameter sweeps | Ray |
| Visualization | matplotlib FuncAnimation |
| LLM (local) | openai>=1.0 (OpenAI-compatible, targets vLLM) |
| LLM (cloud) | anthropic>=0.25 |
| Config loading | pyyaml |
| Tests | pytest + pytest-asyncio |

## Architecture

```
civ-sim/
├── src/
│   └── civ_sim/
│       ├── __main__.py            # CLI entry point (python -m civ_sim)
│       ├── config.py              # SimConfig + ProviderConfig dataclasses
│       ├── replay.py              # Replay CLI
│       ├── agents/
│       │   ├── city.py            # CityAgent(Grid2DMovingAgent) — primary unit
│       │   ├── civilization.py    # CulturalTraits + Civilization
│       │   ├── decisions.py       # Weighted-scoring engine (7 actions)
│       │   └── providers/         # Swappable decision backends
│       │       ├── base.py
│       │       ├── rule_based.py
│       │       ├── openai_compat.py
│       │       ├── anthropic_provider.py
│       │       ├── council_provider.py
│       │       ├── factory.py
│       │       └── prompt.py
│       ├── simulation/
│       │   ├── model.py           # CivModel(mesa.Model)
│       │   └── runner.py          # Single run + Ray sweep
│       ├── storage/
│       │   ├── logger.py          # DuckDB event log
│       │   └── snapshot.py        # Replay snapshots
│       ├── technology/
│       │   └── discovery.py       # Threshold-based emergent tech tree
│       ├── visualization/
│       │   ├── renderer.py        # matplotlib FuncAnimation live map
│       │   └── terminal_renderer.py
│       └── world/
│           ├── grid.py            # ResourceGrid
│           ├── resources.py       # ResourceType enum
│           └── events.py          # Environmental event sampler
├── tests/
│   ├── providers/                 # Provider-specific tests
│   ├── snapshot/                  # Snapshot tests
│   └── test_*.py
├── scripts/
├── examples/
└── docs/
```

## Mesa 3.x API (Breaking Changes from 2.x)

- Grid: `OrthogonalMooreGrid([w, h])` — NOT `MultiGrid(w, h)`
- Agent activation: `self.agents.shuffle_do("step")` — NOT `RandomActivation`
- Agent base for grid: `Grid2DMovingAgent` (from `mesa.discrete_space`)
- Resources: `PropertyLayer` — attached to grid cells
- `noise2array(xs, ys)` returns shape `(len(ys), len(xs))` — must `.T` to get `(width, height)`

## Decision Engine

Seven actions: `gather`, `trade`, `expand`, `fortify`, `attack`, `research`, `recruit`.

Each tick, `choose_action(city)` scores actions using:
```
score(action) = Σ(trait_weight[action][trait] × cultural_trait_value)
              + resource_pressure_modifier(action, city)
```
Then filters by `_feasible()` (checks prerequisites) and picks the max.

- `gather` is always feasible; active harvest is labor-capped at `population × work_rate` total units per tick
- `fortify` is always feasible; builds `city.fortification` (float) from minerals + wood; decays at `fortification_decay` rate per tick; used in combat damage reduction
- `trade` requires a city within 30 tiles (any civ) AND `relations >= trade_relation_threshold` (-0.5 default)
- `expand` requires an unclaimed tile within 3 tiles AND `wood_stock >= expand_wood_cost`
- `attack` requires an enemy city within 25 tiles, `military >= 2` (or 5 if no territorial threat), AND `mineral_stock >= attack_mineral_cost`
- `research` requires `wood_stock >= research_wood_cost` AND `mineral_stock >= research_mineral_cost`
- `recruit` requires `population > initial_pop + recruit_pop_cost` AND `mineral_stock >= recruit_mineral_cost`; drafts population into military respecting the `initial_pop` civilian floor

## LLM Provider System

Each `Civilization` owns one `DecisionProvider`. The model tick is two-phase:

1. **Decide phase** — `asyncio.run(_dispatch_decisions())` batches all cities by provider and runs LLM providers concurrently via `asyncio.gather(return_exceptions=True)`.
2. **Execute phase** — `agents.shuffle_do("step")` reads each city's `_pending_action`.

**Fallback chain (never crashes):**
- LLM timeout or API exception → rule-based fallback pre-computed before the call
- Hallucinated/invalid response → `parse_response()` falls back
- `_pending_action is None` on step → `choose_action(self)` inline

**ProviderConfig fields:**
```python
type: str = "rule_based"           # "rule_based" | "openai_compatible" | "anthropic"
model: str = ""
base_url: str = "http://localhost:8000/v1"
api_key: str = "EMPTY"
temperature: float = 0.2
max_tokens: int = 10
timeout: float = 5.0
```

**YAML config format (`--config path/to/config.yaml`):**
```yaml
civ_providers:
  - type: openai_compatible
    model: meta-llama/Llama-3.1-70B-Instruct
    base_url: http://localhost:8000/v1
    api_key: EMPTY
    timeout: 5.0
  - type: rule_based
```

## Logging

Every module uses `logging.getLogger(__name__)`. Logs are written to `civ_sim.log` in the working directory when `--log-level` is passed:

```bash
.venv/bin/python -m civ_sim --seed 42 --ticks 500 --no-visualize --log-level DEBUG
```

Format: `2026-01-01T12:00:00  INFO      civ_sim.simulation.model  CivModel initialised: seed=42 civs=2 size=80x60`

**Level guide:**
- `DEBUG` — every tick, action choices, council decisions, JSON parse attempts
- `INFO` — city founded/captured/settled, tech discoveries, disease events, sweep progress
- `WARNING` — LLM API failures (fallback to rule-based), worker timeouts, council fallbacks
- `ERROR` — assert-replacement guards (should never fire in normal operation)

The root logger level and handler are configured in `__main__._configure_logging()`. If no `--log-level` is passed, no file handler is attached (silent).

## vLLM on DGX Spark

```bash
# Start vLLM (terminal 1)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85

# Run simulation against it (terminal 2)
.venv/bin/python -m civ_sim --seed 42 --ticks 200 --no-visualize \
  --provider openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct

# Query action distribution
duckdb results.duckdb \
  "SELECT civ_id, action, count(*) FROM events GROUP BY civ_id, action ORDER BY civ_id, count(*) DESC"
```

Recommended models: `meta-llama/Llama-3.1-70B-Instruct` or `Qwen2.5-72B-Instruct`.

## Key Design Decisions

- **Food economy:** cities have a `food_stock` buffer. Passive harvest from city tile (3 units/tick) + active gather from owned tiles in radius, capped by `population × work_rate`. `military_upkeep` and `food_per_person` are intentionally small (0.02 and 0.05) to avoid gather dominating.
- **Labor-limited gather:** `_do_gather` tracks total raw extraction across tiles; loop exits early once `work_done >= population × work_rate`. Larger cities harvest more; a city with 50 pop cannot out-gather one with 500.
- **Fortification as a stat:** `city.fortification` is a float (max `max_fortification`). `_do_fortify` converts minerals+wood into fortification points; `_do_attack` uses `target.fortification / max_fortification` for pillage damage reduction. Fortification decays multiplicatively each tick (`fortification_decay = 0.005`) — cities must keep fortifying to stay defended.
- **RECRUIT action:** 7th action. Converts population surplus (above `initial_pop` floor) into military at mineral cost. Stochastic rounding matches the rest of the codebase. Scorer favors recruit when enemy military exceeds own; capped at +0.6 modifier.
- **Civilization relations:** `model.relations: dict[tuple[int,int], float]` indexed by `(min_id, max_id)` — always symmetric. Updated by trade (+0.05), attack (−0.3), and city capture (−0.5). Decays 0.002/tick toward neutral. Blocks trade feasibility when `rel < −0.5`. Shown in council intel report (`| relations {rel:+.2f}` per enemy) so LLMs can reason about alliances and trade viability.
- **Council P2 awareness (council_prompts.py):** The council state snapshot and schemas expose all P2 mechanics. `build_civ_state_snapshot()` includes `Avg fortification: N` in the own-block and `| relations {rel:+.2f}` per enemy in the intel report. War minister `MINISTER_SPECS` lists `["attack", "fortify", "recruit"]`. `CHIEF_SCHEMA_DICT`, `CHIEF_SCHEMA`, `CHIEF_LITE_SCHEMA_DICT`, and `CHIEF_LITE_SCHEMA` all require all 7 action keys including `"recruit"`.
- **Trade surplus modifier** uses `food_stock` (not grid tile level) to correctly detect surplus.
- **Fortify modifier** scales proportionally with enemy/self military ratio — not a flat bonus.
- **Expand modifier** is capped at 0.5 so it doesn't permanently dominate other actions.
- **`asyncio.run()` from synchronous `step()`** is safe: Mesa has no outer event loop.
- **`return_exceptions=True`** in `asyncio.gather` prevents a misbehaving provider from crashing the tick.

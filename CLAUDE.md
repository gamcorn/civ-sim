# Civ-Sim — Developer Guide

## Project Overview

Research-grade agent-based civilization simulator. Two civilizations compete across a 2D resource grid; each civilization is composed of city-level agents that make strategic decisions. Results log to DuckDB for statistical analysis; live matplotlib animation shows the world in real time.

**Goal:** Study emergent cooperation, war causation, and dominant cultural strategies across 10k+ reproducible runs.

## Run Commands

```bash
# Headless run (rule-based, default)
.venv/bin/python main.py --seed 42 --ticks 500 --no-visualize

# Live visualization
.venv/bin/python main.py --seed 42 --ticks 500

# LLM provider — all civs on local vLLM
.venv/bin/python main.py --ticks 200 --no-visualize \
  --provider openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY

# Per-civ provider via YAML (LLM vs rule-based head-to-head)
.venv/bin/python main.py --ticks 200 --no-visualize --config examples/llm_vs_rule.yaml

# Anthropic API
.venv/bin/python main.py --ticks 100 --no-visualize \
  --provider anthropic \
  --model claude-haiku-4-5-20251001 \
  --api-key $ANTHROPIC_API_KEY

# Ray parameter sweep
.venv/bin/python main.py --sweep --n-runs 100 --output sweep.duckdb

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
├── config.py                  # SimConfig + ProviderConfig dataclasses
├── main.py                    # CLI entry point
├── world/
│   ├── grid.py                # ResourceGrid (OrthogonalMooreGrid + PropertyLayers)
│   ├── resources.py           # ResourceType enum
│   └── events.py              # Environmental event sampler
├── agents/
│   ├── civilization.py        # CulturalTraits + Civilization (not a Mesa agent)
│   ├── city.py                # CityAgent(Grid2DMovingAgent) — primary unit
│   ├── decisions.py           # Weighted-scoring engine (6 actions)
│   └── providers/             # Swappable decision backends
│       ├── base.py            # DecisionProvider ABC
│       ├── rule_based.py      # RuleBasedProvider (wraps decisions.py)
│       ├── openai_compat.py   # OpenAICompatibleProvider (vLLM/Ollama/NIM/OpenAI)
│       ├── anthropic_provider.py  # AnthropicProvider
│       ├── factory.py         # create_provider(ProviderConfig)
│       └── prompt.py          # build_prompt(), parse_response(), SYSTEM_PROMPT
├── technology/
│   └── discovery.py           # Threshold-based emergent tech tree
├── simulation/
│   ├── model.py               # CivModel(mesa.Model)
│   └── runner.py              # Single run + Ray sweep
├── storage/
│   └── logger.py              # DuckDB event log (one row per city per tick)
├── visualization/
│   └── renderer.py            # matplotlib FuncAnimation live map
└── tests/
    ├── conftest.py             # make_mock_city() fixture
    ├── test_model_dispatch.py
    └── providers/              # Provider-specific tests
```

## Mesa 3.x API (Breaking Changes from 2.x)

- Grid: `OrthogonalMooreGrid([w, h])` — NOT `MultiGrid(w, h)`
- Agent activation: `self.agents.shuffle_do("step")` — NOT `RandomActivation`
- Agent base for grid: `Grid2DMovingAgent` (from `mesa.discrete_space`)
- Resources: `PropertyLayer` — attached to grid cells
- `noise2array(xs, ys)` returns shape `(len(ys), len(xs))` — must `.T` to get `(width, height)`

## Decision Engine

Six actions: `gather`, `trade`, `expand`, `fortify`, `attack`, `research`.

Each tick, `choose_action(city)` scores actions using:
```
score(action) = Σ(trait_weight[action][trait] × cultural_trait_value)
              + resource_pressure_modifier(action, city)
```
Then filters by `_feasible()` (checks prerequisites) and picks the max.

- `gather` is always feasible; `fortify` is always feasible
- `trade` requires an enemy city within 15 tiles
- `expand` requires an unclaimed tile within 3 tiles
- `attack` requires an enemy city within 10 tiles and `military >= 5`
- `research` requires `wood > 10` and `minerals > 5` on city tile

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

## vLLM on DGX Spark

```bash
# Start vLLM (terminal 1)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85

# Run simulation against it (terminal 2)
python main.py --seed 42 --ticks 200 --no-visualize \
  --provider openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct

# Query action distribution
duckdb results.duckdb \
  "SELECT civ_id, action, count(*) FROM events GROUP BY civ_id, action ORDER BY civ_id, count(*) DESC"
```

Recommended models: `meta-llama/Llama-3.1-70B-Instruct` or `Qwen2.5-72B-Instruct`.

## Key Design Decisions

- **Food economy:** cities have a `food_stock` buffer. Passive harvest from city tile (5 units/tick) + active gather from all owned tiles in radius. `military_upkeep` and `food_per_person` are intentionally small (0.02 and 0.05) to avoid gather dominating.
- **Trade surplus modifier** uses `food_stock` (not grid tile level) to correctly detect surplus.
- **Fortify modifier** scales proportionally with enemy/self military ratio — not a flat bonus.
- **Expand modifier** is capped at 0.5 so it doesn't permanently dominate other actions.
- **`asyncio.run()` from synchronous `step()`** is safe: Mesa has no outer event loop.
- **`return_exceptions=True`** in `asyncio.gather` prevents a misbehaving provider from crashing the tick.

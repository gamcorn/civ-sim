# Civ-Sim

An agent-based civilization simulator where two civilizations compete over a resource grid. Each civilization is made up of city agents that make strategic decisions — using a rule-based engine, a local LLM via vLLM, or the Anthropic API.

Use it to run reproducible experiments: does cooperation emerge? Is war inevitable? What cultural traits dominate over 10,000 runs?

---

## Requirements

- Python 3.12+
- (Optional) NVIDIA GPU + vLLM for local LLM decisions

---

## Installation

```bash
git clone <repo-url>
cd civ-sim

python3 -m venv .venv
source .venv/bin/activate
```

Install all dependencies (recommended):

```bash
pip install -r requirements.txt
```

Or install only what you need using the optional dependency groups:

```bash
pip install -e .               # core only (no visualization, no Ray sweep)
pip install -e ".[viz]"        # + matplotlib live map
pip install -e ".[sweep]"      # + Ray for parameter sweeps
pip install -e ".[dev]"        # + pytest for development
pip install -e ".[all]"        # everything
```

Verify the install:

```bash
python main.py --seed 42 --ticks 10 --no-visualize
```

Expected output ends with `Simulation ended at tick 10`.

---

## Quick Start

### Headless run (fastest)

```bash
python main.py --seed 42 --ticks 500 --no-visualize
```

### Live map

```bash
python main.py --seed 42 --ticks 500
```

A matplotlib window opens showing territory ownership, food levels, and population/military charts updated each tick.

### Terminal map (SSH / headless)

```bash
python main.py --seed 42 --ticks 500 --terminal-viz
```

Renders the world directly in the terminal using ANSI color codes — no display server needed. Shows a color-coded territory map, per-civilization stats (population, military, food, territory, top actions), and a population bar per civilization. Falls back to this renderer automatically if matplotlib cannot open a display window.

### Query results

```bash
duckdb results.duckdb "SELECT action, count(*) FROM events GROUP BY action ORDER BY count(*) DESC"
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--seed INT` | 42 | RNG seed for reproducibility |
| `--ticks INT` | 500 | Maximum simulation ticks |
| `--width INT` | 80 | Grid width |
| `--height INT` | 60 | Grid height |
| `--cities INT` | 4 | Cities per civilization |
| `--db PATH` | results.duckdb | DuckDB output file |
| `--no-visualize` | off | Disable all visualization |
| `--terminal-viz` | off | ANSI terminal renderer — works over SSH without a display |
| `--sweep` | off | Run Ray parameter sweep |
| `--n-runs INT` | 100 | Runs for sweep |
| `--output PATH` | sweep.duckdb | Sweep results file |
| `--provider` | rule_based | Decision backend: `rule_based`, `openai_compatible`, `anthropic` |
| `--model STR` | meta-llama/Llama-3.1-70B-Instruct | Model name for LLM provider |
| `--base-url URL` | http://localhost:8000/v1 | Endpoint for OpenAI-compatible provider |
| `--api-key STR` | EMPTY | API key (use EMPTY for local vLLM/Ollama) |
| `--config PATH` | — | YAML file with per-civilization provider config |
| `--batch-mode` | off | Send all prompts in one `/v1/completions` call (vLLM DGX mode) |
| `--max-concurrent INT` | 64 | Max concurrent LLM requests when batch-mode is off |
| `--prompt-template STR` | — | Chat template for completions API (`{system}` / `{user}` placeholders); defaults to Llama-3.1 format |
| `--workers INT` | 0 | Ray worker processes for sweep (0 = one per CPU core) |
| `--grid-backend STR` | numpy | Array backend for resource grid: `numpy` or `cupy` |
| `--snapshot-interval INT` | 0 | Write a world snapshot every N ticks for later replay (0 = off) |

---

## LLM Providers

City decisions can be driven by a language model. Each civilization can independently use a different backend.

### Local model via vLLM (DGX Spark / any NVIDIA GPU)

**Terminal 1 — start the server:**

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

**Terminal 2 — run the simulation:**

```bash
python main.py --seed 42 --ticks 200 --no-visualize \
  --provider openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY
```

Works with any OpenAI-compatible server: vLLM, Ollama, LM Studio, NVIDIA NIM, or OpenAI itself.

### Anthropic API

```bash
python main.py --seed 42 --ticks 200 --no-visualize \
  --provider anthropic \
  --model claude-haiku-4-5-20251001 \
  --api-key $ANTHROPIC_API_KEY
```

### LLM vs rule-based head-to-head (per-civilization config)

Create `config.yaml`:

```yaml
civ_providers:
  - type: openai_compatible
    model: meta-llama/Llama-3.1-70B-Instruct
    base_url: http://localhost:8000/v1
    api_key: EMPTY
    timeout: 5.0
  - type: rule_based
```

Run:

```bash
python main.py --seed 42 --ticks 200 --no-visualize --config config.yaml
```

Query who won and what actions each side favored:

```bash
duckdb results.duckdb "
  SELECT civ_id, action, count(*) as n
  FROM events
  GROUP BY civ_id, action
  ORDER BY civ_id, n DESC
"
```

The LLM provider falls back silently to rule-based logic on any timeout, API error, or invalid response — the simulation never halts due to a bad LLM call.

---

## Replay

Long simulations can be replayed after the fact by writing periodic world snapshots during the run and playing them back interactively with `replay.py`.

### Step 1 — Run with snapshots enabled

Add `--snapshot-interval N` to write a full world snapshot every N ticks:

```bash
python main.py --seed 42 --ticks 5000 --no-visualize \
  --snapshot-interval 50 --db run.duckdb
```

This stores grid state (territory ownership + food levels), city positions, population, military, and civilization state as rows in the same `run.duckdb` file alongside the event log. A 5000-tick run at interval 50 writes ~100 snapshots — typically a few megabytes.

### Step 2 — Replay

```bash
python replay.py run.duckdb
```

The renderer is chosen automatically: matplotlib if `$DISPLAY` is set, terminal otherwise.

**Options:**

```bash
python replay.py run.duckdb --renderer terminal     # force terminal
python replay.py run.duckdb --renderer matplotlib   # force matplotlib
python replay.py run.duckdb --from-tick 2000        # start near tick 2000
python replay.py run.duckdb --speed 4               # start at 4× speed
```

### Keyboard controls

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `+` / `=` | Double playback speed (max 32×) |
| `-` | Halve playback speed (min 0.125×) |
| `→` | Skip forward 10 snapshots |
| `←` | Skip back 10 snapshots |
| `q` | Quit |

### Inspect snapshots directly

```bash
duckdb run.duckdb "SELECT tick, width, height FROM snapshots ORDER BY tick"
```

---

## Parameter Sweep

Run many seeds in parallel with Ray and collect results into a single DuckDB file:

```bash
python main.py --sweep --n-runs 1000 --output sweep.duckdb --no-visualize
```

Control parallelism explicitly (useful on shared machines):

```bash
python main.py --sweep --n-runs 1000 --workers 8 --output sweep.duckdb --no-visualize
```

Analyze results:

```bash
duckdb sweep.duckdb "
  SELECT seed, max(tick) as end_tick,
         max(CASE WHEN civ_id=0 THEN pop END) as pop_alpha,
         max(CASE WHEN civ_id=1 THEN pop END) as pop_beta
  FROM events
  GROUP BY seed
  ORDER BY end_tick DESC
  LIMIT 20
"
```

---

## DGX Spark / Multi-GPU

Two extra flags squeeze more throughput out of a DGX Spark node.

**Batch LLM inference** — sends all city prompts in a single `/v1/completions` request instead of N concurrent chat calls. vLLM schedules them as a true GPU batch, eliminating N-1 HTTP round-trips per tick:

```bash
python main.py --ticks 200 --no-visualize \
  --provider openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --base-url http://localhost:8000/v1 \
  --batch-mode
```

**GPU resource grid** — moves all NumPy array operations (terrain, resource regeneration, territory tracking) onto the GPU via CuPy. Install CuPy first:

```bash
pip install cupy-cuda12x   # match your CUDA version
```

Then pass `--grid-backend cupy` to any run or sweep:

```bash
python main.py --seed 42 --ticks 500 --no-visualize --grid-backend cupy
```

Both flags can be combined. For a full DGX sweep:

```bash
python main.py --sweep --n-runs 10000 --output dgx_sweep.duckdb \
  --no-visualize --grid-backend cupy \
  --provider openai_compatible \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --batch-mode
```

---

## Tests

```bash
python -m pytest tests/ -v
```

All 195 tests should pass. The suite covers the grid, events, logger, civilization state, decision engine, city actions and lifecycle, tech tree, all three LLM providers (mocked), provider factory, model dispatch loop, batch completions path, Ray sweep runner, grid backend abstraction, snapshot writer/reader round-trips, CivModel snapshot integration, and replay player helpers and renderer duck-typing.

---

## Project Structure

```
civ-sim/
├── config.py             # All simulation parameters
├── main.py               # CLI entry point
├── world/                # Resource grid + environmental events
├── agents/
│   ├── city.py           # CityAgent — the primary simulation unit
│   ├── civilization.py   # Civilization state + cultural traits
│   ├── decisions.py      # Rule-based weighted scoring engine
│   └── providers/        # Swappable LLM / rule-based backends
├── technology/           # Emergent tech tree
├── simulation/           # Mesa model + Ray sweep runner
├── storage/              # DuckDB event logger + snapshot writer/reader
├── visualization/        # Live matplotlib + ANSI terminal renderers
├── replay.py             # Interactive replay player for completed runs
└── tests/                # pytest suite
```

See `CLAUDE.md` for architecture details, Mesa 3.x gotchas, and design decisions.

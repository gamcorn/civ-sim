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

A matplotlib window opens with a seven-panel dashboard updated each tick — see [Visualization Dashboard](#visualization-dashboard) for details.

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

## Simulation Dynamics

Several mechanics keep the simulation in motion across long runs (1000–5000 ticks) rather than settling into a frozen partition.

### Environmental events

Events are sampled each tick and applied before city decisions:

| Event | Probability/tick | Effect |
|---|---|---|
| Drought | 4% | Halves food in a random 10×10 patch |
| Disease | 2.5% baseline | Epidemic with random transmission rate β — see below |
| Mineral boom | 0.4% | Triples minerals in a random 6×6 patch |
| Climate shift | 0.2% | Suppresses global food regeneration for 20 ticks |

**Disease / pandemic model.** Each outbreak rolls a transmission rate β uniformly from [0.1, 3.0]. Mortality per city is `min(85%, β × base_rate × proximity_factor)`, where `proximity_factor` grows by 25% for each same-civilization city within 20 tiles. A β=0.1 event is a mild flu (< 1% loss); a β=3.0 event with 7 neighbors nearby can wipe out over 80% of a city's population. Disease probability also scales with land occupation: `base_prob × (1 + occupation × 3.0)` — the more territory a civilization controls, the greater the zoonotic spillover risk, matching the real-world pattern where agricultural expansion into animal habitats drives pandemic emergence.

### Military attrition

Cities with more than 50 military units lose `int(military × 2%)` units per tick, regardless of action. This prevents indefinite arms parity — a civilization that stops fortifying will bleed out over hundreds of ticks, giving rivals a window to attack. Cities below 50 military are exempt from decay so early-game expansion is not disrupted.

### Border tile reversion

Each tick, tiles on the contested border between two civilizations have a 2% chance of reverting to unclaimed. This keeps the `expand` action viable across the full run (not just the early land-grab) and creates ongoing territorial flux at contact lines even without direct combat.

City home tiles are protected from reversion — a city can never lose its own founding tile this way.

### Population and expansion

Growth rate is dynamic rather than fixed. Each tick a city grows by `rate_max × food_ratio × demo_factor`, where `food_ratio` is how full the city tile is relative to half-capacity, and `demo_factor` falls linearly as total civilization population approaches 3,000 (floored at 0.1× so growth never fully stops). This reproduces the demographic transition: small, resource-rich civilizations expand rapidly; large, land-saturated ones stagnate.

When a city's population hits the cap (500 by default) it automatically sends settlers to found a daughter city on the nearest unclaimed frontier tile at least 6 tiles from any existing city. Settling depletes 50% of the food on tiles within radius 3 of the new site, creating a local resource shock that prevents immediate chaining. A 50-tick cooldown applies before the same city can settle again. Each civilization is capped at 20 cities from voluntary founding, though military conquest can push the count higher.

### Combat and city capture

Attack is feasible when an enemy city is within **25 Manhattan tiles** and the attacker has at least 5 military. The scoring engine favors attack when `civ_total_military > enemy_total_military × 0.8` — any slight edge is enough. Trade is feasible within **30 tiles**.

**Resource-driven conflict.** Civilizations under food stress don't just starve — they first seek land (EXPAND scores higher when food stock is low) and then turn to war (ATTACK scores spike under severe scarcity). Cities that detect enemy territory encroaching within 5 tiles get a defensive attack bonus and can attack with as few as 2 military, regardless of overall civ strength.

**City capture.** When a defeated city's population falls below 30% of its founding value, it changes hands: the city, its home tile, and all surrounding territory claimed in the assault transfer to the attacker's civilization. Captures appear as `action=capture` events in the DuckDB log and can dramatically shift the balance mid-run.

Victories capture territory around the defeated city; defeats cost the attacker 25% of its military. Either outcome shifts the balance for subsequent turns.

---

## Visualization Dashboard

The live matplotlib window (`python main.py --seed 42 --ticks 500`) renders seven panels simultaneously.

### World map *(left, full height)*

Territory ownership colored by civilization (blue / red / green / purple); brightness encodes local food level — dark tiles are depleted, bright tiles are rich. City markers are scaled by population.

Two overlay rings appear on top of city markers when a city is under stress:

| Ring color | Meaning | Duration |
|---|---|---|
| Orange | Hit by the current epidemic | 8 ticks after impact |
| Yellow | Famine — food stockpile empty | While `food_stock < 0.1` |

### Population *(middle top)*

Per-civilization population over time. Dotted vertical lines mark every epidemic event; line color encodes transmission rate β (green = mild, red = catastrophic), making it easy to correlate population dips with specific outbreaks.

### Military *(middle center)*

Per-civilization total military strength over time. Attrition and combat losses appear as drops; fortify actions appear as recoveries.

### City count *(middle bottom)*

Number of cities per civilization over time. Jumps up on a settle or capture event; drops when a city collapses from starvation or combat.

### Total grid resources *(right top)*

Three lines — food (green), minerals (grey), wood (brown) — summed across the entire map each tick. A sustained downward trend means civilizations are consuming faster than the world regenerates; a floor indicates the regen/consumption equilibrium.

### Food on owned territory *(right center)*

Per-civilization sum of food on all tiles the civilization owns. Unlike the global resource chart this reflects each civ's actual food wealth. Diverging lines indicate one civilization is winning the land and resource competition; a sudden drop can precede a famine-driven war.

### Epidemic log *(right bottom)*

One dot per outbreak at position `(tick, β)`:

| Visual property | Encodes |
|---|---|
| Dot size | Lives lost in that outbreak — larger = more deaths |
| Dot color | β transmission rate — green (mild) → red (catastrophic) |
| Number above dot | Exact death toll |

Reference lines at β = 1.0 and β = 2.0 divide the chart into mild / severe / catastrophic bands. This panel makes it possible to distinguish a high-β epidemic that hit a small population (small dot, red color) from a moderate-β pandemic that swept a large, dense empire (large dot, yellow color).

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

All 202 tests should pass. The suite covers the grid, events, logger, civilization state, decision engine, city actions and lifecycle, tech tree, all three LLM providers (mocked), provider factory, model dispatch loop, batch completions path, Ray sweep runner, grid backend abstraction, snapshot writer/reader round-trips, CivModel snapshot integration, and replay player helpers and renderer duck-typing.

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

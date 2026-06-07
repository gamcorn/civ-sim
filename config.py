from dataclasses import dataclass, field


@dataclass
class ProviderConfig:
    type: str = "rule_based"          # "rule_based" | "openai_compatible" | "anthropic"
    model: str = ""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.2
    max_tokens: int = 10
    timeout: float = 5.0
    # DGX Spark batching
    use_completions_api: bool = False  # send all prompts in one /v1/completions call
    max_concurrent: int = 64           # semaphore cap when chat API is used
    prompt_template: str = ""          # chat template; empty = Llama-3.1 default


@dataclass
class SimConfig:
    # World
    width: int = 80
    height: int = 60
    # Civilizations
    num_civs: int = 2
    cities_per_civ: int = 4
    # Simulation
    max_ticks: int = 500
    rng_seed: int = 42
    # Resource regeneration per tick (fraction of max)
    food_regen: float = 0.04
    water_regen: float = 0.02
    wood_regen: float = 0.015
    mineral_regen: float = 0.0   # non-renewable by default
    resource_max: float = 100.0
    # Population
    initial_pop: int = 100
    pop_cap: int = 500               # hard ceiling per city
    pop_growth_rate: float = 0.003   # per tick when food is sufficient
    pop_starvation_rate: float = 0.04
    food_per_person: float = 0.05    # food consumed per person per tick
    # Military
    initial_military: int = 10
    military_upkeep: float = 0.02    # food per military unit per tick
    # Harvest
    harvest_radius: int = 5          # tiles from city that gather action reaches
    # Technology multipliers on production/military
    tech_food_bonus: float = 0.2     # +20% food per tech discovered
    tech_military_bonus: float = 0.3
    # Environmental events (probability per tick)
    drought_prob: float = 0.04
    disease_prob: float = 0.025
    mineral_boom_prob: float = 0.004
    climate_shift_prob: float = 0.002
    # Border dynamics
    border_reversion_prob: float = 0.02
    # Cultural trait init ranges: (min, max)
    trait_range: tuple = (0.1, 0.9)
    # Logging
    db_path: str = "results.duckdb"
    db_flush_interval: int = 10
    snapshot_interval: int = 0   # 0 = disabled; write snapshot every N ticks
    # Visualization
    visualize: bool = True
    viz_interval_ms: int = 100
    civ_providers: list = field(
        default_factory=lambda: [ProviderConfig(), ProviderConfig()]
    )
    # DGX Spark sweep / grid
    num_sweep_workers: int = 0   # Ray workers; 0 = auto (one per CPU core)
    grid_backend: str = "numpy"  # "numpy" | "cupy"

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
    # Council provider settings
    directive_period: int = 10         # ticks between scheduled council sessions
    max_rounds: int = 2                # deliberation rounds per council session
    sector_model: str = ""             # model for sector ministers (falls back to model)
    chief_model: str = ""              # model for chief of staff (falls back to model)
    emergency_triggers: bool = True    # allow out-of-schedule emergency councils
    emergency_cooldown_ticks: int = 3  # min ticks between emergency sessions
    council_off: bool = False          # skip minister debate; single chief call only
    chief_timeout: float = 0.0         # chief-of-staff timeout; 0 = timeout * 3
    guided_json: bool = False          # use vLLM guided JSON (constrained decoding)


@dataclass
class SimConfig:
    # World
    width: int = 100
    height: int = 80
    # Civilizations
    num_civs: int = 2
    cities_per_civ: int = 2
    # Simulation
    max_ticks: int = 500
    rng_seed: int = 42
    # Resource regeneration per tick (fraction of max)
    food_regen: float = 0.04
    water_regen: float = 0.015
    wood_regen: float = 0.01
    mineral_regen: float = 0.005  # slow renewal keeps research viable
    resource_max: float = 100.0
    # Population
    initial_pop: int = 50
    pop_cap: int = 1000             # hard ceiling per city
    pop_growth_rate_max: float = 0.012  # max growth rate (small civ, abundant food)
    pop_demographic_cap: int = 10000   # civ total pop at which growth → 10% of max; realistic for game scale
    pop_starvation_rate: float = 0.04
    food_per_person: float = 0.05    # food consumed per person per tick
    capture_threshold: float = 0.3   # city captured when pop < initial_pop × this
    settle_cooldown: int = 50        # ticks a city must wait before settling again
    settle_land_drain: float = 0.5   # fraction of food drained on tiles near a new city
    max_cities_per_civ: int = 100     # hard ceiling on city count per civilization
    # Military
    initial_military: int = 10
    military_upkeep: float = 0.08    # food per military unit per tick
    # Stockpile initial values
    initial_wood_stock: float = 20.0
    initial_mineral_stock: float = 20.0
    # Wood/mineral upkeep per tick
    wood_per_person: float = 0.01
    mineral_per_person: float = 0.005
    mineral_per_military: float = 0.02
    wood_per_military: float = 0.01
    # Shortage penalties
    wood_shortage_rate: float = 0.02
    mineral_shortage_rate: float = 0.03
    # Action costs (stockpile-based)
    fortify_mineral_cost: float = 8.0
    fortify_wood_cost: float = 4.0
    expand_wood_cost: float = 5.0
    settle_wood_cost: float = 20.0
    settle_mineral_cost: float = 10.0
    research_wood_cost: float = 8.0
    research_mineral_cost: float = 5.0
    attack_mineral_cost: float = 3.0
    # War economy
    fortify_defense_bonus: float = 0.8
    max_fortification: float = 100.0      # cap for city.fortification
    fortification_decay: float = 0.005    # per-tick fractional decay of fortification
    recruit_pop_cost: int = 10         # population drafted per recruit action
    recruit_mineral_cost: float = 3.0  # minerals for equipment per recruit action
    recruit_military_ratio: float = 1.0  # military units gained per person drafted
    battle_pillage_rate: float = 0.25
    capture_reconstruct_wood: float = 15.0
    capture_reconstruct_mineral: float = 10.0
    # Harvest
    harvest_radius: int = 5          # tiles from city that gather action reaches
    work_rate: float = 2.0   # harvest units a person can gather per tick
    # Science-point accrual per unit of resource spent in research
    science_per_wood: float = 1.0
    science_per_mineral: float = 1.5
    # Environmental events (probability per tick)
    drought_prob: float = 0.04
    disease_prob: float = 0.025        # baseline when no land is occupied
    disease_land_scale: float = 3.0   # multiplier added per unit of occupation ratio
    mineral_boom_prob: float = 0.004
    climate_shift_prob: float = 0.002
    # Border dynamics
    border_reversion_prob: float = 0.02
    # Labor economy
    cultivate_wood_cost:    float = 3.0
    cultivate_mineral_cost: float = 2.0
    mine_mineral_cost:      float = 3.0
    woodcut_wood_cost:      float = 2.0
    labor_ratio_delta:      float = 0.05
    degradation_rate:       float = 0.002
    recovery_rate:          float = 0.001
    efficiency_max:  float = 3.0
    # Diplomatic relations
    relation_decay: float = 0.002           # per-tick decay toward 0
    trade_relation_bonus: float = 0.05      # relations gain per trade
    attack_relation_penalty: float = 0.3   # relations loss per attack
    capture_relation_penalty: float = 0.5  # relations loss on city capture
    trade_relation_threshold: float = -0.5 # below this, trade is blocked
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
    # Intel quality for council providers (0 = perfect, 1 = very noisy)
    fog_of_war: float = 0.0
    # DGX Spark sweep / grid
    num_sweep_workers: int = 0   # Ray workers; 0 = auto (one per CPU core)
    grid_backend: str = "numpy"  # "numpy" | "cupy"

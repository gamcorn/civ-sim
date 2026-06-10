from __future__ import annotations
import asyncio
import math
import random as stdlib_random

import mesa
import numpy as np

from civ_sim.config import SimConfig
from civ_sim.world.grid import ResourceGrid
from civ_sim.world.events import EventSampler
from civ_sim.world.resources import ResourceType
from civ_sim.agents.civilization import Civilization, CulturalTraits
from civ_sim.agents.city import CityAgent
from civ_sim.technology.discovery import TechEngine
from civ_sim.storage.logger import EventLogger


class CivModel(mesa.Model):
    """Top-level Mesa model: owns the world, all agents, and the event log."""

    def __init__(self, config: SimConfig):
        super().__init__(rng=config.rng_seed)
        self.config = config
        self._np_rng = np.random.default_rng(config.rng_seed)

        # World
        self.grid = ResourceGrid(config.width, config.height, config, self.random)

        # Civilizations
        self.civilizations: list[Civilization] = self._create_civs()

        # Place cities on the grid
        self._place_cities()

        # Sub-systems
        self.event_sampler = EventSampler(config, self.random)
        self.tech_engine = TechEngine()
        self.logger = EventLogger(config.db_path, config.rng_seed, config.db_flush_interval)

        from civ_sim.storage.snapshot import SnapshotWriter
        self._snapshot_writer = (
            SnapshotWriter(config.db_path, config.rng_seed)
            if config.snapshot_interval > 0
            else None
        )

        # Climate-shift counter (ticks remaining with reduced food regen)
        self._climate_penalty_ticks: int = 0

        # Epidemic log for visualization: list of (tick, beta)
        self._epidemic_log: list[tuple[int, float]] = []

        # Attack events for visualization: list of (src_x, src_y, tgt_x, tgt_y, civ_id)
        self._attack_events: list[tuple[int, int, int, int, int]] = []

        # History for visualization
        self.history: dict[str, list] = {"tick": []}
        for i in range(config.num_civs):
            self.history[f"pop_{i}"] = []
            self.history[f"mil_{i}"] = []
            self.history[f"cities_{i}"] = []
            self.history[f"food_civ_{i}"] = []
        self.history["food_total"] = []
        self.history["minerals_total"] = []
        self.history["wood_total"] = []

    # ------------------------------------------------------------------

    def step(self) -> None:
        self._attack_events = []

        # Compute climate penalty for this tick's regen BEFORE grid.step()
        food_regen_mult = 0.85 if self._climate_penalty_ticks > 0 else 1.0
        if self._climate_penalty_ticks > 0:
            self._climate_penalty_ticks -= 1

        # 1. Resource regeneration (with optional climate dampening)
        self.grid.step(food_regen_mult=food_regen_mult)

        # 2. Border pressure — contested tiles revert to unclaimed
        self._apply_border_reversion()

        # 3. Environmental events
        events = self.event_sampler.sample(self.grid)
        if any(e.name == "climate_shift" for e in events):
            self._climate_penalty_ticks = 20

        disease_events = [e for e in events if e.name == "disease"]
        if disease_events:
            self._apply_disease(beta=disease_events[0].transmission_rate)

        # 4. Dispatch all decisions (LLM async batch, rule-based sync)
        asyncio.run(self._dispatch_decisions())

        # 5. Activate all city agents in random order (reads _pending_action)
        self.agents.shuffle_do("step")

        # 6. Update civilization aggregates
        for civ in self.civilizations:
            cities = [a for a in self.agents if isinstance(a, CityAgent) and a.civ is civ]
            civ.update_aggregates(cities)

        # 7. Log each city's state
        env_tag = ",".join(e.name for e in events) if events else ""
        for agent in list(self.agents):
            if isinstance(agent, CityAgent):
                self.logger.log_event(
                    tick=self.steps,
                    agent_id=str(agent.unique_id),
                    civ_id=agent.civ.civ_id,
                    action=agent.last_action,
                    pop=agent.population,
                    military=agent.military,
                    tech_level=agent.civ.tech_level,
                    territory=self.grid.territory_count(agent.civ.civ_id),
                    env_event=env_tag,
                )

        # 8. Record history snapshot
        food_layer = self.grid.layers[ResourceType.FOOD].data
        min_layer  = self.grid.layers[ResourceType.MINERALS].data
        wood_layer = self.grid.layers[ResourceType.WOOD].data
        ownership  = self.grid.ownership
        self.history["tick"].append(self.steps)
        self.history["food_total"].append(float(food_layer.sum()))
        self.history["minerals_total"].append(float(min_layer.sum()))
        self.history["wood_total"].append(float(wood_layer.sum()))
        for i, civ in enumerate(self.civilizations):
            mask = ownership == i
            self.history[f"pop_{i}"].append(civ.total_pop)
            self.history[f"mil_{i}"].append(civ.total_military)
            self.history[f"food_civ_{i}"].append(float(food_layer[mask].sum()))
            self.history[f"cities_{i}"].append(
                sum(1 for a in self.agents if isinstance(a, CityAgent) and a.civ is civ)
            )

        # 9. Write snapshot if interval is configured
        if self._snapshot_writer and self.steps % self.config.snapshot_interval == 0:
            self._snapshot_writer.write(
                self.steps, self.grid, list(self.agents), self.civilizations
            )

        # 10. Stop if only one civ remains or max ticks reached
        alive = [c for c in self.civilizations if c.alive]
        if len(alive) <= 1 or self.steps >= self.config.max_ticks:
            self.running = False
            self.logger.close()
            if self._snapshot_writer:
                self._snapshot_writer.close()
                self._snapshot_writer = None

    # ------------------------------------------------------------------

    async def _dispatch_decisions(self) -> None:
        """Batch all city decisions by provider, run LLM providers concurrently."""
        from collections import defaultdict
        from civ_sim.agents.city import CityAgent

        by_provider: dict = defaultdict(list)
        for agent in self.agents:
            if isinstance(agent, CityAgent):
                by_provider[agent.civ.provider].append(agent)

        async def _one_batch(provider, cities):
            actions = await provider.choose_actions_batch(cities)
            for city, action in zip(cities, actions):
                city._pending_action = action

        await asyncio.gather(*[
            _one_batch(provider, cities)
            for provider, cities in by_provider.items()
        ], return_exceptions=True)

    # ------------------------------------------------------------------

    def _apply_disease(self, beta: float = 1.0) -> None:
        """Apply epidemic with transmission rate β.

        Mortality = β × base_rate, amplified by nearby same-civ cities (urban density
        accelerates spread). β~0.1 is a mild flu; β~3.0 is a catastrophic plague.
        """
        cities = [a for a in self.agents if isinstance(a, CityAgent)]
        base_rate = self.config.pop_starvation_rate  # ~4% base

        total_deaths = 0
        for city in cities:
            # Count same-civ cities within 20 tiles — dense networks spread disease faster
            nearby = sum(
                1 for other in cities
                if other is not city
                and other.civ.civ_id == city.civ.civ_id
                and abs(other.x - city.x) + abs(other.y - city.y) <= 20
            )
            proximity_factor = 1.0 + nearby * 0.25
            mortality = min(0.85, beta * base_rate * proximity_factor)
            hit = math.ceil(city.population * mortality)
            total_deaths += hit
            city.population = max(1, city.population - hit)
            city._disease_hit_ticks = 8  # show disease overlay for 8 ticks
        self._epidemic_log.append((self.steps, beta, total_deaths))

    def _apply_border_reversion(self) -> None:
        """Tiles at the border of two civs revert to unclaimed with border_reversion_prob."""
        ownership = self.grid.ownership
        prob = self.config.border_reversion_prob
        # Snapshot ownership so all border detection uses the same pre-tick state
        snap = ownership.copy()
        revert_mask = np.zeros(ownership.shape, dtype=bool)
        rolls = self._np_rng.random(ownership.shape) < prob
        for civ in self.civilizations:
            owned = (snap == civ.civ_id)
            enemy_adj = np.zeros(ownership.shape, dtype=bool)
            for other in self.civilizations:
                if other.civ_id == civ.civ_id:
                    continue
                e = (snap == other.civ_id)
                # Orthogonal neighbor detection without edge wrapping
                e_adj = np.zeros(ownership.shape, dtype=bool)
                e_adj[1:, :]  |= e[:-1, :]   # neighbor above
                e_adj[:-1, :] |= e[1:, :]    # neighbor below
                e_adj[:, 1:]  |= e[:, :-1]   # neighbor left
                e_adj[:, :-1] |= e[:, 1:]    # neighbor right
                enemy_adj |= e_adj
            border = owned & enemy_adj
            revert_mask |= border & rolls
        # Protect city home tiles from reversion
        for agent in self.agents:
            if isinstance(agent, CityAgent):
                revert_mask[agent.x, agent.y] = False
        ownership[revert_mask] = -1

    def _find_settle_location(self, civ: "Civilization") -> tuple[int, int] | None:
        """Return an unclaimed frontier tile at least 6 tiles from every same-civ city."""
        from civ_sim.agents.city import CityAgent
        own_positions = [(a.x, a.y) for a in self.agents if isinstance(a, CityAgent) and a.civ is civ]
        ownership = np.asarray(self.grid.ownership)
        owned = ownership == civ.civ_id
        unclaimed = ownership == -1
        # Frontier: unclaimed tiles adjacent to owned territory
        adj = np.zeros_like(owned)
        adj[1:, :] |= owned[:-1, :]
        adj[:-1, :] |= owned[1:, :]
        adj[:, 1:] |= owned[:, :-1]
        adj[:, :-1] |= owned[:, 1:]
        frontier_xs, frontier_ys = np.where(unclaimed & adj)
        if len(frontier_xs) == 0:
            # Fall back to any unclaimed tile
            frontier_xs, frontier_ys = np.where(unclaimed)
        if len(frontier_xs) == 0:
            return None
        n = min(200, len(frontier_xs))
        indices = self.random.sample(range(len(frontier_xs)), n)
        for i in indices:
            tx, ty = int(frontier_xs[i]), int(frontier_ys[i])
            if not own_positions or min(abs(cx - tx) + abs(cy - ty) for cx, cy in own_positions) >= 6:
                return tx, ty
        return None

    def _create_civs(self) -> list[Civilization]:
        cfg = self.config
        if not cfg.civ_providers:
            from civ_sim.config import ProviderConfig
            cfg.civ_providers = [ProviderConfig()]
        rng = self.random
        civs = []
        names = ["Alpha", "Beta", "Gamma", "Delta"]
        for i in range(cfg.num_civs):
            lo, hi = cfg.trait_range
            traits = CulturalTraits(
                aggressiveness=rng.uniform(lo, hi),
                trust=rng.uniform(lo, hi),
                innovation=rng.uniform(lo, hi),
                tribalism=rng.uniform(lo, hi),
                risk_tolerance=rng.uniform(lo, hi),
            )
            from civ_sim.agents.providers.factory import create_provider
            provider_cfg = (
                cfg.civ_providers[i]
                if i < len(cfg.civ_providers)
                else cfg.civ_providers[0]
            )
            provider = create_provider(provider_cfg)
            civs.append(Civilization(civ_id=i, name=names[i], traits=traits, provider=provider))
        return civs

    def _place_cities(self) -> None:
        cfg = self.config
        rng = self.random
        # Split the grid into vertical halves, one per civ
        half = cfg.width // 2
        regions = [
            (0, half - 5),
            (half + 5, cfg.width - 1),
        ]
        for i, civ in enumerate(self.civilizations):
            x_min, x_max = regions[i % len(regions)]
            for _ in range(cfg.cities_per_civ):
                for _attempt in range(50):
                    x = rng.randint(x_min, x_max)
                    y = rng.randint(5, cfg.height - 5)
                    # Avoid placing two cities too close together
                    from civ_sim.agents.city import CityAgent as _CA
                    too_close = any(
                        abs(a.x - x) + abs(a.y - y) < 8
                        for a in self.agents
                        if isinstance(a, _CA)
                    )
                    if not too_close:
                        CityAgent(self, civ, x, y)
                        break

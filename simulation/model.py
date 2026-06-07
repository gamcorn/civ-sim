from __future__ import annotations
import asyncio
import math
import random as stdlib_random

import mesa
import numpy as np

from config import SimConfig
from world.grid import ResourceGrid
from world.events import EventSampler
from agents.civilization import Civilization, CulturalTraits
from agents.city import CityAgent
from technology.discovery import TechEngine
from storage.logger import EventLogger


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

        from storage.snapshot import SnapshotWriter
        self._snapshot_writer = (
            SnapshotWriter(config.db_path, config.rng_seed)
            if config.snapshot_interval > 0
            else None
        )

        # Climate-shift counter (ticks remaining with reduced food regen)
        self._climate_penalty_ticks: int = 0

        # History for visualization
        self.history: dict[str, list] = {
            "tick": [],
            "pop_0": [],
            "pop_1": [],
            "mil_0": [],
            "mil_1": [],
        }

    # ------------------------------------------------------------------

    def step(self) -> None:
        # 1. Resource regeneration
        self.grid.step()

        # 2. Border pressure — contested tiles revert to unclaimed
        self._apply_border_reversion()

        # 3. Environmental events
        events = self.event_sampler.sample(self.grid)
        if any(e.name == "climate_shift" for e in events):
            self._climate_penalty_ticks = 20

        if self._climate_penalty_ticks > 0:
            # Temporarily suppress food regen via a one-tick multiplier
            self.grid.layers[
                __import__("world.resources", fromlist=["ResourceType"]).ResourceType.FOOD
            ].data *= 0.85
            self._climate_penalty_ticks -= 1

        if any(e.name == "disease" for e in events):
            self._apply_disease()

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
        self.history["tick"].append(self.steps)
        for i, civ in enumerate(self.civilizations):
            self.history[f"pop_{i}"].append(civ.total_pop)
            self.history[f"mil_{i}"].append(civ.total_military)

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
        from agents.city import CityAgent

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

    def _apply_disease(self) -> None:
        """Apply epidemic: each city loses ~20% population (5× starvation rate)."""
        for agent in list(self.agents):
            if isinstance(agent, CityAgent):
                hit = math.ceil(agent.population * self.config.pop_starvation_rate * 5)
                agent.population = max(1, agent.population - hit)

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
        """Return an owned tile at least 10 tiles from every existing city, or None."""
        from agents.city import CityAgent
        city_positions = [(a.x, a.y) for a in self.agents if isinstance(a, CityAgent)]
        ownership = np.asarray(self.grid.ownership)
        xs, ys = np.where(ownership == civ.civ_id)
        if len(xs) == 0:
            return None
        n = min(100, len(xs))
        indices = self.random.sample(range(len(xs)), n)
        for i in indices:
            tx, ty = int(xs[i]), int(ys[i])
            if not city_positions or min(abs(cx - tx) + abs(cy - ty) for cx, cy in city_positions) >= 10:
                return tx, ty
        return None

    def _create_civs(self) -> list[Civilization]:
        cfg = self.config
        if not cfg.civ_providers:
            from config import ProviderConfig
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
            from agents.providers.factory import create_provider
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
                    from agents.city import CityAgent as _CA
                    too_close = any(
                        abs(a.x - x) + abs(a.y - y) < 8
                        for a in self.agents
                        if isinstance(a, _CA)
                    )
                    if not too_close:
                        CityAgent(self, civ, x, y)
                        break

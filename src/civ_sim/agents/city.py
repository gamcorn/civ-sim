from __future__ import annotations
import math
from typing import TYPE_CHECKING

from mesa.discrete_space import Grid2DMovingAgent

from civ_sim.agents.decisions import (
    choose_action, GATHER, TRADE, EXPAND, FORTIFY, ATTACK, RESEARCH,
    _attack_target, _has_unclaimed_neighbor,
)
from civ_sim.world.resources import ResourceType

if TYPE_CHECKING:
    from civ_sim.agents.civilization import Civilization
    from civ_sim.simulation.model import CivModel


class CityAgent(Grid2DMovingAgent):
    """One city — the primary simulation unit (Level 2 granularity)."""

    def __init__(self, model: "CivModel", civ: "Civilization", x: int, y: int):
        super().__init__(model)
        self.civ = civ
        self.x = x
        self.y = y
        self.cell = model.grid.cell(x, y)

        self.population: int = model.config.initial_pop
        self.military: int = model.config.initial_military
        self.food_stock: float = 30.0
        self.wood_stock: float = model.config.initial_wood_stock
        self.mineral_stock: float = model.config.initial_mineral_stock
        self.last_action: str = "spawn"
        self.age: int = 0
        self._pending_action: str | None = None
        self._settle_cooldown: int = 0  # ticks until this city can settle again
        self._disease_hit_ticks: int = 0  # ticks remaining to show disease overlay

        # Claim starting tile
        model.grid.claim(x, y, civ.civ_id)

    # ------------------------------------------------------------------

    def step(self) -> None:
        self.age += 1
        if self._disease_hit_ticks > 0:
            self._disease_hit_ticks -= 1
        self._consume_resources()
        if self.population <= 0:
            self._collapse()
            return
        action = self._pending_action if self._pending_action is not None else choose_action(self)
        self._pending_action = None
        self._execute(action)
        self.last_action = action
        self._grow_population()
        self._maybe_settle()

    # ------------------------------------------------------------------

    def _consume_resources(self) -> None:
        cfg = self.model.config
        # Passive harvest from the city tile (3 < regen 4/tick so tile stays positive)
        self.food_stock += self.model.grid.consume(self.x, self.y, ResourceType.FOOD, 3.0)
        self.wood_stock += self.model.grid.consume(self.x, self.y, ResourceType.WOOD, 1.5)
        self.mineral_stock += self.model.grid.consume(self.x, self.y, ResourceType.MINERALS, 1.0)

        # Consumption from stockpile
        needed = self.population * cfg.food_per_person + self.military * cfg.military_upkeep
        self.food_stock = max(0.0, self.food_stock - needed)

        # Starvation when stockpile is empty
        if self.food_stock == 0.0 and needed > 0:
            loss = int(math.ceil(self.population * cfg.pop_starvation_rate))
            self.population = max(0, self.population - loss)

        # Wood upkeep
        wood_needed = self.population * cfg.wood_per_person + self.military * cfg.wood_per_military
        self.wood_stock = max(0.0, self.wood_stock - wood_needed)
        if self.wood_stock == 0.0 and wood_needed > 0:
            loss = int(math.ceil(self.population * cfg.wood_shortage_rate))
            self.population = max(0, self.population - loss)

        # Mineral upkeep
        mineral_needed = self.population * cfg.mineral_per_person + self.military * cfg.mineral_per_military
        self.mineral_stock = max(0.0, self.mineral_stock - mineral_needed)
        if self.mineral_stock == 0.0 and mineral_needed > 0:
            decay_f = self.military * cfg.mineral_shortage_rate
            decay = int(decay_f)
            if self.model.random.random() < (decay_f - decay):
                decay += 1
            self.military = max(0, self.military - decay)

        # Military attrition — stochastic rounding so small forces still decay
        decay_f = self.military * 0.02
        decay = int(decay_f)
        if self.model.random.random() < (decay_f - decay):
            decay += 1
        self.military = max(0, self.military - decay)

    def _grow_population(self) -> None:
        cfg = self.model.config
        if self.population >= cfg.pop_cap:
            return

        # Food security: how many ticks of food the stockpile covers (0–1 scale)
        daily_need = max(self.population * cfg.food_per_person, 1e-6)
        security = min(1.0, self.food_stock / (daily_need * 10))  # 10-tick horizon
        if security <= 0.0:
            return

        # Geography bonus: tile food quality adds up to 0.3 to the security ratio
        food_here = self.model.grid.get(self.x, self.y, ResourceType.FOOD)
        geo_bonus = min(0.3, food_here / cfg.resource_max * 0.3)

        food_ratio = min(1.0, security + geo_bonus)
        civ_pop = max(1, self.civ.total_pop)
        demo_factor = max(0.1, 1.0 - civ_pop / cfg.pop_demographic_cap)
        tech_bonus = len(self.civ.discovered_techs) * 0.001
        rate = cfg.pop_growth_rate_max * food_ratio * demo_factor + tech_bonus
        growth_f = self.population * rate
        growth = int(growth_f)
        if self.model.random.random() < (growth_f - growth):
            growth += 1
        self.population = min(self.population + growth, cfg.pop_cap)

    def _execute(self, action: str) -> None:
        if action == GATHER:
            self._do_gather()
        elif action == TRADE:
            self._do_trade()
        elif action == EXPAND:
            self._do_expand()
        elif action == FORTIFY:
            self._do_fortify()
        elif action == ATTACK:
            self._do_attack()
        elif action == RESEARCH:
            self._do_research()

    # ------------------------------------------------------------------

    def _do_gather(self) -> None:
        """Actively harvest food from all claimed tiles within harvest_radius."""
        grid = self.model.grid
        cfg = self.model.config
        r = cfg.harvest_radius
        bonus = self.civ.harvest_bonus
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    if grid.ownership[nx, ny] == self.civ.civ_id:
                        self.food_stock += grid.consume(nx, ny, ResourceType.FOOD, 3.0) * bonus
                        self.food_stock += grid.consume(nx, ny, ResourceType.WATER, 1.0) * 0.5 * bonus
                        self.wood_stock += grid.consume(nx, ny, ResourceType.WOOD, 1.0) * bonus
                        self.mineral_stock += grid.consume(nx, ny, ResourceType.MINERALS, 0.5) * bonus

    def _do_trade(self) -> None:
        """Transfer surplus food to nearest city (any civ) and receive minerals/wood back."""
        grid = self.model.grid
        surplus = grid.get(self.x, self.y, ResourceType.FOOD) * 0.15
        if surplus < 1:
            return
        # Find closest city
        best = None
        best_dist = 999
        for other in self.model.agents_by_type.get(type(self), []):
            dist = abs(other.x - self.x) + abs(other.y - self.y)
            if other is not self and dist < best_dist:
                best_dist = dist
                best = other
        if best and best_dist <= 30:
            grid.consume(self.x, self.y, ResourceType.FOOD, surplus)
            grid.deposit(best.x, best.y, ResourceType.FOOD, surplus * 0.7)
            # Receive minerals in return
            received = grid.consume(best.x, best.y, ResourceType.MINERALS, surplus * 0.3)
            grid.deposit(self.x, self.y, ResourceType.MINERALS, received)

    def _do_expand(self) -> None:
        """Claim an adjacent unclaimed tile with the best resource value."""
        grid = self.model.grid
        best_tile = None
        best_val = -1.0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    if grid.ownership[nx, ny] == -1:
                        val = (grid.get(nx, ny, ResourceType.FOOD) +
                               grid.get(nx, ny, ResourceType.MINERALS) * 2)
                        if val > best_val:
                            best_val = val
                            best_tile = (nx, ny)
        if best_tile:
            self.wood_stock = max(0.0, self.wood_stock - self.model.config.expand_wood_cost)
            grid.claim(*best_tile, self.civ.civ_id)

    def _do_fortify(self) -> None:
        cfg = self.model.config
        consumed_m = min(self.mineral_stock, cfg.fortify_mineral_cost)
        consumed_w = min(self.wood_stock, cfg.fortify_wood_cost)
        self.mineral_stock -= consumed_m
        self.wood_stock -= consumed_w
        self.military += int((consumed_m + consumed_w) / 2)

    def _do_attack(self) -> None:
        cfg = self.model.config
        self.mineral_stock = max(0.0, self.mineral_stock - cfg.attack_mineral_cost)

        target = _attack_target(self)
        if target is None:
            return

        fort_factor = min(1.0, target.military / cfg.max_defense_military)
        damage_factor = 1.0 - fort_factor * cfg.fortify_defense_bonus

        self.model._attack_events.append(
            (self.x, self.y, target.x, target.y, self.civ.civ_id)
        )
        my_str = self.military * (1 + len(self.civ.discovered_techs) * cfg.tech_military_bonus)
        enemy_str = target.military * (1 + len(target.civ.discovered_techs) * cfg.tech_military_bonus)
        win_prob = my_str / (my_str + enemy_str + 1e-6)

        if self.model.random.random() < win_prob:
            # Attacker takes proportional losses from enemy resistance (Lanchester)
            attacker_loss = max(1, int(self.military * (enemy_str / (my_str + enemy_str + 1e-6)) * 0.3))
            self.military = max(0, self.military - attacker_loss)
            target.military = max(0, target.military - int(self.military * 0.3))
            target.population = max(0, target.population - int(target.population * 0.2))

            raid_f = cfg.battle_pillage_rate * damage_factor
            target.food_stock    = max(0.0, target.food_stock    * (1.0 - raid_f))
            target.wood_stock    = max(0.0, target.wood_stock    * (1.0 - raid_f))
            target.mineral_stock = max(0.0, target.mineral_stock * (1.0 - raid_f))

            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = target.x + dx, target.y + dy
                    if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height:
                        if self.model.grid.ownership[nx, ny] == target.civ.civ_id:
                            self.model.grid.claim(nx, ny, self.civ.civ_id)

            if target.population < cfg.initial_pop * cfg.capture_threshold:
                self._capture_city(target, damage_factor=damage_factor)
        else:
            self.military = max(0, self.military - int(self.military * 0.25))

    def _capture_city(self, target: "CityAgent", damage_factor: float = 1.0) -> None:
        """Transfer a weakened enemy city to this civilization."""
        cfg = self.model.config

        self.food_stock    += target.food_stock
        self.wood_stock    += target.wood_stock
        self.mineral_stock += target.mineral_stock
        target.food_stock    = 0.0
        target.wood_stock    = 0.0
        target.mineral_stock = 0.0

        self.wood_stock    = max(0.0, self.wood_stock    - cfg.capture_reconstruct_wood    * damage_factor)
        self.mineral_stock = max(0.0, self.mineral_stock - cfg.capture_reconstruct_mineral * damage_factor)

        target.civ = self.civ
        self.model.grid.claim(target.x, target.y, self.civ.civ_id)
        self.model.logger.log_event(
            tick=self.model.steps,
            agent_id=str(target.unique_id),
            civ_id=self.civ.civ_id,
            action="capture",
            pop=target.population,
            military=target.military,
            tech_level=self.civ.tech_level,
            territory=self.model.grid.territory_count(self.civ.civ_id),
            env_event="",
        )

    def _maybe_settle(self) -> None:
        """Found a daughter city when population hits the cap and cooldown has elapsed."""
        cfg = self.model.config
        if self._settle_cooldown > 0:
            self._settle_cooldown -= 1
            return
        if self.population < cfg.pop_cap:
            return
        civ_city_count = sum(
            1 for a in self.model.agents
            if isinstance(a, CityAgent) and a.civ is self.civ
        )
        if civ_city_count >= cfg.max_cities_per_civ:
            return
        pos = self.model._find_settle_location(self.civ)
        if pos is None:
            return
        settler_pop = cfg.initial_pop // 2
        self.population -= settler_pop
        self.wood_stock    = max(0.0, self.wood_stock    - cfg.settle_wood_cost)
        self.mineral_stock = max(0.0, self.mineral_stock - cfg.settle_mineral_cost)
        self._settle_cooldown = cfg.settle_cooldown
        new_city = CityAgent(self.model, self.civ, pos[0], pos[1])
        new_city.population = settler_pop
        new_city.food_stock = 20.0
        new_city.wood_stock    = cfg.initial_wood_stock * 0.5
        new_city.mineral_stock = cfg.initial_mineral_stock * 0.5
        # Drain food from surrounding tiles — land clearing stresses local resources
        r = 3
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height:
                    current = self.model.grid.get(nx, ny, ResourceType.FOOD)
                    self.model.grid.consume(nx, ny, ResourceType.FOOD, current * cfg.settle_land_drain)
        self.model.logger.log_event(
            tick=self.model.steps,
            agent_id=str(new_city.unique_id),
            civ_id=self.civ.civ_id,
            action="settle",
            pop=new_city.population,
            military=0,
            tech_level=self.civ.tech_level,
            territory=self.model.grid.territory_count(self.civ.civ_id),
            env_event="",
        )

    def _do_research(self) -> None:
        from civ_sim.technology.discovery import TechEngine
        cfg = self.model.config
        self.wood_stock    = max(0.0, self.wood_stock    - cfg.research_wood_cost)
        self.mineral_stock = max(0.0, self.mineral_stock - cfg.research_mineral_cost)
        self.model.tech_engine.check(self)

    def _collapse(self) -> None:
        self.model.grid.ownership[self.x, self.y] = -1  # release home tile only
        self.model.logger.log_event(
            tick=self.model.steps,
            agent_id=str(self.unique_id),
            civ_id=self.civ.civ_id,
            action="collapse",
            pop=0,
            military=0,
            tech_level=self.civ.tech_level,
            territory=0,
            env_event="",
        )
        self.remove()


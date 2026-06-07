from __future__ import annotations
import math
from typing import TYPE_CHECKING

from mesa.discrete_space import Grid2DMovingAgent

from agents.decisions import (
    choose_action, GATHER, TRADE, EXPAND, FORTIFY, ATTACK, RESEARCH,
    _attack_target, _has_unclaimed_neighbor,
)
from world.resources import ResourceType

if TYPE_CHECKING:
    from agents.civilization import Civilization
    from simulation.model import CivModel


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
        self.last_action: str = "spawn"
        self.age: int = 0
        self._pending_action: str | None = None

        # Claim starting tile
        model.grid.claim(x, y, civ.civ_id)

    # ------------------------------------------------------------------

    def step(self) -> None:
        self.age += 1
        self._consume_resources()
        if self.population <= 0:
            self._collapse()
            return
        action = self._pending_action if self._pending_action is not None else choose_action(self)
        self._pending_action = None
        self._execute(action)
        self.last_action = action
        self._grow_population()

    # ------------------------------------------------------------------

    def _consume_resources(self) -> None:
        cfg = self.model.config
        # Passive harvest from the city tile (3 < regen 4/tick so tile stays positive)
        self.food_stock += self.model.grid.consume(self.x, self.y, ResourceType.FOOD, 3.0)

        # Consumption from stockpile
        needed = self.population * cfg.food_per_person + self.military * cfg.military_upkeep
        self.food_stock = max(0.0, self.food_stock - needed)

        # Starvation when stockpile is empty
        if self.food_stock == 0.0 and needed > 0:
            loss = int(math.ceil(self.population * cfg.pop_starvation_rate))
            self.population = max(0, self.population - loss)

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
        food_here = self.model.grid.get(self.x, self.y, ResourceType.FOOD)
        if food_here > cfg.resource_max * 0.2:
            tech_bonus = len(self.civ.discovered_techs) * 0.005
            growth_f = self.population * (cfg.pop_growth_rate + tech_bonus)
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
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < grid.width and 0 <= ny < grid.height:
                    if grid.ownership[nx, ny] == self.civ.civ_id:
                        self.food_stock += grid.consume(nx, ny, ResourceType.FOOD, 3.0)
                        self.food_stock += grid.consume(nx, ny, ResourceType.WATER, 1.0) * 0.5

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
            grid.claim(*best_tile, self.civ.civ_id)

    def _do_fortify(self) -> None:
        # Consume minerals and wood to build military
        consumed_m = self.model.grid.consume(self.x, self.y, ResourceType.MINERALS, 5.0)
        consumed_w = self.model.grid.consume(self.x, self.y, ResourceType.WOOD, 3.0)
        built = int((consumed_m + consumed_w) / 2)
        self.military += built

    def _do_attack(self) -> None:
        target = _attack_target(self)
        if target is None:
            return
        my_str = self.military * (1 + len(self.civ.discovered_techs) * self.model.config.tech_military_bonus)
        enemy_str = target.military * (1 + len(target.civ.discovered_techs) * self.model.config.tech_military_bonus)
        win_prob = my_str / (my_str + enemy_str + 1e-6)
        if self.model.random.random() < win_prob:
            # Victory: capture target's territory
            target.military = max(0, target.military - int(self.military * 0.3))
            target.population = max(0, target.population - int(target.population * 0.2))
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = target.x + dx, target.y + dy
                    if 0 <= nx < self.model.grid.width and 0 <= ny < self.model.grid.height:
                        if self.model.grid.ownership[nx, ny] == target.civ.civ_id:
                            self.model.grid.claim(nx, ny, self.civ.civ_id)
        else:
            # Defeat: attacker takes losses
            self.military = max(0, self.military - int(self.military * 0.25))

    def _do_research(self) -> None:
        from technology.discovery import TechEngine
        self.model.tech_engine.check(self)

    def _collapse(self) -> None:
        self.model.grid.ownership[
            self.model.grid.ownership == self.civ.civ_id
        ] = -1  # surrender territory
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


from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent

from civ_sim.world.resources import ResourceType

# Each entry: tech_name → {prerequisite_tech_or_resource: threshold}
# String keys prefixed with "tech:" mean a discovered tech is required.
# Other keys are ResourceType enum values.
TECH_TREE: dict[str, dict] = {
    "agriculture":   {ResourceType.FOOD: 30.0},
    "irrigation":    {"tech:agriculture": True, ResourceType.WATER: 25.0},
    "metallurgy":    {ResourceType.MINERALS: 40.0, ResourceType.WOOD: 20.0},
    "masonry":       {ResourceType.MINERALS: 30.0},
    "bronze_tools":  {"tech:metallurgy": True, ResourceType.MINERALS: 60.0},
    "iron_working":  {"tech:bronze_tools": True, ResourceType.MINERALS: 80.0},
    "sailing":       {ResourceType.WOOD: 60.0, ResourceType.WATER: 40.0},
    "writing":       {"tech:agriculture": True, "tech:masonry": True},
    "mathematics":   {"tech:writing": True},
    "steam_power":   {"tech:iron_working": True, ResourceType.WOOD: 80.0},
    "industrialism": {"tech:steam_power": True, ResourceType.MINERALS: 90.0},
}

# Multipliers applied to a city when a tech is first discovered
TECH_EFFECTS: dict[str, dict[str, float]] = {
    "agriculture":   {"food_regen": 0.1},
    "irrigation":    {"food_regen": 0.15},
    "metallurgy":    {"military_bonus": 0.15},
    "bronze_tools":  {"military_bonus": 0.2, "food_regen": 0.05},
    "iron_working":  {"military_bonus": 0.3},
    "sailing":       {"trade_range": 10},
    "steam_power":   {"food_regen": 0.2, "military_bonus": 0.25},
    "industrialism": {"food_regen": 0.3, "military_bonus": 0.4},
}


class TechEngine:
    def check(self, city: "CityAgent") -> None:
        """Check if the city's civilization can discover any new technologies."""
        civ = city.civ
        grid = city.model.grid

        for tech, reqs in TECH_TREE.items():
            if tech in civ.discovered_techs:
                continue
            if self._requirements_met(tech, reqs, city, civ, grid):
                self._discover(tech, city)

    def _requirements_met(self, tech, reqs, city, civ, grid) -> bool:
        for req, threshold in reqs.items():
            if isinstance(req, str) and req.startswith("tech:"):
                needed = req[5:]
                if needed not in civ.discovered_techs:
                    return False
            elif isinstance(req, ResourceType):
                if grid.get(city.x, city.y, req) < threshold:
                    return False
        return True

    def _discover(self, tech: str, city: "CityAgent") -> None:
        civ = city.civ
        civ.discovered_techs.add(tech)
        civ.tech_level = len(civ.discovered_techs)

        effects = TECH_EFFECTS.get(tech, {})
        if "food_regen" in effects:
            civ.harvest_bonus += effects["food_regen"]   # per-civ, not global config

        city.model.logger.log_event(
            tick=city.model.steps,
            agent_id=str(city.unique_id),
            civ_id=civ.civ_id,
            action=f"discover:{tech}",
            pop=city.population,
            military=city.military,
            tech_level=civ.tech_level,
            territory=city.model.grid.territory_count(civ.civ_id),
            env_event="",
        )

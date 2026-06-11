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
    "mining":        {ResourceType.MINERALS: 35.0},
    "forestry":      {ResourceType.WOOD: 35.0},
    "bronze_tools":  {"tech:metallurgy": True, ResourceType.MINERALS: 60.0},
    "iron_working":  {"tech:bronze_tools": True, ResourceType.MINERALS: 80.0},
    "sailing":       {ResourceType.WOOD: 60.0, ResourceType.WATER: 40.0},
    "writing":       {"tech:agriculture": True, "tech:masonry": True},
    "mathematics":   {"tech:writing": True},
    "steam_power":   {"tech:iron_working": True, ResourceType.WOOD: 80.0},
    "industrialism": {"tech:steam_power": True, ResourceType.MINERALS: 90.0},
}

# Science-point cost per technology (accumulated via _do_research)
TECH_COSTS: dict[str, float] = {
    "agriculture":    50.0,
    "masonry":        40.0,
    "metallurgy":     60.0,
    "mining":         60.0,
    "forestry":       60.0,
    "sailing":        80.0,
    "irrigation":     80.0,
    "bronze_tools":  100.0,
    "writing":       120.0,
    "iron_working":  150.0,
    "mathematics":   180.0,
    "steam_power":   250.0,
    "industrialism": 400.0,
}

# Multipliers applied to a city when a tech is first discovered
TECH_EFFECTS: dict[str, dict] = {
    "agriculture":   {"food_regen": 0.1, "land_productivity": 0.3,  "unlock": "cultivate"},
    "irrigation":    {"food_regen": 0.05, "land_productivity": 0.25},
    "metallurgy":    {"military_bonus": 0.15},
    "mining":        {"mining_efficiency": 0.3,  "unlock": "mine"},
    "forestry":      {"forestry_efficiency": 0.3, "unlock": "woodcut"},
    "bronze_tools":  {"military_bonus": 0.2, "land_productivity": 0.1, "mining_efficiency": 0.15},
    "iron_working":  {"military_bonus": 0.3, "mining_efficiency": 0.25},
    "sailing":       {"trade_range": 10},
    "steam_power":   {"food_regen": 0.1, "land_productivity": 0.2, "mining_efficiency": 0.2,
                      "forestry_efficiency": 0.2, "military_bonus": 0.25},
    "industrialism": {"food_regen": 0.1, "land_productivity": 0.3, "mining_efficiency": 0.3,
                      "forestry_efficiency": 0.3, "military_bonus": 0.4},
}


class TechEngine:
    def check(self, city: "CityAgent") -> None:
        """Check if the city's civilization can discover any new tech (at most one per call)."""
        civ = city.civ
        grid = city.model.grid

        for tech, reqs in TECH_TREE.items():
            if tech in civ.discovered_techs:
                continue
            if self._requirements_met(tech, reqs, city, civ, grid):
                self._discover(tech, city)
                return   # at most one discovery per research action

    def _requirements_met(self, tech, reqs, city, civ, grid) -> bool:
        # Check science-point cost
        point_cost = TECH_COSTS.get(tech, 100.0)
        if civ.science_points < point_cost:
            return False
        # Check prerequisites (tech and tile-resource thresholds)
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
        civ.science_points -= TECH_COSTS.get(tech, 100.0)

        effects = TECH_EFFECTS.get(tech, {})
        cfg = city.model.config
        if "food_regen" in effects:
            civ.harvest_bonus += effects["food_regen"]
        if "military_bonus" in effects:
            civ.military_tech_bonus += effects["military_bonus"]
        if "trade_range" in effects:
            civ.trade_range_bonus += effects["trade_range"]
        if "land_productivity" in effects:
            civ.land_productivity = min(
                cfg.land_productivity_max,
                civ.land_productivity + effects["land_productivity"],
            )
        if "mining_efficiency" in effects:
            civ.mining_efficiency = min(
                cfg.land_productivity_max,
                civ.mining_efficiency + effects["mining_efficiency"],
            )
        if "forestry_efficiency" in effects:
            civ.forestry_efficiency = min(
                cfg.land_productivity_max,
                civ.forestry_efficiency + effects["forestry_efficiency"],
            )
        if "unlock" in effects:
            civ.unlocked_actions.add(effects["unlock"])

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

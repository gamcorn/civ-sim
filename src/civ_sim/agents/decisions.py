from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent


# Action names
GATHER = "gather"
TRADE = "trade"
EXPAND = "expand"
FORTIFY = "fortify"
ATTACK = "attack"
RESEARCH = "research"

ALL_ACTIONS = [GATHER, TRADE, EXPAND, FORTIFY, ATTACK, RESEARCH]

# Base weights per action for each cultural trait
# Format: action → {trait → weight}
_WEIGHTS: dict[str, dict[str, float]] = {
    GATHER:   {"aggressiveness": -0.1, "trust": 0.1,  "innovation": 0.0,  "tribalism": 0.1,  "risk_tolerance": -0.1},
    TRADE:    {"aggressiveness": -0.4, "trust": 0.6,  "innovation": 0.1,  "tribalism": -0.2, "risk_tolerance": 0.0},
    EXPAND:   {"aggressiveness": 0.2,  "trust": 0.0,  "innovation": 0.1,  "tribalism": 0.5,  "risk_tolerance": 0.2},
    FORTIFY:  {"aggressiveness": 0.3,  "trust": -0.2, "innovation": 0.0,  "tribalism": 0.2,  "risk_tolerance": -0.3},
    ATTACK:   {"aggressiveness": 0.6,  "trust": -0.5, "innovation": 0.0,  "tribalism": 0.3,  "risk_tolerance": 0.4},
    RESEARCH: {"aggressiveness": -0.2, "trust": 0.1,  "innovation": 0.7,  "tribalism": 0.0,  "risk_tolerance": 0.1},
}


def get_action_scores(agent: "CityAgent") -> dict[str, float]:
    """Compute raw action scores (trait weights + resource modifiers) before feasibility filtering."""
    traits = agent.civ.traits
    t = traits.as_dict()
    return {
        action: sum(_WEIGHTS[action][k] * t[k] for k in t) + _resource_modifier(action, agent)
        for action in ALL_ACTIONS
    }


def choose_action(agent: "CityAgent") -> str:
    scores = get_action_scores(agent)
    feasible = _feasible(agent, scores)
    if not feasible:
        return GATHER
    return max(feasible, key=lambda a: feasible[a])


def _resource_modifier(action: str, agent: "CityAgent") -> float:
    from civ_sim.world.resources import ResourceType

    minerals = agent.model.grid.get(agent.x, agent.y, ResourceType.MINERALS)
    wood = agent.model.grid.get(agent.x, agent.y, ResourceType.WOOD)
    max_r = agent.model.config.resource_max

    # Use food_stock for food pressure (reflects city buffer, not just tile level)
    stock_ratio = min(1.0, agent.food_stock / (max_r * 2))
    min_ratio = minerals / max_r

    if action == GATHER:
        wood_ratio = min(1.0, agent.wood_stock / 50.0)
        min_ratio_s = min(1.0, agent.mineral_stock / 30.0)
        food_urgency = max(0.0, 0.4 - stock_ratio) * 1.5
        wood_urgency = max(0.0, 0.3 - wood_ratio) * 0.8
        min_urgency  = max(0.0, 0.3 - min_ratio_s) * 0.6
        return food_urgency + wood_urgency + min_urgency
    if action == TRADE:
        # Surplus to share when stock is high
        return (stock_ratio - 0.4) * 0.6
    if action == EXPAND:
        cfg = agent.model.config
        pop_pressure = (agent.population - 80) / 100.0
        land_hunger = max(0.0, 0.3 - stock_ratio) * 1.5
        wood_pen = 0.0 if agent.wood_stock >= cfg.expand_wood_cost else \
            -0.4 * (1.0 - agent.wood_stock / cfg.expand_wood_cost)
        return min(0.5, max(0.0, pop_pressure + land_hunger)) + wood_pen
    if action == FORTIFY:
        cfg = agent.model.config
        civ_mil = max(1, agent.civ.total_military)
        enemy_mil = _enemy_military(agent)
        ratio = enemy_mil / civ_mil
        if ratio > 1.0:
            base = min(0.7, (ratio - 1.0) * 0.5)
        elif agent.military < 15:
            base = 0.3 * (1.0 - agent.military / 15.0)
        else:
            base = 0.0
        pen = 0.0
        if agent.mineral_stock < cfg.fortify_mineral_cost:
            pen -= 0.3 * (1.0 - agent.mineral_stock / cfg.fortify_mineral_cost)
        if agent.wood_stock < cfg.fortify_wood_cost:
            pen -= 0.2 * (1.0 - agent.wood_stock / cfg.fortify_wood_cost)
        return base + pen
    if action == ATTACK:
        cfg = agent.model.config
        civ_mil = agent.civ.total_military
        enemy_mil = _enemy_military(agent)
        military_mod = 0.5 if civ_mil > enemy_mil * 0.8 else -0.5
        desperation = max(0.0, 0.1 - stock_ratio) * 6.0
        defense = _territorial_threat(agent) * 0.8
        ammo_pen = 0.0 if agent.mineral_stock >= cfg.attack_mineral_cost else -0.4
        return military_mod + desperation + defense + ammo_pen
    if action == RESEARCH:
        from civ_sim.technology.discovery import TECH_TREE
        if len(agent.civ.discovered_techs) >= len(TECH_TREE):
            return -1.0  # nothing left to discover
        cfg = agent.model.config
        wood_ready = min(1.0, agent.wood_stock / (cfg.research_wood_cost * 2))
        min_ready  = min(1.0, agent.mineral_stock / (cfg.research_mineral_cost * 2))
        return (wood_ready + min_ready) * 0.3 - 0.1

    return 0.0


def _enemy_military(agent: "CityAgent") -> int:
    for civ in agent.model.civilizations:
        if civ.civ_id != agent.civ.civ_id:
            return civ.total_military
    return 0


def _feasible(agent: "CityAgent", scores: dict[str, float]) -> dict[str, float]:
    feasible = {GATHER: scores[GATHER]}  # always available

    # Trade: needs at least one reachable enemy city
    if _has_trade_partner(agent):
        feasible[TRADE] = scores[TRADE]

    cfg = agent.model.config

    # Expand: needs unclaimed adjacent tile AND enough wood to build
    if _has_unclaimed_neighbor(agent) and agent.wood_stock >= cfg.expand_wood_cost * 0.5:
        feasible[EXPAND] = scores[EXPAND]

    # Fortify: always possible (partial fortify with whatever stocks are available)
    feasible[FORTIFY] = scores[FORTIFY]

    # Territorial defenders can attack with less military than normal aggressors
    mil_threshold = 2 if _territorial_threat(agent) > 0.1 else 5
    if (_attack_target(agent) is not None
            and agent.military >= mil_threshold
            and agent.mineral_stock >= cfg.attack_mineral_cost):
        feasible[ATTACK] = scores[ATTACK]

    # Research: needs stockpiled wood and minerals
    if (agent.wood_stock >= cfg.research_wood_cost
            and agent.mineral_stock >= cfg.research_mineral_cost):
        feasible[RESEARCH] = scores[RESEARCH]

    return feasible


def _has_trade_partner(agent: "CityAgent") -> bool:
    for other in agent.model.agents_by_type.get(type(agent), []):
        if other.civ.civ_id != agent.civ.civ_id:
            dist = abs(other.x - agent.x) + abs(other.y - agent.y)
            if dist <= 30:
                return True
    return False


def _has_unclaimed_neighbor(agent: "CityAgent") -> bool:
    grid = agent.model.grid
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                if grid.ownership[nx, ny] == -1:
                    return True
    return False


def _territorial_threat(agent: "CityAgent") -> float:
    """Fraction of tiles within 5 tiles of the city owned by enemies."""
    grid = agent.model.grid
    total = enemy = 0
    for dx in range(-5, 6):
        for dy in range(-5, 6):
            nx, ny = agent.x + dx, agent.y + dy
            if 0 <= nx < grid.width and 0 <= ny < grid.height:
                total += 1
                owner = int(grid.ownership[nx, ny])
                if owner >= 0 and owner != agent.civ.civ_id:
                    enemy += 1
    return enemy / max(1, total)


def _attack_target(agent: "CityAgent"):
    """Return the nearest enemy city within 25 tiles, or None."""
    best = None
    best_dist = 26
    for other in agent.model.agents_by_type.get(type(agent), []):
        if other.civ.civ_id != agent.civ.civ_id:
            dist = abs(other.x - agent.x) + abs(other.y - agent.y)
            if dist < best_dist:
                best_dist = dist
                best = other
    return best


def get_feasible_actions(agent: "CityAgent") -> list[str]:
    """Return the list of action names feasible for this city right now."""
    return list(_feasible(agent, get_action_scores(agent)).keys())

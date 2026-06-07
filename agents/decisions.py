from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.city import CityAgent


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


def choose_action(agent: "CityAgent") -> str:
    traits = agent.civ.traits
    t = traits.as_dict()
    scores: dict[str, float] = {}

    for action in ALL_ACTIONS:
        base = sum(_WEIGHTS[action][k] * t[k] for k in t)
        modifier = _resource_modifier(action, agent)
        scores[action] = base + modifier

    # Filter infeasible actions
    feasible = _feasible(agent, scores)
    if not feasible:
        return GATHER

    best = max(feasible, key=lambda a: feasible[a])
    return best


def _resource_modifier(action: str, agent: "CityAgent") -> float:
    from world.resources import ResourceType

    minerals = agent.model.grid.get(agent.x, agent.y, ResourceType.MINERALS)
    wood = agent.model.grid.get(agent.x, agent.y, ResourceType.WOOD)
    max_r = agent.model.config.resource_max

    # Use food_stock for food pressure (reflects city buffer, not just tile level)
    stock_ratio = min(1.0, agent.food_stock / (max_r * 2))
    min_ratio = minerals / max_r

    if action == GATHER:
        # Urgency when food stock is low
        return max(0.0, 0.4 - stock_ratio) * 1.5
    if action == TRADE:
        # Surplus to share when stock is high
        return (stock_ratio - 0.4) * 0.6
    if action == EXPAND:
        # Capped at 0.5 so expand doesn't permanently dominate all other actions
        pop_pressure = (agent.population - 80) / 100.0
        return min(0.5, max(0.0, pop_pressure))
    if action == FORTIFY:
        # More appealing the more outgunned we are
        civ_mil = max(1, agent.civ.total_military)
        enemy_mil = _enemy_military(agent)
        ratio = enemy_mil / civ_mil
        if ratio > 1.0:
            return min(0.7, (ratio - 1.0) * 0.5)
        # Baseline: cities want to maintain a minimum garrison
        if agent.military < 15:
            return 0.3 * (1.0 - agent.military / 15.0)
        return 0.0
    if action == ATTACK:
        # Only when we have military advantage
        civ_mil = agent.civ.total_military
        enemy_mil = _enemy_military(agent)
        if civ_mil > enemy_mil * 0.8:    # was 1.2 — any slight edge is enough
            return 0.5                    # was 0.3
        return -0.5   # Penalise suicidal attacks
    if action == RESEARCH:
        # Need surplus wood + minerals
        return (wood / max_r + min_ratio) * 0.3 - 0.1

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

    # Expand: needs unclaimed adjacent tile
    if _has_unclaimed_neighbor(agent):
        feasible[EXPAND] = scores[EXPAND]

    # Fortify: always possible
    feasible[FORTIFY] = scores[FORTIFY]

    # Attack: needs an enemy city within 3 tiles and military advantage > 0.8×
    if _attack_target(agent) is not None and agent.military >= 5:
        feasible[ATTACK] = scores[ATTACK]

    # Research: needs minimal wood and minerals
    from world.resources import ResourceType
    has_resources = (
        agent.model.grid.get(agent.x, agent.y, ResourceType.WOOD) > 10
        and agent.model.grid.get(agent.x, agent.y, ResourceType.MINERALS) > 5
    )
    if has_resources:
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
    t = agent.civ.traits.as_dict()
    scores: dict[str, float] = {}
    for action in ALL_ACTIONS:
        base = sum(_WEIGHTS[action][k] * t[k] for k in t)
        modifier = _resource_modifier(action, agent)
        scores[action] = base + modifier
    return list(_feasible(agent, scores).keys())

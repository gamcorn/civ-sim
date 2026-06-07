# agents/providers/council_prompts.py
from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.civilization import Civilization, CulturalTraits
    from agents.city import CityAgent
    from simulation.model import CivModel


MINISTER_SPECS: list[dict] = [
    {
        "name": "Minister of War",
        "domain": "military strength, threats, and defense",
        "actions": ["attack", "fortify"],
        "trait_key": "aggressiveness",
    },
    {
        "name": "Minister of Economy",
        "domain": "food stocks, trade, and resource management",
        "actions": ["gather", "trade"],
        "trait_key": "trust",
    },
    {
        "name": "Minister of Science",
        "domain": "technology and research advancement",
        "actions": ["research"],
        "trait_key": "innovation",
    },
    {
        "name": "Minister of Expansion",
        "domain": "territorial growth and settlement",
        "actions": ["expand"],
        "trait_key": "tribalism",
    },
]

SECTOR_SCHEMA = json.dumps({
    "analysis": "string",
    "recommendation": "string",
    "weight_requests": {"action_name": 0.5},
    "disagreement_level": 0.0,
}, indent=2)

BUDGET_SCHEMA = json.dumps({
    "veto": False,
    "veto_reason": None,
    "approved_weights": {"action_name": 0.5},
}, indent=2)

CHIEF_SCHEMA = json.dumps({
    "era_goal": "one sentence strategy",
    "action_weights": {
        "gather": 0.0, "trade": 0.0, "expand": 0.0,
        "fortify": 0.0, "attack": 0.0, "research": 0.0,
    },
    "reasoning": "synthesis of minister positions",
    "veto_overridden": False,
    "veto_override_justification": None,
}, indent=2)


def build_sector_persona(spec: dict, traits: "CulturalTraits") -> str:
    trait_val = getattr(traits, spec["trait_key"])
    intensity = "strongly" if trait_val > 0.7 else "moderately" if trait_val > 0.4 else "cautiously"
    return (
        f"You are the {spec['name']} of a civilization. "
        f"Your domain is {spec['domain']}. "
        f"You {intensity} advocate for your portfolio. "
        f"You influence actions: {', '.join(spec['actions'])}. "
        "Reply ONLY with valid JSON matching the schema. No text outside the JSON."
    )


def build_budget_persona(traits: "CulturalTraits") -> str:
    return (
        "You are the Budget Minister. Your only mission is resource arbitration. "
        "You have no sectoral agenda — you enforce what is fiscally feasible given "
        "current food, minerals, and wood levels. "
        "Reply ONLY with valid JSON matching the schema. No text outside the JSON."
    )


def build_chief_persona(traits: "CulturalTraits") -> str:
    boldness = "bold and decisive" if traits.risk_tolerance > 0.6 else "cautious and deliberate"
    return (
        f"You are the Chief of Staff. You are {boldness}. "
        "You synthesize all minister positions into a single strategic directive. "
        "Set action_weights as additive float modifiers in range [-1.0, 1.0]. "
        "Positive weights bias cities toward that action; negative weights bias away. "
        "Reply ONLY with valid JSON matching the schema. No text outside the JSON."
    )


def build_civ_state_snapshot(
    civ: "Civilization", cities: list["CityAgent"], model: "CivModel"
) -> str:
    from world.resources import ResourceType
    if not cities:
        return f"Civilization: {civ.name} — no cities remaining."
    total_pop = sum(c.population for c in cities)
    total_military = sum(c.military for c in cities)
    avg_food = sum(c.food_stock for c in cities) / len(cities)
    enemy_mil = sum(
        c.total_military for c in model.civilizations if c.civ_id != civ.civ_id
    )
    enemy_pop = sum(
        getattr(c, "total_pop", 0) for c in model.civilizations if c.civ_id != civ.civ_id
    )
    total_minerals = sum(model.grid.get(c.x, c.y, ResourceType.MINERALS) for c in cities)
    total_wood = sum(model.grid.get(c.x, c.y, ResourceType.WOOD) for c in cities)
    techs = ", ".join(sorted(civ.discovered_techs)) or "none"
    return (
        f"Turn: {model.steps}\n"
        f"Civilization: {civ.name}\n"
        f"Population: {total_pop}  Military: {total_military}  Cities: {len(cities)}\n"
        f"Enemy population: {enemy_pop}  Enemy military: {enemy_mil}\n"
        f"Avg food stock: {avg_food:.1f}  Minerals (cities): {total_minerals:.0f}"
        f"  Wood (cities): {total_wood:.0f}\n"
        f"Tech level: {civ.tech_level}  Technologies: {techs}\n"
        f"Territory: {model.grid.territory_count(civ.civ_id)} tiles"
    )


def build_sector_user_message(
    spec: dict,
    state_snapshot: str,
    previous_opinions: list[str] | None = None,
) -> str:
    parts = [f"## Civilization State\n{state_snapshot}"]
    if previous_opinions:
        parts.append("## Previous Round Positions\n" + "\n---\n".join(previous_opinions))
    parts.append(
        f"\nProvide your analysis and weight_requests for actions: {', '.join(spec['actions'])}"
    )
    return "\n\n".join(parts)


def build_budget_user_message(state_snapshot: str, sector_outputs: list[dict]) -> str:
    opinions = "\n".join(
        f"**{o.get('name', 'Minister')}**: {o.get('recommendation', '')} "
        f"| weights: {o.get('weight_requests', {})}"
        for o in sector_outputs
    )
    return (
        f"## Civilization State\n{state_snapshot}\n\n"
        f"## Minister Proposals\n{opinions}\n\n"
        "Arbitrate feasibility given current resources."
    )


def build_chief_user_message(
    state_snapshot: str,
    sector_outputs: list[dict],
    budget_output: dict,
    round_num: int,
    max_rounds: int,
) -> str:
    opinions = "\n".join(
        f"**{o.get('name', 'Minister')}** "
        f"(disagreement={o.get('disagreement_level', 0):.0%}): {o.get('recommendation', '')}"
        for o in sector_outputs
    )
    budget_str = (
        f"Veto: {'YES' if budget_output.get('veto') else 'no'}\n"
        f"Reason: {budget_output.get('veto_reason') or '—'}\n"
        f"Approved weights: {budget_output.get('approved_weights', {})}"
    )
    instruction = (
        "This is the final round — produce definitive action_weights."
        if round_num == max_rounds
        else "If ministers converge, produce action_weights now."
    )
    return (
        f"## Civilization State\n{state_snapshot}\n\n"
        f"## Minister Positions\n{opinions}\n\n"
        f"## Budget\n{budget_str}\n\n"
        f"{instruction}"
    )

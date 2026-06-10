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

def build_sector_schema_dict(spec: dict) -> dict:
    """Per-spec guided_json schema that requires the spec's specific action keys."""
    return {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "recommendation": {"type": "string"},
            "weight_requests": {
                "type": "object",
                "properties": {a: {"type": "number"} for a in spec["actions"]},
                "required": spec["actions"],
            },
            "disagreement_level": {"type": "number"},
        },
        "required": ["analysis", "recommendation", "weight_requests", "disagreement_level"],
    }


def build_sector_schema_str(spec: dict) -> str:
    """Human-readable schema example showing actual action names for this spec."""
    return json.dumps({
        "analysis": "brief situation assessment",
        "recommendation": spec["actions"][0],
        "weight_requests": {a: 0.5 for a in spec["actions"]},
        "disagreement_level": 0.3,
    }, indent=2)


# Kept for backwards compat / non-spec-specific uses
SECTOR_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "analysis": {"type": "string"},
        "recommendation": {"type": "string"},
        "weight_requests": {"type": "object", "additionalProperties": {"type": "number"}},
        "disagreement_level": {"type": "number"},
    },
    "required": ["analysis", "recommendation", "weight_requests", "disagreement_level"],
}
SECTOR_SCHEMA = json.dumps({
    "analysis": "brief situation assessment",
    "recommendation": "one action verb",
    "weight_requests": {"gather": 0.5},
    "disagreement_level": 0.3,
}, indent=2)

BUDGET_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "veto": {"type": "boolean"},
        "veto_reason": {"type": ["string", "null"]},
        "approved_weights": {"type": "object", "additionalProperties": {"type": "number"}},
    },
    "required": ["veto", "veto_reason", "approved_weights"],
}
BUDGET_SCHEMA = json.dumps({
    "veto": False,
    "veto_reason": None,
    "approved_weights": {"action_name": 0.5},
}, indent=2)

# veto_overridden removed — simplifies output for smaller models
CHIEF_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "era_goal": {"type": "string"},
        "action_weights": {
            "type": "object",
            "properties": {a: {"type": "number"} for a in ["gather", "trade", "expand", "fortify", "attack", "research"]},
            "required": ["gather", "trade", "expand", "fortify", "attack", "research"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["era_goal", "action_weights", "reasoning"],
}
CHIEF_SCHEMA = json.dumps({
    "era_goal": "one sentence strategy",
    "action_weights": {
        "gather": 0.0, "trade": 0.0, "expand": 0.0,
        "fortify": 0.0, "attack": 0.0, "research": 0.0,
    },
    "reasoning": "brief synthesis",
}, indent=2)

# Lite-mode chief schema (council_off): action_weights only
CHIEF_LITE_SCHEMA_DICT = {
    "type": "object",
    "properties": {
        "action_weights": {
            "type": "object",
            "properties": {a: {"type": "number"} for a in ["gather", "trade", "expand", "fortify", "attack", "research"]},
            "required": ["gather", "trade", "expand", "fortify", "attack", "research"],
        },
    },
    "required": ["action_weights"],
}
CHIEF_LITE_SCHEMA = json.dumps({
    "action_weights": {
        "gather": 0.0, "trade": 0.0, "expand": 0.0,
        "fortify": 0.0, "attack": 0.0, "research": 0.0,
    },
}, indent=2)


_JSON_ONLY = (
    "Output ONLY a JSON object. Do not write any explanation, reasoning, or commentary. "
    "Your entire response must start with { and end with }."
)


def build_sector_persona(spec: dict, traits: "CulturalTraits") -> str:
    trait_val = getattr(traits, spec["trait_key"])
    intensity = "strongly" if trait_val > 0.7 else "moderately" if trait_val > 0.4 else "cautiously"
    return (
        f"You are the {spec['name']} of a civilization. "
        f"Your domain is {spec['domain']}. "
        f"You {intensity} advocate for your portfolio. "
        f"You influence actions: {', '.join(spec['actions'])}. "
        f"{_JSON_ONLY}"
    )


def build_budget_persona(traits: "CulturalTraits") -> str:
    return (
        "You are the Budget Minister. Your only mission is resource arbitration. "
        "You have no sectoral agenda — you enforce what is fiscally feasible given "
        f"current food, minerals, and wood levels. {_JSON_ONLY}"
    )


def build_chief_persona(traits: "CulturalTraits") -> str:
    boldness = "bold and decisive" if traits.risk_tolerance > 0.6 else "cautious and deliberate"
    return (
        f"You are the Chief of Staff. You are {boldness}. "
        "You synthesize all minister positions into a single strategic directive. "
        "Set action_weights as additive float modifiers in range [-1.0, 1.0]. "
        f"Positive weights bias cities toward that action; negative weights bias away. {_JSON_ONLY}"
    )


def build_chief_lite_persona(traits: "CulturalTraits") -> str:
    boldness = "bold and decisive" if traits.risk_tolerance > 0.6 else "cautious and deliberate"
    return (
        f"You are a {boldness} strategic commander. "
        "Assign action_weights (floats in [-1.0, 1.0]) to guide your civilization. "
        f"Positive = prioritize that action, negative = avoid it. {_JSON_ONLY}"
    )


def build_civ_state_snapshot(
    civ: "Civilization", cities: list["CityAgent"], model: "CivModel",
    fog_of_war: float = 0.0,
) -> str:
    from world.resources import ResourceType
    if not cities:
        return f"Civilization: {civ.name} — no cities remaining."
    total_pop = sum(c.population for c in cities)
    total_military = sum(c.military for c in cities)
    avg_food = sum(c.food_stock for c in cities) / len(cities)
    total_minerals = sum(model.grid.get(c.x, c.y, ResourceType.MINERALS) for c in cities)
    total_wood = sum(model.grid.get(c.x, c.y, ResourceType.WOOD) for c in cities)
    techs = ", ".join(sorted(civ.discovered_techs)) or "none"

    own_block = (
        f"Turn: {model.steps}\n"
        f"Civilization: {civ.name}\n"
        f"Population: {total_pop}  Military: {total_military}  Cities: {len(cities)}\n"
        f"Avg food stock: {avg_food:.1f}  Minerals (cities): {total_minerals:.0f}"
        f"  Wood (cities): {total_wood:.0f}\n"
        f"Tech level: {civ.tech_level}  Technologies: {techs}\n"
        f"Territory: {model.grid.territory_count(civ.civ_id)} tiles"
    )

    enemies = [c for c in model.civilizations if c.civ_id != civ.civ_id]
    if not enemies:
        return own_block

    noisy = fog_of_war > 0.0
    rng = model.random if noisy else None

    def _fmt(val: float | int) -> str:
        if not noisy:
            return str(int(val))
        factor = rng.uniform(max(0.01, 1.0 - fog_of_war), 1.0 + fog_of_war)
        return f"~{int(round(float(val) * factor))}"

    header = (
        f"Intelligence Report (fog={fog_of_war:.0%}):"
        if noisy else "Intelligence Report:"
    )
    lines = []
    for e in enemies:
        city_count = getattr(e, "city_count", 0)
        territory = model.grid.territory_count(e.civ_id)
        lines.append(
            f"  {e.name}: pop {_fmt(e.total_pop)} | military {_fmt(e.total_military)}"
            f" | cities {_fmt(city_count)} | tech {_fmt(e.tech_level)}"
            f" | territory {_fmt(territory)} tiles"
        )

    return own_block + "\n\n" + header + "\n" + "\n".join(lines)


def build_sector_user_message(
    spec: dict,
    state_snapshot: str,
    previous_opinions: list[str] | None = None,
) -> str:
    parts = [f"## Civilization State\n{state_snapshot}"]
    if previous_opinions:
        parts.append("## Previous Round Positions\n" + "\n---\n".join(previous_opinions))
    action_keys = ", ".join(f'"{a}"' for a in spec["actions"])
    parts.append(
        f"\nProvide your analysis. In weight_requests, use exactly these keys: {action_keys}. "
        f"Values are floats (0.0 = neutral, 1.0 = strongly prioritize)."
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


def build_chief_lite_user_message(state_snapshot: str) -> str:
    return (
        f"## Civilization State\n{state_snapshot}\n\n"
        "Set action_weights to guide your cities this period."
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

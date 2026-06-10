from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent

SYSTEM_PROMPT = (
    "You are the strategic advisor for a city in a civilization simulation.\n"
    "Respond with ONLY one word — the action to take this turn.\n"
    "Choose from the list provided. No explanation."
)


def build_prompt(city: "CityAgent", feasible_actions: list[str]) -> str:
    civ = city.civ
    t = civ.traits
    grid = city.model.grid
    enemy_mil = sum(
        c.total_military for c in city.model.civilizations if c.civ_id != civ.civ_id
    )
    techs = ", ".join(sorted(civ.discovered_techs)) or "none"
    return (
        f"Turn {city.model.steps} | City of {civ.name} civilization\n"
        f"Population: {city.population}  Military: {city.military}"
        f"  Food stock: {city.food_stock:.0f}\n"
        f"Traits: aggression={t.aggressiveness:.2f}  trust={t.trust:.2f}"
        f"  innovation={t.innovation:.2f}\n"
        f"Technologies: {techs}\n"
        f"Territory: {grid.territory_count(civ.civ_id)} tiles"
        f"  |  Enemy military: {enemy_mil}\n"
        f"Available actions: {', '.join(feasible_actions)}"
    )


def parse_response(text: str, feasible: list[str], fallback: str) -> str:
    """Extract the first word from LLM text; return fallback if not in feasible set."""
    if not text or not text.strip():
        return fallback
    first_word = text.strip().lower().split()[0]
    cleaned = re.sub(r"[^\w]", "", first_word)
    return cleaned if cleaned in feasible else fallback

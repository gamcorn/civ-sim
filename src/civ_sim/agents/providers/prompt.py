from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from civ_sim.agents.city import CityAgent

logger = logging.getLogger(__name__)

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

    # Build relations line
    other_civs = [c for c in city.model.civilizations if c.civ_id != civ.civ_id]
    if other_civs:
        rel_parts = []
        for other in other_civs:
            rel = city.model.get_relation(civ.civ_id, other.civ_id)
            rel_parts.append(f"{other.name}: {rel:+.2f}")
        relations_line = "Relations: " + ", ".join(rel_parts)
    else:
        relations_line = ""

    prompt = (
        f"Turn {city.model.steps} | City of {civ.name} civilization\n"
        f"Population: {city.population}  Military: {city.military}"
        f"  Food stock: {city.food_stock:.0f}\n"
        f"Traits: aggression={t.aggressiveness:.2f}  trust={t.trust:.2f}"
        f"  innovation={t.innovation:.2f}\n"
        f"Technologies: {techs}\n"
        f"Territory: {grid.territory_count(civ.civ_id)} tiles"
        f"  |  Enemy military: {enemy_mil}\n"
    )
    if relations_line:
        prompt += f"{relations_line}\n"
    prompt += f"Available actions: {', '.join(feasible_actions)}"
    return prompt


def parse_response(text: str, feasible: list[str], fallback: str) -> str:
    """Extract the first word from LLM text; return fallback if not in feasible set."""
    if not text or not text.strip():
        logger.debug("Empty LLM response; using fallback=%s", fallback)
        return fallback
    first_word = text.strip().lower().split()[0]
    cleaned = re.sub(r"[^\w]", "", first_word)
    if cleaned not in feasible:
        logger.debug(
            "LLM response %r not in feasible set %s; using fallback=%s",
            cleaned,
            feasible,
            fallback,
        )
        return fallback
    return cleaned

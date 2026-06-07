# agents/providers/council_ministers.py
from __future__ import annotations
import asyncio
import json
from typing import Any, TYPE_CHECKING

import openai

from agents.providers.council_prompts import (
    SECTOR_SCHEMA, BUDGET_SCHEMA, CHIEF_SCHEMA,
    build_sector_persona, build_budget_persona, build_chief_persona,
    build_sector_user_message, build_budget_user_message, build_chief_user_message,
)

if TYPE_CHECKING:
    from agents.civilization import CulturalTraits


def _parse_json_safe(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def call_sector_minister(
    spec: dict,
    state_snapshot: str,
    traits: "CulturalTraits",
    client: openai.AsyncOpenAI,
    model: str,
    timeout: float,
    previous_opinions: list[str] | None = None,
) -> dict:
    persona = build_sector_persona(spec, traits)
    user_msg = build_sector_user_message(spec, state_snapshot, previous_opinions)
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{persona}\n\nSchema:\n{SECTOR_SCHEMA}"},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=256,
            ),
            timeout=timeout,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json_safe(raw)
        if parsed is not None:
            parsed["name"] = spec["name"]
            return parsed
    except Exception:
        pass
    return {
        "name": spec["name"],
        "analysis": "",
        "recommendation": "",
        "weight_requests": {},
        "disagreement_level": 0.0,
    }


async def call_budget_minister(
    state_snapshot: str,
    sector_outputs: list[dict],
    traits: "CulturalTraits",
    client: openai.AsyncOpenAI,
    model: str,
    timeout: float,
) -> dict:
    persona = build_budget_persona(traits)
    user_msg = build_budget_user_message(state_snapshot, sector_outputs)
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{persona}\n\nSchema:\n{BUDGET_SCHEMA}"},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=256,
            ),
            timeout=timeout,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json_safe(raw)
        if parsed is not None:
            return parsed
    except Exception:
        pass
    return {"veto": False, "veto_reason": None, "approved_weights": {}}


async def call_chief(
    state_snapshot: str,
    sector_outputs: list[dict],
    budget_output: dict,
    traits: "CulturalTraits",
    client: openai.AsyncOpenAI,
    model: str,
    timeout: float,
    round_num: int = 1,
    max_rounds: int = 2,
) -> dict | None:
    from agents.decisions import ALL_ACTIONS
    persona = build_chief_persona(traits)
    user_msg = build_chief_user_message(
        state_snapshot, sector_outputs, budget_output, round_num, max_rounds
    )
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{persona}\n\nSchema:\n{CHIEF_SCHEMA}"},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=512,
            ),
            timeout=timeout,
        )
        raw = resp.choices[0].message.content or ""
        parsed = _parse_json_safe(raw)
        if parsed is None:
            return None
        weights = parsed.get("action_weights", {})
        parsed["action_weights"] = {
            a: max(-1.0, min(1.0, float(weights.get(a, 0.0))))
            for a in ALL_ACTIONS
        }
        return parsed
    except Exception:
        return None

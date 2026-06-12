# agents/providers/council_ministers.py
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import openai

from civ_sim.agents.decisions import ALL_ACTIONS
from civ_sim.agents.providers.council_prompts import (
    BUDGET_SCHEMA,
    BUDGET_SCHEMA_DICT,
    CHIEF_LITE_SCHEMA,
    CHIEF_LITE_SCHEMA_DICT,
    CHIEF_SCHEMA,
    CHIEF_SCHEMA_DICT,
    build_budget_persona,
    build_budget_user_message,
    build_chief_lite_persona,
    build_chief_lite_user_message,
    build_chief_persona,
    build_chief_user_message,
    build_sector_persona,
    build_sector_schema_dict,
    build_sector_schema_str,
    build_sector_user_message,
)

if TYPE_CHECKING:
    from civ_sim.agents.civilization import CulturalTraits

logger = logging.getLogger(__name__)


def _parse_json_safe(raw: str) -> dict[str, Any] | None:
    text = raw.strip()

    # Thinking models (e.g. Nemotron-Nano) output <think>...</think> CoT before JSON.
    # Strip everything up to and including the closing </think> tag.
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()

    # Strip markdown code block wrapper
    if text.startswith("```"):
        lines = text.splitlines()
        end = next(
            (i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"),
            len(lines),
        )
        text = "\n".join(lines[1:end]).strip()

    # Direct parse (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract the outermost {...} blob — model may have prefixed with residual text
    last_brace = text.rfind("}")
    if last_brace == -1:
        logger.debug(
            "JSON parse failed: no closing brace in response (len=%d)", len(raw)
        )
        return None
    text = text[: last_brace + 1]
    pos = text.rfind("{")
    while pos != -1:
        try:
            result = json.loads(text[pos:])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        pos = text.rfind("{", 0, pos)
    logger.debug(
        "JSON parse failed: no valid JSON object found in response (len=%d)", len(raw)
    )
    return None


async def _call_llm(
    client: openai.AsyncOpenAI,
    model: str,
    timeout: float,
    *,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    guided_json: dict | None = None,
) -> dict[str, Any] | None:
    extra: dict[str, Any] = {}
    if guided_json is not None:
        extra["extra_body"] = {"guided_json": guided_json}
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            ),
            timeout=timeout,
        )
        raw = resp.choices[0].message.content or ""
        return _parse_json_safe(raw)
    except Exception as exc:
        logger.warning("Council LLM call failed (model=%s): %s", model, exc)
        return None


async def call_sector_minister(
    spec: dict,
    state_snapshot: str,
    traits: "CulturalTraits",
    client: openai.AsyncOpenAI,
    model: str,
    timeout: float,
    previous_opinions: list[str] | None = None,
    guided_json: bool = False,
) -> dict:
    persona = build_sector_persona(spec, traits)
    user_msg = build_sector_user_message(spec, state_snapshot, previous_opinions)
    schema_str = build_sector_schema_str(spec)
    schema_dict = build_sector_schema_dict(spec) if guided_json else None
    parsed = await _call_llm(
        client,
        model,
        timeout,
        system=f"{persona}\n\nSchema:\n{schema_str}",
        user=user_msg,
        temperature=0.3,
        max_tokens=1024,
        guided_json=schema_dict,
    )
    if parsed is not None:
        parsed["name"] = spec["name"]
        return parsed
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
    guided_json: bool = False,
) -> dict:
    persona = build_budget_persona(traits)
    user_msg = build_budget_user_message(state_snapshot, sector_outputs)
    parsed = await _call_llm(
        client,
        model,
        timeout,
        system=f"{persona}\n\nSchema:\n{BUDGET_SCHEMA}",
        user=user_msg,
        temperature=0.2,
        max_tokens=512,
        guided_json=BUDGET_SCHEMA_DICT if guided_json else None,
    )
    return (
        parsed
        if parsed is not None
        else {"veto": False, "veto_reason": None, "approved_weights": {}}
    )


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
    guided_json: bool = False,
) -> dict | None:
    persona = build_chief_persona(traits)
    user_msg = build_chief_user_message(
        state_snapshot, sector_outputs, budget_output, round_num, max_rounds
    )
    parsed = await _call_llm(
        client,
        model,
        timeout,
        system=f"{persona}\n\nSchema:\n{CHIEF_SCHEMA}",
        user=user_msg,
        temperature=0.2,
        max_tokens=2048,
        guided_json=CHIEF_SCHEMA_DICT if guided_json else None,
    )
    if parsed is None:
        return None
    weights = parsed.get("action_weights", {})
    parsed["action_weights"] = {
        a: max(-1.0, min(1.0, float(weights.get(a, 0.0)))) for a in ALL_ACTIONS
    }
    return parsed


async def call_chief_lite(
    state_snapshot: str,
    traits: "CulturalTraits",
    client: openai.AsyncOpenAI,
    model: str,
    timeout: float,
    guided_json: bool = False,
) -> dict | None:
    """Single-call chief used when council_off=True. Skips minister debate."""
    persona = build_chief_lite_persona(traits)
    user_msg = build_chief_lite_user_message(state_snapshot)
    parsed = await _call_llm(
        client,
        model,
        timeout,
        system=f"{persona}\n\nSchema:\n{CHIEF_LITE_SCHEMA}",
        user=user_msg,
        temperature=0.2,
        max_tokens=512,
        guided_json=CHIEF_LITE_SCHEMA_DICT if guided_json else None,
    )
    if parsed is None:
        return None
    weights = parsed.get("action_weights", {})
    parsed["action_weights"] = {
        a: max(-1.0, min(1.0, float(weights.get(a, 0.0)))) for a in ALL_ACTIONS
    }
    return parsed

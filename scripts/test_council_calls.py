#!/usr/bin/env python3
"""
Verification: test live council LLM calls with the improved parsing.
Run: .venv/bin/python scripts/test_council_calls.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai

from civ_sim.agents.civilization import CulturalTraits
from civ_sim.agents.providers.council_ministers import (
    _parse_json_safe,
    call_budget_minister,
    call_chief,
    call_sector_minister,
)
from civ_sim.agents.providers.council_prompts import MINISTER_SPECS

BASE_URL = "http://localhost:8000/v1"
MODEL = "model"
TIMEOUT = 10.0
CHIEF_TIMEOUT = 30.0

FAKE_SNAPSHOT = """\
Turn: 50
Civilization: Alpha
Population: 100  Military: 20  Cities: 2
Enemy population: 120  Enemy military: 25
Avg food stock: 45.0  Minerals (cities): 200  Wood (cities): 400
Tech level: 3  Technologies: agriculture, masonry
Territory: 150 tiles\
"""

TRAITS = CulturalTraits(
    aggressiveness=0.6,
    trust=0.4,
    innovation=0.5,
    tribalism=0.5,
    risk_tolerance=0.6,
)

SEP = "─" * 72


async def main():
    client = openai.AsyncOpenAI(base_url=BASE_URL, api_key="EMPTY")
    n_samples = 2

    print(f"=== Testing improved parsing — {n_samples} samples each ===\n")

    # ── Test _parse_json_safe improvements ────────────────────────────
    print("## _parse_json_safe edge cases")
    test_cases = [
        (
            "think tag",
            'We should do X.\n</think>\n{"analysis": "ok", "recommendation": "gather", "weight_requests": {"gather": 0.7}, "disagreement_level": 0.1}',
        ),
        (
            "text before",
            'Here is my analysis: {"analysis": "ok", "recommendation": "gather", "weight_requests": {}, "disagreement_level": 0.0}',
        ),
        (
            "schema confusion",
            'The schema shows { "action_name": 0.5 }. My answer: {"analysis": "ok", "recommendation": "research", "weight_requests": {"research": 0.8}, "disagreement_level": 0.0}',
        ),
    ]
    for label, text in test_cases:
        result = _parse_json_safe(text)
        status = "OK" if result else "FAIL"
        print(f"  [{status}] {label}: {list(result.keys()) if result else 'None'}")
    print()

    # ── Sector ministers without guided_json ─────────────────────────
    print("## Sector ministers (standard, max_tokens=1024)")
    ok_count = 0
    total = 0
    for spec in MINISTER_SPECS:
        for i in range(n_samples):
            total += 1
            result = await call_sector_minister(
                spec,
                FAKE_SNAPSHOT,
                TRAITS,
                client,
                MODEL,
                TIMEOUT,
                guided_json=False,
            )
            has_weights = bool(result.get("weight_requests"))
            status = "OK" if has_weights else "empty"
            if has_weights:
                ok_count += 1
            print(
                f"  [{status}] {spec['name']} s{i+1}: weights={result.get('weight_requests', {})}"
            )
    print(f"  Success rate: {ok_count}/{total}\n")

    # ── Sector ministers with guided_json ────────────────────────────
    print("## Sector ministers (guided_json=True)")
    ok_gj = 0
    for spec in MINISTER_SPECS:
        for i in range(n_samples):
            result = await call_sector_minister(
                spec,
                FAKE_SNAPSHOT,
                TRAITS,
                client,
                MODEL,
                TIMEOUT,
                guided_json=True,
            )
            has_weights = bool(result.get("weight_requests"))
            status = "OK" if has_weights else "empty"
            if has_weights:
                ok_gj += 1
            print(
                f"  [{status}] {spec['name']} s{i+1}: weights={result.get('weight_requests', {})}"
            )
    print(f"  Success rate: {ok_gj}/{total}\n")

    # ── Budget minister ───────────────────────────────────────────────
    print("## Budget minister")
    fake_sector_outputs = [
        {
            "name": "Minister of War",
            "recommendation": "fortify",
            "weight_requests": {"fortify": 0.5},
            "disagreement_level": 0.2,
        },
        {
            "name": "Minister of Economy",
            "recommendation": "gather",
            "weight_requests": {"gather": 0.7},
            "disagreement_level": 0.1,
        },
        {
            "name": "Minister of Science",
            "recommendation": "research",
            "weight_requests": {"research": 0.6},
            "disagreement_level": 0.2,
        },
        {
            "name": "Minister of Expansion",
            "recommendation": "expand",
            "weight_requests": {"expand": 0.4},
            "disagreement_level": 0.4,
        },
    ]
    for i in range(n_samples):
        result = await call_budget_minister(
            FAKE_SNAPSHOT,
            fake_sector_outputs,
            TRAITS,
            client,
            MODEL,
            TIMEOUT,
            guided_json=True,
        )
        print(
            f"  Budget s{i+1}: veto={result.get('veto')}, approved={result.get('approved_weights', {})}"
        )
    print()

    # ── Chief (full council, guided_json) ─────────────────────────────
    print(f"## Chief of Staff (guided_json=True, timeout={CHIEF_TIMEOUT}s)")
    fake_budget = {
        "veto": False,
        "veto_reason": None,
        "approved_weights": {"gather": 0.7, "fortify": 0.5},
    }
    for i in range(n_samples):
        result = await call_chief(
            FAKE_SNAPSHOT,
            fake_sector_outputs,
            fake_budget,
            TRAITS,
            client,
            MODEL,
            CHIEF_TIMEOUT,
            round_num=2,
            max_rounds=2,
            guided_json=True,
        )
        if result:
            weights = result.get("action_weights", {})
            print(f"  [OK] Chief s{i+1}: era_goal={result.get('era_goal', '')!r}")
            print(f"       weights: {weights}")
        else:
            print(f"  [FAIL] Chief s{i+1}: returned None (timeout or parse error)")
    print()

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())

"""Tests for agents/decisions.py — choose_action, modifiers, and feasibility."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from civ_sim.agents.decisions import (
    ALL_ACTIONS,
    ATTACK,
    EXPAND,
    FORTIFY,
    GATHER,
    RESEARCH,
    TRADE,
    _attack_target,
    _has_unclaimed_neighbor,
    _resource_modifier,
    choose_action,
    get_action_scores,
    get_feasible_actions,
)
from tests.conftest import make_mock_city


# ---------------------------------------------------------------------------
# get_action_scores
# ---------------------------------------------------------------------------


def test_get_action_scores_returns_all_six_actions():
    city = make_mock_city()
    scores = get_action_scores(city)
    assert set(scores.keys()) == set(ALL_ACTIONS)
    assert all(isinstance(v, float) for v in scores.values())


def test_get_action_scores_gather_higher_when_food_low():
    city_hungry = make_mock_city(food_stock=0.0)
    city_full = make_mock_city(food_stock=500.0)
    assert get_action_scores(city_hungry)[GATHER] > get_action_scores(city_full)[GATHER]


def test_choose_action_picks_max_feasible_score():
    city = make_mock_city()
    scores = get_action_scores(city)
    action = choose_action(city)
    feasible = get_feasible_actions(city)
    assert action in feasible
    best_feasible_score = max(scores[a] for a in feasible)
    assert scores[action] == best_feasible_score


# ---------------------------------------------------------------------------
# choose_action — basic
# ---------------------------------------------------------------------------


def test_choose_action_returns_valid_action():
    city = make_mock_city()
    result = choose_action(city)
    assert result in ALL_ACTIONS


def test_choose_action_high_aggressiveness_prefers_attack_or_fortify():
    """High aggressiveness + strong military + enemy nearby → attack or fortify.

    Tribalism is set to 0 to suppress the expand score (tribalism weight 0.5 on
    expand easily beats everything otherwise).  risk_tolerance=0.8 further boosts
    attack.  trust=0 removes the trade bonus.
    """
    city = make_mock_city(
        aggressiveness=0.95,
        military=50,
        enemy_military=10,
        tribalism=0.0,
        trust=0.0,
        risk_tolerance=0.8,
    )
    # Add an enemy city within 10 tiles so ATTACK is feasible
    enemy = MagicMock()
    enemy.civ.civ_id = 1  # different from default civ_id=0
    enemy.x = 14  # Manhattan distance 4 from (10,10)
    enemy.y = 10
    city.model.agents_by_type = {type(city): [enemy]}
    result = choose_action(city)
    assert result in (ATTACK, FORTIFY)


def test_choose_action_high_innovation_can_choose_research():
    """High innovation + sufficient resources → must not crash; result is valid."""
    city = make_mock_city(innovation=1.0, wood=80.0, minerals=60.0)
    result = choose_action(city)
    assert result in ALL_ACTIONS


# ---------------------------------------------------------------------------
# _resource_modifier — gather
# ---------------------------------------------------------------------------


def test_resource_modifier_gather_high_when_food_stock_low():
    """When food_stock is 0 the gather modifier should be well above 0.3."""
    city = make_mock_city(food_stock=0.0)
    modifier = _resource_modifier(GATHER, city)
    assert modifier > 0.3


def test_resource_modifier_gather_low_when_food_stock_high():
    """When food_stock is very high the gather modifier collapses to 0.0."""
    # stock_ratio = min(1.0, 200 / 200) = 1.0 → modifier = max(0, 0.4-1.0)*1.5 = 0.0
    city = make_mock_city(food_stock=200.0)
    modifier = _resource_modifier(GATHER, city)
    assert modifier == 0.0


# ---------------------------------------------------------------------------
# _resource_modifier — fortify
# ---------------------------------------------------------------------------


def test_resource_modifier_fortify_scales_with_enemy_military():
    """A weak city (outgunned) must receive a larger fortify modifier than a strong one."""
    weak_city = make_mock_city(military=5, enemy_military=50)
    strong_city = make_mock_city(military=50, enemy_military=5)
    weak_mod = _resource_modifier(FORTIFY, weak_city)
    strong_mod = _resource_modifier(FORTIFY, strong_city)
    assert weak_mod > strong_mod


def test_resource_modifier_fortify_zero_when_stronger():
    """If we outgun the enemy, fortify modifier must be exactly 0.0."""
    city = make_mock_city(military=100, enemy_military=10)
    modifier = _resource_modifier(FORTIFY, city)
    assert modifier == 0.0


# ---------------------------------------------------------------------------
# _has_unclaimed_neighbor
# ---------------------------------------------------------------------------


def test_has_unclaimed_neighbor_true_by_default():
    """Default mock city has unclaimed tiles in the grid → should return True."""
    city = make_mock_city()
    assert _has_unclaimed_neighbor(city) is True


def test_has_unclaimed_neighbor_false_when_all_claimed():
    """When every tile in the grid is claimed, the function returns False."""
    city = make_mock_city()
    # Claim every cell so nothing is -1
    city.model.grid.ownership[:] = 0
    assert _has_unclaimed_neighbor(city) is False


# ---------------------------------------------------------------------------
# _attack_target
# ---------------------------------------------------------------------------


def test_attack_target_returns_none_when_no_enemies():
    """No agents in agents_by_type → _attack_target must return None."""
    city = make_mock_city()
    city.model.agents_by_type = {}
    assert _attack_target(city) is None


def test_attack_target_returns_nearest_enemy_within_10():
    """An enemy at Manhattan distance 5 should be returned as the attack target."""
    city = make_mock_city()
    enemy = MagicMock()
    enemy.civ.civ_id = 1  # different from city.civ.civ_id == 0
    enemy.x = 15  # |15-10| + |10-10| = 5
    enemy.y = 10
    city.model.agents_by_type = {type(city): [enemy]}
    result = _attack_target(city)
    assert result is enemy


def test_attack_target_returns_none_when_only_enemy_beyond_25():
    """An enemy at distance 30 is outside the 25-tile radius → None."""
    city = make_mock_city()
    enemy = MagicMock()
    enemy.civ.civ_id = 1
    enemy.x = 40  # |40-10| = 30
    enemy.y = 10
    city.model.agents_by_type = {type(city): [enemy]}
    result = _attack_target(city)
    assert result is None


# ---------------------------------------------------------------------------
# get_feasible_actions — GATHER and FORTIFY always present
# ---------------------------------------------------------------------------


def test_feasible_always_includes_gather_and_fortify():
    """GATHER and FORTIFY must always appear in the feasible action list."""
    city = make_mock_city()
    feasible = get_feasible_actions(city)
    assert GATHER in feasible
    assert FORTIFY in feasible


def test_feasible_excludes_attack_when_military_below_5():
    """ATTACK is infeasible when military < 5, even with an enemy nearby."""
    city = make_mock_city(military=4, enemy_military=10)
    # Place an enemy within 10 tiles to satisfy the distance check
    enemy = MagicMock()
    enemy.civ.civ_id = 1
    enemy.x = 14
    enemy.y = 10
    city.model.agents_by_type = {type(city): [enemy]}
    feasible = get_feasible_actions(city)
    assert ATTACK not in feasible


def test_feasible_excludes_research_when_resources_low():
    """RESEARCH requires wood_stock >= research_wood_cost and mineral_stock >= research_mineral_cost."""
    city = make_mock_city(wood_stock=3.0, mineral_stock=2.0)
    feasible = get_feasible_actions(city)
    assert RESEARCH not in feasible


# ---------------------------------------------------------------------------
# _resource_modifier — attack (lowered threshold)
# ---------------------------------------------------------------------------


def test_attack_modifier_positive_with_slight_military_advantage():
    """civ_mil > enemy * 0.8 (equal strength qualifies) → +0.5 modifier."""
    city = make_mock_city(military=20, enemy_military=20)
    # civ.total_military=20, enemy.total_military=20 → 20 > 20*0.8=16 → True
    modifier = _resource_modifier(ATTACK, city)
    assert modifier == 0.5, f"Expected 0.5, got {modifier}"


def test_attack_modifier_negative_when_clearly_outgunned():
    """civ_mil not > enemy * 0.8 → -0.5 modifier discourages suicidal attack."""
    city = make_mock_city(military=10, enemy_military=20)
    # 10 > 20*0.8=16 → False
    modifier = _resource_modifier(ATTACK, city)
    assert modifier == -0.5, f"Expected -0.5, got {modifier}"

import asyncio
from tests.conftest import make_mock_city


def test_rule_based_returns_one_action_per_city():
    from civ_sim.agents.providers.rule_based import RuleBasedProvider
    provider = RuleBasedProvider()
    cities = [make_mock_city(), make_mock_city()]
    results = asyncio.run(provider.choose_actions_batch(cities))
    assert len(results) == 2


def test_rule_based_returns_valid_action_strings():
    from civ_sim.agents.providers.rule_based import RuleBasedProvider
    from civ_sim.agents.decisions import ALL_ACTIONS
    provider = RuleBasedProvider()
    city = make_mock_city()
    results = asyncio.run(provider.choose_actions_batch([city]))
    assert results[0] in ALL_ACTIONS


def test_rule_based_empty_batch_returns_empty_list():
    from civ_sim.agents.providers.rule_based import RuleBasedProvider
    provider = RuleBasedProvider()
    results = asyncio.run(provider.choose_actions_batch([]))
    assert results == []


def test_rule_based_is_deterministic_for_same_input():
    from civ_sim.agents.providers.rule_based import RuleBasedProvider
    provider = RuleBasedProvider()
    city = make_mock_city(aggressiveness=0.9, trust=0.1, tribalism=0.8)
    r1 = asyncio.run(provider.choose_actions_batch([city]))
    r2 = asyncio.run(provider.choose_actions_batch([city]))
    assert r1 == r2

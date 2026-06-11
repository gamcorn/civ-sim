from tests.conftest import make_mock_city


def test_get_feasible_actions_always_includes_gather():
    from civ_sim.agents.decisions import get_feasible_actions

    city = make_mock_city()
    actions = get_feasible_actions(city)
    assert "gather" in actions


def test_get_feasible_actions_returns_list_of_strings():
    from civ_sim.agents.decisions import get_feasible_actions

    city = make_mock_city()
    actions = get_feasible_actions(city)
    assert isinstance(actions, list)
    assert all(isinstance(a, str) for a in actions)


def test_get_feasible_actions_expand_requires_unclaimed_neighbor():
    import numpy as np

    from civ_sim.agents.decisions import get_feasible_actions

    city = make_mock_city()
    # Fill all tiles as owned — no unclaimed neighbors
    city.model.grid.ownership = np.full((80, 60), 0, dtype=np.int8)
    actions = get_feasible_actions(city)
    assert "expand" not in actions


def test_provider_config_defaults():
    from civ_sim.config import ProviderConfig

    cfg = ProviderConfig()
    assert cfg.type == "rule_based"
    assert cfg.timeout == 5.0
    assert cfg.max_tokens == 10


def test_sim_config_has_civ_providers():
    from civ_sim.config import ProviderConfig, SimConfig

    cfg = SimConfig()
    assert len(cfg.civ_providers) == 2
    assert all(isinstance(p, ProviderConfig) for p in cfg.civ_providers)


def test_build_prompt_contains_turn_and_civ_name():
    from civ_sim.agents.providers.prompt import build_prompt

    city = make_mock_city(tick=42, civ_name="Alpha")
    prompt = build_prompt(city, ["gather", "expand"])
    assert "Turn 42" in prompt
    assert "Alpha" in prompt


def test_build_prompt_contains_feasible_actions():
    from civ_sim.agents.providers.prompt import build_prompt

    city = make_mock_city()
    prompt = build_prompt(city, ["gather", "research"])
    assert "gather" in prompt
    assert "research" in prompt


def test_parse_response_returns_valid_action():
    from civ_sim.agents.providers.prompt import parse_response

    assert parse_response("expand\n", ["gather", "expand"], "gather") == "expand"


def test_parse_response_falls_back_on_garbage():
    from civ_sim.agents.providers.prompt import parse_response

    assert (
        parse_response("I cannot decide!", ["gather", "expand"], "gather") == "gather"
    )


def test_parse_response_falls_back_on_empty():
    from civ_sim.agents.providers.prompt import parse_response

    assert parse_response("", ["gather"], "gather") == "gather"


def test_parse_response_strips_punctuation():
    from civ_sim.agents.providers.prompt import parse_response

    assert parse_response("expand.", ["gather", "expand"], "gather") == "expand"


def test_build_prompt_includes_relations_line():
    from civ_sim.agents.providers.prompt import build_prompt

    city = make_mock_city(civ_name="Alpha")
    prompt = build_prompt(city, ["gather"])
    assert (
        "Relations:" in prompt
    ), "Prompt must include a Relations line for diplomatic context"

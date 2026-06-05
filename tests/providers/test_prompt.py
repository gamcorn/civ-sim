from tests.conftest import make_mock_city


def test_get_feasible_actions_always_includes_gather():
    from agents.decisions import get_feasible_actions
    city = make_mock_city()
    actions = get_feasible_actions(city)
    assert "gather" in actions


def test_get_feasible_actions_returns_list_of_strings():
    from agents.decisions import get_feasible_actions
    city = make_mock_city()
    actions = get_feasible_actions(city)
    assert isinstance(actions, list)
    assert all(isinstance(a, str) for a in actions)


def test_get_feasible_actions_expand_requires_unclaimed_neighbor():
    from agents.decisions import get_feasible_actions
    import numpy as np
    city = make_mock_city()
    # Fill all tiles as owned — no unclaimed neighbors
    city.model.grid.ownership = np.full((80, 60), 0, dtype=np.int8)
    actions = get_feasible_actions(city)
    assert "expand" not in actions


def test_provider_config_defaults():
    from config import ProviderConfig
    cfg = ProviderConfig()
    assert cfg.type == "rule_based"
    assert cfg.timeout == 5.0
    assert cfg.max_tokens == 10


def test_sim_config_has_civ_providers():
    from config import SimConfig, ProviderConfig
    cfg = SimConfig()
    assert len(cfg.civ_providers) == 2
    assert all(isinstance(p, ProviderConfig) for p in cfg.civ_providers)

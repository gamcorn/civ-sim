from config import SimConfig


def test_civilization_has_provider_attribute():
    from agents.providers.rule_based import RuleBasedProvider
    from agents.civilization import Civilization, CulturalTraits

    civ = Civilization(civ_id=0, name="Alpha", traits=CulturalTraits())
    assert hasattr(civ, "provider")
    assert isinstance(civ.provider, RuleBasedProvider)


def test_civilization_accepts_custom_provider():
    from agents.providers.rule_based import RuleBasedProvider
    from agents.civilization import Civilization, CulturalTraits

    provider = RuleBasedProvider()
    civ = Civilization(civ_id=0, name="Alpha", traits=CulturalTraits(), provider=provider)
    assert civ.provider is provider


def test_city_agent_has_pending_action_none_initially():
    cfg = SimConfig(rng_seed=1, max_ticks=1, visualize=False,
                    db_path="/tmp/test_wire.duckdb", cities_per_civ=1)
    from simulation.model import CivModel
    from agents.city import CityAgent
    model = CivModel(cfg)
    for agent in model.agents:
        if isinstance(agent, CityAgent):
            assert agent._pending_action is None
            break

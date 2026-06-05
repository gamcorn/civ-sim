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


def test_dispatch_sets_pending_action_for_all_cities():
    import asyncio
    cfg = SimConfig(rng_seed=5, max_ticks=1, visualize=False,
                    db_path="/tmp/test_dispatch.duckdb", cities_per_civ=2)
    from simulation.model import CivModel
    from agents.city import CityAgent
    model = CivModel(cfg)
    asyncio.run(model._dispatch_decisions())
    for agent in model.agents:
        if isinstance(agent, CityAgent):
            assert agent._pending_action is not None, \
                f"City {agent.unique_id} has no pending action after dispatch"


def test_dispatch_uses_civ_provider():
    """Provider's choose_actions_batch is called with the city's civ's provider."""
    import asyncio
    cfg = SimConfig(rng_seed=7, max_ticks=1, visualize=False,
                    db_path="/tmp/test_dispatch2.duckdb", cities_per_civ=1)
    from simulation.model import CivModel
    from agents.city import CityAgent
    model = CivModel(cfg)

    called_with = []

    async def fake_batch(cities):
        called_with.extend(cities)
        return ["gather"] * len(cities)

    for civ in model.civilizations:
        civ.provider.choose_actions_batch = fake_batch

    asyncio.run(model._dispatch_decisions())

    city_agents = [a for a in model.agents if isinstance(a, CityAgent)]
    assert len(called_with) == len(city_agents)

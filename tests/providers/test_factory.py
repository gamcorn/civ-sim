def test_factory_creates_rule_based():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.factory import create_provider
    from civ_sim.agents.providers.rule_based import RuleBasedProvider

    cfg = ProviderConfig(type="rule_based")
    provider = create_provider(cfg)
    assert isinstance(provider, RuleBasedProvider)


def test_factory_creates_openai_compatible():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.factory import create_provider
    from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider

    cfg = ProviderConfig(type="openai_compatible", model="test-model")
    provider = create_provider(cfg)
    assert isinstance(provider, OpenAICompatibleProvider)


def test_factory_creates_anthropic():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.factory import create_provider
    from civ_sim.agents.providers.anthropic_provider import AnthropicProvider

    cfg = ProviderConfig(type="anthropic", model="claude-haiku-4-5-20251001")
    provider = create_provider(cfg)
    assert isinstance(provider, AnthropicProvider)


def test_factory_creates_council_provider():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.factory import create_provider
    from civ_sim.agents.providers.council_provider import CouncilProvider

    cfg = ProviderConfig(
        type="council",
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    provider = create_provider(cfg)
    assert isinstance(provider, CouncilProvider)


def test_factory_raises_on_unknown_type():
    from civ_sim.config import ProviderConfig
    from civ_sim.agents.providers.factory import create_provider

    cfg = ProviderConfig(type="unknown_backend")
    try:
        create_provider(cfg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown_backend" in str(e)

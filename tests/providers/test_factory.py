def test_factory_creates_rule_based():
    from config import ProviderConfig
    from agents.providers.factory import create_provider
    from agents.providers.rule_based import RuleBasedProvider

    cfg = ProviderConfig(type="rule_based")
    provider = create_provider(cfg)
    assert isinstance(provider, RuleBasedProvider)


def test_factory_creates_openai_compatible():
    from config import ProviderConfig
    from agents.providers.factory import create_provider
    from agents.providers.openai_compat import OpenAICompatibleProvider

    cfg = ProviderConfig(type="openai_compatible", model="test-model")
    provider = create_provider(cfg)
    assert isinstance(provider, OpenAICompatibleProvider)


def test_factory_creates_anthropic():
    from config import ProviderConfig
    from agents.providers.factory import create_provider
    from agents.providers.anthropic_provider import AnthropicProvider

    cfg = ProviderConfig(type="anthropic", model="claude-haiku-4-5-20251001")
    provider = create_provider(cfg)
    assert isinstance(provider, AnthropicProvider)


def test_factory_raises_on_unknown_type():
    from config import ProviderConfig
    from agents.providers.factory import create_provider

    cfg = ProviderConfig(type="unknown_backend")
    try:
        create_provider(cfg)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "unknown_backend" in str(e)

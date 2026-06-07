import pytest


def test_factory_creates_council_provider():
    from config import ProviderConfig
    from agents.providers.factory import create_provider
    from agents.providers.council_provider import CouncilProvider

    cfg = ProviderConfig(
        type="council",
        model="test-model",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    provider = create_provider(cfg)
    assert isinstance(provider, CouncilProvider)


def test_factory_raises_on_unknown_type():
    from config import ProviderConfig
    from agents.providers.factory import create_provider

    cfg = ProviderConfig(type="does_not_exist")
    with pytest.raises(ValueError, match="Unknown provider type"):
        create_provider(cfg)

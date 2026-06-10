from __future__ import annotations
from typing import TYPE_CHECKING

from agents.providers.base import DecisionProvider

if TYPE_CHECKING:
    from config import ProviderConfig


def create_provider(config: "ProviderConfig") -> DecisionProvider:
    """Instantiate the correct DecisionProvider from a ProviderConfig."""
    if config.type == "rule_based":
        from agents.providers.rule_based import RuleBasedProvider
        return RuleBasedProvider()

    if config.type == "openai_compatible":
        from agents.providers.openai_compat import OpenAICompatibleProvider
        return OpenAICompatibleProvider(config)

    if config.type == "anthropic":
        from agents.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(config)

    if config.type == "council":
        from agents.providers.council_provider import CouncilProvider
        return CouncilProvider(config)

    raise ValueError(f"Unknown provider type: {config.type!r}. "
                     f"Choose: rule_based, openai_compatible, anthropic, council")

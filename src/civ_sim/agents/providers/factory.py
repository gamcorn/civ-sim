from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from civ_sim.agents.providers.base import DecisionProvider

if TYPE_CHECKING:
    from civ_sim.config import ProviderConfig

logger = logging.getLogger(__name__)


def create_provider(config: "ProviderConfig") -> DecisionProvider:
    """Instantiate the correct DecisionProvider from a ProviderConfig."""
    if config.type == "rule_based":
        from civ_sim.agents.providers.rule_based import RuleBasedProvider

        logger.info("Creating RuleBasedProvider")
        return RuleBasedProvider()

    if config.type == "openai_compatible":
        from civ_sim.agents.providers.openai_compat import OpenAICompatibleProvider

        logger.info("Creating OpenAICompatibleProvider model=%s", config.model)
        return OpenAICompatibleProvider(config)

    if config.type == "anthropic":
        from civ_sim.agents.providers.anthropic_provider import AnthropicProvider

        logger.info("Creating AnthropicProvider model=%s", config.model)
        return AnthropicProvider(config)

    if config.type == "council":
        from civ_sim.agents.providers.council_provider import CouncilProvider

        logger.info("Creating CouncilProvider model=%s", config.model)
        return CouncilProvider(config)

    raise ValueError(
        f"Unknown provider type: {config.type!r}. "
        f"Choose: rule_based, openai_compatible, anthropic, council"
    )

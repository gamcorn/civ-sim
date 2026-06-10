from __future__ import annotations
import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclasses.dataclass
class CulturalTraits:
    aggressiveness: float = 0.5
    trust: float = 0.5
    innovation: float = 0.5
    tribalism: float = 0.5
    risk_tolerance: float = 0.5

    def mutate(self, rng, sigma: float = 0.05) -> "CulturalTraits":
        """Return a new CulturalTraits with Gaussian noise on each trait."""
        def clamp(v):
            return max(0.0, min(1.0, v))

        return CulturalTraits(
            aggressiveness=clamp(self.aggressiveness + rng.gauss(0, sigma)),
            trust=clamp(self.trust + rng.gauss(0, sigma)),
            innovation=clamp(self.innovation + rng.gauss(0, sigma)),
            tribalism=clamp(self.tribalism + rng.gauss(0, sigma)),
            risk_tolerance=clamp(self.risk_tolerance + rng.gauss(0, sigma)),
        )

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class Civilization:
    """Aggregate state for one civilization.  Not a Mesa Agent — city agents hold the ref."""

    def __init__(self, civ_id: int, name: str, traits: CulturalTraits, provider=None):
        self.civ_id = civ_id
        self.name = name
        self.traits = traits

        if provider is None:
            from civ_sim.agents.providers.rule_based import RuleBasedProvider
            provider = RuleBasedProvider()
        self.provider = provider

        # Aggregate stats updated each tick by the model
        self.total_pop: int = 0
        self.total_military: int = 0
        self.city_count: int = 0
        self.tech_level: int = 0
        self.discovered_techs: set[str] = set()
        self.alive: bool = True

        # Council emergency tracking — updated by CouncilProvider after each directive
        self._pop_at_last_directive: int = 0
        self._techs_at_last_directive: int = 0
        self._city_count_at_last_directive: int = 0

    def update_aggregates(self, cities: list) -> None:
        self.total_pop = sum(c.population for c in cities)
        self.total_military = sum(c.military for c in cities)
        self.city_count = len(cities)
        self.alive = self.total_pop > 0

    def __repr__(self) -> str:
        return (
            f"Civ({self.name} pop={self.total_pop} mil={self.total_military} "
            f"tech={self.tech_level})"
        )

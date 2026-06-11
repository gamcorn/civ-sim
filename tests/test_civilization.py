import random
from unittest.mock import MagicMock

from civ_sim.agents.civilization import Civilization, CulturalTraits


def test_cultural_traits_as_dict_has_all_keys():
    t = CulturalTraits()
    d = t.as_dict()
    assert set(d.keys()) == {
        "aggressiveness",
        "trust",
        "innovation",
        "tribalism",
        "risk_tolerance",
    }


def test_cultural_traits_mutate_returns_new_instance():
    t = CulturalTraits(
        aggressiveness=0.5, trust=0.5, innovation=0.5, tribalism=0.5, risk_tolerance=0.5
    )
    rng = random.Random(42)
    mutated = t.mutate(rng, sigma=0.05)
    assert mutated is not t
    assert isinstance(mutated, CulturalTraits)


def test_cultural_traits_mutate_values_are_clamped():
    t = CulturalTraits(
        aggressiveness=0.99,
        trust=0.01,
        innovation=0.5,
        tribalism=0.5,
        risk_tolerance=0.5,
    )
    rng = random.Random(0)
    for _ in range(50):
        m = t.mutate(rng, sigma=0.5)
        for v in m.as_dict().values():
            assert 0.0 <= v <= 1.0


def test_cultural_traits_mutate_introduces_variation():
    t = CulturalTraits(aggressiveness=0.5)
    rng = random.Random(1)
    results = {t.mutate(rng).aggressiveness for _ in range(20)}
    assert len(results) > 1  # not all identical


def test_civilization_update_aggregates_sums_correctly():
    civ = Civilization(civ_id=0, name="Test", traits=CulturalTraits())
    c1 = MagicMock()
    c1.population = 100
    c1.military = 10
    c2 = MagicMock()
    c2.population = 200
    c2.military = 20
    civ.update_aggregates([c1, c2])
    assert civ.total_pop == 300
    assert civ.total_military == 30


def test_civilization_alive_is_false_when_pop_is_zero():
    civ = Civilization(civ_id=0, name="Dead", traits=CulturalTraits())
    c = MagicMock()
    c.population = 0
    c.military = 0
    civ.update_aggregates([c])
    assert civ.alive is False


def test_civilization_alive_is_true_when_pop_positive():
    civ = Civilization(civ_id=0, name="Alive", traits=CulturalTraits())
    c = MagicMock()
    c.population = 50
    c.military = 5
    civ.update_aggregates([c])
    assert civ.alive is True


def test_civilization_repr_contains_name():
    civ = Civilization(civ_id=0, name="Alpha", traits=CulturalTraits())
    assert "Alpha" in repr(civ)

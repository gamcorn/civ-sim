"""Tests for the rich-map visualisation additions.

Coverage:
  - CivModel._attack_events buffer (model.py + city.py)
  - Renderer pure helpers: _city_fill_radius, _city_wall_linewidth, _compute_threat
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from civ_sim.agents.city import CityAgent

# ---------------------------------------------------------------------------
# Helpers shared across groups
# ---------------------------------------------------------------------------


def _get_cities(model) -> list[CityAgent]:
    return [a for a in model.agents if isinstance(a, CityAgent)]


def _make_fake_city(x: int, y: int, military: int) -> SimpleNamespace:
    """Minimal city-like object for threat heatmap tests."""
    return SimpleNamespace(x=x, y=y, military=military)


# ===========================================================================
# Group 1 — CivModel._attack_events buffer
# ===========================================================================


def test_model_initializes_attack_events_as_empty_list(mini_model):
    """Model must expose _attack_events as an empty list at startup."""
    assert hasattr(
        mini_model, "_attack_events"
    ), "CivModel must have a _attack_events attribute"
    assert (
        mini_model._attack_events == []
    ), "_attack_events must be initialised as an empty list"


def test_attack_event_recorded_when_attack_resolves(mini_model):
    """A successful _do_attack() call must append one entry to model._attack_events."""
    cities = _get_cities(mini_model)
    assert len(cities) >= 2, "mini_model must have 2 cities"

    attacker = cities[0]
    defender = next(c for c in cities if c.civ.civ_id != attacker.civ.civ_id)

    # Place defender within attack range
    defender.x = attacker.x + 5
    defender.y = attacker.y
    defender.cell = mini_model.grid.cell(defender.x, defender.y)

    attacker.military = 1000
    defender.military = 1

    events_before = len(mini_model._attack_events)

    with patch.object(mini_model.random, "random", return_value=0.0):
        attacker._do_attack()

    assert (
        len(mini_model._attack_events) == events_before + 1
    ), "One attack event must be appended after a resolved attack"

    ev = mini_model._attack_events[-1]
    assert ev[0] == attacker.x, "event[0] must be attacker x"
    assert ev[1] == attacker.y, "event[1] must be attacker y"
    assert ev[2] == defender.x, "event[2] must be target x"
    assert ev[3] == defender.y, "event[3] must be target y"
    assert ev[4] == attacker.civ.civ_id, "event[4] must be attacker civ_id"


def test_attack_events_cleared_at_start_of_each_step(mini_model):
    """_attack_events must be empty at the start of each model.step() call."""
    # Manually plant a stale event
    mini_model._attack_events.append((0, 0, 1, 1, 0))

    # step() should clear the buffer before activating agents
    # We verify by running a step and checking the buffer is empty afterwards
    # (any new attacks from this step are fine, but the stale one must be gone)
    stale_entry = (0, 0, 1, 1, 0)
    mini_model.step()

    assert (
        stale_entry not in mini_model._attack_events
    ), "Stale attack events from a previous tick must be cleared at step start"


# ===========================================================================
# Group 2 — Renderer pure helper: _city_fill_radius
# ===========================================================================


def test_city_fill_radius_minimum_for_zero_population():
    """Radius must be at least 0.3 data units even for an empty city."""
    from civ_sim.visualization.renderer import _city_fill_radius

    r = _city_fill_radius(0)
    assert r >= 0.3, f"Expected radius >= 0.3 for pop=0, got {r}"


def test_city_fill_radius_grows_with_population():
    """A more populous city must have a larger radius."""
    from civ_sim.visualization.renderer import _city_fill_radius

    r_small = _city_fill_radius(50)
    r_large = _city_fill_radius(200)
    assert (
        r_large > r_small
    ), f"Radius should grow with population; got {r_small} (pop=50), {r_large} (pop=200)"


def test_city_fill_radius_reference_value():
    """At pop=200 (reference population) radius should equal base + full scale."""
    from civ_sim.visualization.renderer import _city_fill_radius

    # Formula: 0.3 + 1.2 * sqrt(pop / 200) → at pop=200: 0.3 + 1.2 = 1.5
    r = _city_fill_radius(200)
    assert abs(r - 1.5) < 1e-6, f"Expected 1.5 for pop=200, got {r}"


# ===========================================================================
# Group 3 — Renderer pure helper: _city_wall_linewidth
# ===========================================================================


def test_city_wall_linewidth_minimum_for_zero_military():
    """Linewidth must be at least 0.5 even for an undefended city."""
    from civ_sim.visualization.renderer import _city_wall_linewidth

    lw = _city_wall_linewidth(0)
    assert lw >= 0.5, f"Expected linewidth >= 0.5 for military=0, got {lw}"


def test_city_wall_linewidth_maximum_capped():
    """Linewidth must be capped at 4.0 regardless of military size."""
    from civ_sim.visualization.renderer import _city_wall_linewidth

    lw = _city_wall_linewidth(10_000)
    assert lw <= 4.0, f"Expected linewidth <= 4.0 for high military, got {lw}"


def test_city_wall_linewidth_grows_with_military():
    """A more fortified city must have a thicker wall ring."""
    from civ_sim.visualization.renderer import _city_wall_linewidth

    lw_weak = _city_wall_linewidth(10)
    lw_strong = _city_wall_linewidth(100)
    assert (
        lw_strong > lw_weak
    ), f"Linewidth should grow with military; got {lw_weak} (mil=10), {lw_strong} (mil=100)"


# ===========================================================================
# Group 4 — Renderer pure helper: _compute_threat
# ===========================================================================


def test_compute_threat_all_zeros_when_no_cities():
    """An empty city list must produce an all-zero threat array."""
    from civ_sim.visualization.renderer import _compute_threat

    threat = _compute_threat([], width=20, height=20)
    assert threat.shape == (20, 20), f"Expected shape (20, 20), got {threat.shape}"
    assert np.all(threat == 0.0), "Threat must be zero everywhere when no cities exist"


def test_compute_threat_nonzero_near_city():
    """Threat must be positive near a city with military > 0."""
    from civ_sim.visualization.renderer import _compute_threat

    city = _make_fake_city(x=10, y=10, military=50)
    threat = _compute_threat([city], width=20, height=20)
    assert threat[10, 10] > 0.0, "Threat must be nonzero at the city's own tile"


def test_compute_threat_peaks_at_city_location():
    """The maximum threat value must be at the city's location."""
    from civ_sim.visualization.renderer import _compute_threat

    city = _make_fake_city(x=5, y=8, military=100)
    threat = _compute_threat([city], width=20, height=20)
    max_idx = np.unravel_index(np.argmax(threat), threat.shape)
    assert max_idx == (
        5,
        8,
    ), f"Threat should peak at city location (5, 8), peaked at {max_idx}"


def test_compute_threat_decays_with_distance():
    """Threat must be strictly lower further from the city."""
    from civ_sim.visualization.renderer import _compute_threat

    city = _make_fake_city(x=10, y=10, military=100)
    threat = _compute_threat([city], width=30, height=30)
    # Threat at distance 1 vs distance 5 from city
    t_near = threat[10, 11]  # 1 step away
    t_far = threat[10, 15]  # 5 steps away
    assert (
        t_near > t_far
    ), f"Threat should decay with distance; near={t_near:.4f}, far={t_far:.4f}"


def test_compute_threat_zero_military_contributes_nothing():
    """A city with military=0 should not raise the threat level."""
    from civ_sim.visualization.renderer import _compute_threat

    city = _make_fake_city(x=5, y=5, military=0)
    threat = _compute_threat([city], width=20, height=20)
    assert np.all(
        threat == 0.0
    ), "A city with military=0 should not contribute to threat"

# tests/test_council_integration.py

from civ_sim.agents.decisions import ALL_ACTIONS


def test_council_provider_issues_directive_on_first_step():
    """First model step: CouncilProvider runs council, directive is issued and logged."""
    from civ_sim.agents.providers.council_provider import (
        CouncilProvider,
        StrategicDirective,
    )
    from civ_sim.config import ProviderConfig, SimConfig
    from civ_sim.simulation.model import CivModel

    cfg = SimConfig(
        width=20,
        height=20,
        num_civs=2,
        cities_per_civ=1,
        max_ticks=5,
        rng_seed=1,
        db_path=":memory:",
        visualize=False,
        civ_providers=[
            ProviderConfig(
                type="council", model="test-model", directive_period=5, max_rounds=1
            ),
            ProviderConfig(type="rule_based"),
        ],
    )
    model = CivModel(cfg)
    civ0 = model.civilizations[0]
    civ0_provider = civ0.provider
    assert isinstance(civ0_provider, CouncilProvider)

    expected_directive = StrategicDirective(
        era_goal="Expand and research",
        action_weights={a: 0.0 for a in ALL_ACTIONS} | {"expand": 0.4, "research": 0.3},
        reasoning="Territory and tech win the long game",
        issued_at_tick=1,
        valid_for_ticks=5,
        emergency=False,
    )

    async def mock_run_council(civ, cities, tick):  # noqa: E501
        civ0_provider._directive = expected_directive
        civ0_provider._last_council_tick = tick
        civ._pop_at_last_directive = civ.total_pop
        civ._techs_at_last_directive = len(civ.discovered_techs)
        civ._city_count_at_last_directive = len(cities)
        cities[0].model.logger.log_directive(tick, civ.civ_id, expected_directive)

    civ0_provider._run_council = mock_run_council

    model.step()

    assert civ0_provider._directive is not None
    assert civ0_provider._directive.era_goal == "Expand and research"
    assert civ0_provider._directive.action_weights["expand"] == 0.4
    assert civ0_provider._last_council_tick == 1

    # Directive was logged to DuckDB
    rows = model.logger._con.execute("SELECT * FROM directives").fetchall()
    assert len(rows) == 1

    model.logger.close()

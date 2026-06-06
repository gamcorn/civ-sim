from visualization.terminal_renderer import _tile_char, _city_char


def test_tile_char_empty():
    assert _tile_char(0.0, 100.0) == "·"


def test_tile_char_bucket0_boundary():
    # 19.9 / 100.0 = 19.9% — still bucket 0
    assert _tile_char(19.9, 100.0) == "·"


def test_tile_char_bucket1_boundary():
    # 20.0 / 100.0 = 20.0% — first value in bucket 1
    assert _tile_char(20.0, 100.0) == "░"


def test_tile_char_quarter():
    assert _tile_char(30.0, 100.0) == "░"


def test_tile_char_mid():
    assert _tile_char(50.0, 100.0) == "▒"


def test_tile_char_rich():
    assert _tile_char(70.0, 100.0) == "▓"


def test_tile_char_full():
    assert _tile_char(100.0, 100.0) == "█"


def test_tile_char_zero_max():
    # max_food=0 must not raise ZeroDivisionError
    assert _tile_char(0.0, 0.0) == "·"


def test_tile_char_overshoot():
    # food > max_food — min(..., 4) guard ensures "█" not IndexError
    assert _tile_char(150.0, 100.0) == "█"


def test_city_char_civ0_small():
    assert _city_char(0, 50.0) == "◦"


def test_city_char_civ0_large():
    assert _city_char(0, 150.0) == "●"


def test_city_char_civ0_exactly_threshold():
    # population == 100 is "large"
    assert _city_char(0, 100.0) == "●"


def test_city_char_civ1_small():
    assert _city_char(1, 0.0) == "◇"


def test_city_char_civ1_large():
    assert _city_char(1, 200.0) == "◆"


def test_city_char_unknown_civ():
    # Falls back to civ-0 glyphs — must not raise
    result = _city_char(99, 50.0)
    assert isinstance(result, str) and len(result) == 1


def test_terminal_renderer_update_no_crash():
    """update() must not raise on a real model tick."""
    import io
    import sys
    from unittest.mock import patch

    from config import SimConfig
    from simulation.model import CivModel
    from visualization.terminal_renderer import TerminalRenderer

    cfg = SimConfig(width=20, height=15, cities_per_civ=1, max_ticks=2, visualize=False)
    model = CivModel(cfg)
    captured = io.StringIO()
    with patch("sys.stdout", captured):
        renderer = TerminalRenderer(model)
        model.step()
        renderer.update(model)
    # Any output to stdout is acceptable; just no exception

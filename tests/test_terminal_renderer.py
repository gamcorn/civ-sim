from visualization.terminal_renderer import _tile_char


def test_tile_char_empty():
    assert _tile_char(0.0, 100.0) == "·"


def test_tile_char_low():
    assert _tile_char(15.0, 100.0) == "·"


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

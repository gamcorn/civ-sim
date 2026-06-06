from visualization.terminal_renderer import _tile_char


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

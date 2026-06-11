from civ_sim.replay import _apply_key, _nearest_idx


def test_nearest_idx_exact():
    assert _nearest_idx([10, 20, 30], 20) == 1


def test_nearest_idx_rounds_up():
    assert _nearest_idx([10, 20, 30], 15) == 1


def test_nearest_idx_before_start():
    assert _nearest_idx([10, 20, 30], 0) == 0


def test_nearest_idx_after_end():
    assert _nearest_idx([10, 20, 30], 999) == 2


def test_apply_key_space_toggles_pause():
    state = {"idx": 0, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("space", state, 3)
    assert state["paused"] is True
    _apply_key("space", state, 3)
    assert state["paused"] is False


def test_apply_key_plus_doubles_speed():
    state = {"idx": 0, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("+", state, 3)
    assert state["speed"] == 2.0


def test_apply_key_minus_halves_speed():
    state = {"idx": 0, "paused": False, "speed": 2.0, "quit": False}
    _apply_key("-", state, 3)
    assert state["speed"] == 1.0


def test_apply_key_speed_clamped():
    state = {"idx": 0, "paused": False, "speed": 32.0, "quit": False}
    _apply_key("+", state, 3)
    assert state["speed"] == 32.0

    state["speed"] = 0.125
    _apply_key("-", state, 3)
    assert state["speed"] == 0.125


def test_apply_key_right_advances():
    state = {"idx": 0, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("right", state, 10)
    assert state["idx"] == 9


def test_apply_key_right_clamped_at_last():
    state = {"idx": 5, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("right", state, 6)
    assert state["idx"] == 5


def test_apply_key_left_rewinds():
    state = {"idx": 15, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("left", state, 20)
    assert state["idx"] == 5


def test_apply_key_left_clamped_at_zero():
    state = {"idx": 3, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("left", state, 10)
    assert state["idx"] == 0


def test_apply_key_quit():
    state = {"idx": 0, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("q", state, 5)
    assert state["quit"] is True


def test_nearest_idx_empty_list():
    assert _nearest_idx([], 42) == 0


def test_apply_key_equals_alias_for_plus():
    state = {"idx": 0, "paused": False, "speed": 1.0, "quit": False}
    _apply_key("=", state, 3)
    assert state["speed"] == 2.0


def test_apply_key_unknown_key_is_noop():
    state = {"idx": 5, "paused": True, "speed": 2.0, "quit": False}
    _apply_key("x", state, 10)
    assert state == {"idx": 5, "paused": True, "speed": 2.0, "quit": False}

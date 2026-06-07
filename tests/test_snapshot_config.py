from config import SimConfig


def test_snapshot_interval_default():
    cfg = SimConfig()
    assert cfg.snapshot_interval == 0


def test_snapshot_interval_custom():
    cfg = SimConfig(snapshot_interval=50)
    assert cfg.snapshot_interval == 50

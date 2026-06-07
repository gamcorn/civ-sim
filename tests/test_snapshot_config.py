import sys
from unittest.mock import patch
from config import SimConfig


def test_snapshot_interval_default():
    cfg = SimConfig()
    assert cfg.snapshot_interval == 0


def test_snapshot_interval_custom():
    cfg = SimConfig(snapshot_interval=50)
    assert cfg.snapshot_interval == 50


def test_snapshot_interval_argparse_wiring():
    """--snapshot-interval CLI flag flows through to SimConfig."""
    import main as main_mod
    argv = ["prog", "--snapshot-interval", "25", "--no-visualize",
            "--ticks", "1", "--seed", "1"]
    with patch.object(sys, "argv", argv):
        args = main_mod.parse_args()
    assert args.snapshot_interval == 25


def test_snapshot_interval_argparse_default():
    """--snapshot-interval defaults to 0 when not supplied."""
    import main as main_mod
    argv = ["prog", "--no-visualize"]
    with patch.object(sys, "argv", argv):
        args = main_mod.parse_args()
    assert args.snapshot_interval == 0

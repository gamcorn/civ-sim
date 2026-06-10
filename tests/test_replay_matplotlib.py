import tempfile, os
import sys
import numpy as np
from unittest.mock import patch, MagicMock, ANY
from civ_sim.storage.snapshot import SnapshotWriter, SnapshotReader
from civ_sim.world.resources import ResourceType


def _write_one_snapshot(db_path):
    writer = SnapshotWriter(db_path, seed=1)
    grid = MagicMock()
    grid.ownership = np.zeros((20, 15), dtype=np.int8)
    food_layer = MagicMock()
    food_layer.data = np.zeros((20, 15), dtype=np.float32)
    grid.layers = {ResourceType.FOOD: food_layer}
    grid.config.resource_max = 100.0
    writer.write(tick=1, grid=grid, agents=[], civilizations=[])
    writer.close()


def test_replay_matplotlib_calls_plt_show():
    """Verify replay_matplotlib triggers plt.show when given a valid reader."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    os.unlink(db_path)
    try:
        _write_one_snapshot(db_path)
        reader = SnapshotReader(db_path)

        # Mock matplotlib components before importing replay to avoid import errors
        mock_plt = MagicMock()
        mock_anim = MagicMock()
        mock_renderer = MagicMock()

        with patch.dict(sys.modules, {
            "matplotlib": MagicMock(),
            "matplotlib.pyplot": mock_plt,
            "matplotlib.animation": MagicMock(FuncAnimation=mock_anim),
            "civ_sim.visualization.renderer": MagicMock(Renderer=mock_renderer),
        }):
            # Reload replay module to get mocked imports
            import importlib
            import civ_sim.replay as replay
            importlib.reload(replay)

            replay.plt.show = MagicMock()
            replay.replay_matplotlib(reader, from_tick=0, speed=1.0)
            replay.plt.show.assert_called_once_with(block=True)

        reader.close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)

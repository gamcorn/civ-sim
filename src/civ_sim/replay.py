"""Replay a completed simulation from a DuckDB snapshot file.

Usage:
    python replay.py results.duckdb
    python replay.py results.duckdb --renderer terminal --from-tick 500 --speed 4
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


def _nearest_idx(ticks: list[int], target: int) -> int:
    """Return index of first tick >= target; last index if all are smaller."""
    if not ticks:
        return 0
    for i, t in enumerate(ticks):
        if t >= target:
            return i
    return len(ticks) - 1


def _apply_key(key: str, state: dict, n_ticks: int) -> None:
    """Mutate playback state in response to a key event."""
    if key == "space":
        state["paused"] = not state["paused"]
    elif key in ("+", "="):
        state["speed"] = min(32.0, state["speed"] * 2.0)
    elif key == "-":
        state["speed"] = max(0.125, state["speed"] / 2.0)
    elif key == "right":
        state["idx"] = min(n_ticks - 1, state["idx"] + 10)
    elif key == "left":
        state["idx"] = max(0, state["idx"] - 10)
    elif key == "q":
        state["quit"] = True


# ---------------------------------------------------------------------------
# Terminal keyboard reader (daemon thread)
# ---------------------------------------------------------------------------


def _read_keys_raw(key_q: queue.Queue) -> None:
    """Read raw stdin bytes and push decoded key names into key_q."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[A":
                    key_q.put("up")
                elif seq == "[B":
                    key_q.put("down")
                elif seq == "[C":
                    key_q.put("right")
                elif seq == "[D":
                    key_q.put("left")
                else:
                    key_q.put("esc")
            elif ch == " ":
                key_q.put("space")
            elif ch in ("+", "="):
                key_q.put("+")
            elif ch == "-":
                key_q.put("-")
            elif ch in ("q", "Q"):
                key_q.put("q")
                break
    except Exception:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Terminal replay path
# ---------------------------------------------------------------------------


def replay_terminal(reader, from_tick: int, speed: float) -> None:
    import termios

    from civ_sim.visualization.terminal_renderer import TerminalRenderer

    ticks = reader.ticks()
    if not ticks:
        print("No snapshots found in this database.", file=sys.stderr)
        return

    # Save original terminal state before setting raw mode
    fd = sys.stdin.fileno()
    try:
        _saved_term = termios.tcgetattr(fd)
    except Exception:
        _saved_term = None

    state = {
        "idx": _nearest_idx(ticks, from_tick),
        "paused": False,
        "speed": speed,
        "quit": False,
    }

    frame = reader.load(ticks[state["idx"]])
    renderer = TerminalRenderer(frame)

    key_q: queue.Queue = queue.Queue()
    import threading

    t = threading.Thread(target=_read_keys_raw, args=(key_q,), daemon=True)
    t.start()

    try:
        while not state["quit"]:
            while True:
                try:
                    key = key_q.get_nowait()
                except queue.Empty:
                    break
                _apply_key(key, state, len(ticks))

            if state["quit"]:
                break

            if not state["paused"]:
                frame = reader.load(ticks[state["idx"]])
                renderer.update(frame)
                state["idx"] = min(len(ticks) - 1, state["idx"] + 1)
                if state["idx"] == len(ticks) - 1:
                    state["paused"] = True

            time.sleep(1.0 / max(state["speed"], 0.125))
    finally:
        if _saved_term is not None:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, _saved_term)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Imports for matplotlib path (Task 8 will replace the stub above)
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.artist import Artist

    from civ_sim.visualization.renderer import Renderer

    _MPL_AVAILABLE = True
except Exception:
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Matplotlib replay path (implemented in Task 8)
# ---------------------------------------------------------------------------


def replay_matplotlib(reader, from_tick: int, speed: float) -> None:
    if not _MPL_AVAILABLE:
        print("matplotlib is not available. Use --renderer terminal.", file=sys.stderr)
        return

    ticks = reader.ticks()
    if not ticks:
        print("No snapshots found in this database.", file=sys.stderr)
        return

    state = {
        "idx": _nearest_idx(ticks, from_tick),
        "paused": False,
        "speed": speed,
        "quit": False,
    }

    first_frame = reader.load(ticks[state["idx"]])
    renderer = Renderer(first_frame)

    def _on_key(event) -> None:
        key_map = {
            " ": "space",
            "+": "+",
            "=": "+",
            "-": "-",
            "right": "right",
            "left": "left",
            "q": "q",
        }
        k = key_map.get(event.key)
        if k:
            _apply_key(k, state, len(ticks))
        if state["quit"]:
            plt.close("all")

    renderer.fig.canvas.mpl_connect("key_press_event", _on_key)

    def _animate(_frame_num) -> list[Artist]:
        if state["quit"] or state["paused"]:
            return []
        frame = reader.load(ticks[state["idx"]])
        renderer.update(frame)
        state["idx"] = min(len(ticks) - 1, state["idx"] + 1)
        if state["idx"] == len(ticks) - 1:
            state["paused"] = True
        return []

    interval_ms = int(1000.0 / max(state["speed"], 0.125))
    _anim = FuncAnimation(
        renderer.fig,
        _animate,
        interval=interval_ms,
        cache_frame_data=False,
    )

    plt.show(block=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Replay a civ-sim DuckDB snapshot")
    p.add_argument("db_path", help="Path to .duckdb file with snapshots table")
    p.add_argument(
        "--renderer", choices=["terminal", "matplotlib", "auto"], default="auto"
    )
    p.add_argument(
        "--from-tick",
        type=int,
        default=0,
        metavar="N",
        help="Start near this tick (default: beginning)",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed in frames/sec (default: 1.0)",
    )
    args = p.parse_args()

    from civ_sim.storage.snapshot import SnapshotReader

    reader = SnapshotReader(args.db_path)

    renderer = args.renderer
    if renderer == "auto":
        import os

        renderer = "matplotlib" if os.environ.get("DISPLAY") else "terminal"

    try:
        if renderer == "terminal":
            replay_terminal(reader, args.from_tick, args.speed)
        else:
            replay_matplotlib(reader, args.from_tick, args.speed)
    finally:
        reader.close()


if __name__ == "__main__":
    main()

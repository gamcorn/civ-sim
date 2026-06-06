"""Terminal-based renderer using ANSI escape codes — no display server required.

Works over SSH and in headless environments. Redraws state in-place each tick
using cursor positioning to minimise flicker.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulation.model import CivModel

# ANSI helpers
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"

_FG    = {-1: "\033[32m",  0: "\033[34m",  1: "\033[31m"}   # green / blue / red
_BRITE = {-1: "\033[92m",  0: "\033[94m",  1: "\033[91m"}   # bright variants

_GRADIENT = ("·", "░", "▒", "▓", "█")


def _tile_char(food: float, max_food: float) -> str:
    """Return a block character encoding food level as a 5-step gradient."""
    frac = food / max(float(max_food), 1.0)
    idx = min(int(frac * 5), 4)
    return _GRADIENT[idx]


_CITY_GLYPHS: dict[int, tuple[str, str]] = {
    0: ("◦", "●"),  # (small, large)
    1: ("◇", "◆"),
}
_CITY_THRESHOLD: float = 100.0


def _city_char(civ_id: int, population: float) -> str:
    """Return a glyph for a city: shape encodes civ, size encodes population."""
    small, large = _CITY_GLYPHS.get(civ_id, _CITY_GLYPHS[0])
    return large if population >= _CITY_THRESHOLD else small


# Characters used in the map
_TILE  = {-1: "·",  0: "░",  1: "░"}
_CITY  = "@"


def _bar(value: float, maximum: float, width: int = 10, char: str = "█") -> str:
    filled = int(round(width * max(0.0, min(value, maximum)) / max(maximum, 1)))
    return char * filled + "░" * (width - filled)


class TerminalRenderer:
    """Redraws simulation state each tick using ANSI escape codes."""

    def __init__(self, model: "CivModel"):
        try:
            term_cols, term_rows = os.get_terminal_size()
        except OSError:
            term_cols, term_rows = 120, 40

        # Reserve rows: header(1) + blank(1) + col-header(2) + civs + blank(1) + legend(1) + blank(1)
        reserved = 7 + len(model.civilizations)
        map_rows = max(8, term_rows - reserved)
        map_cols = min(term_cols - 2, model.config.width)

        self.scale_x = max(1, model.config.width  // map_cols)
        self.scale_y = max(1, model.config.height // map_rows)
        self.map_cols = model.config.width  // self.scale_x
        self.map_rows = model.config.height // self.scale_y

        # Clear once; subsequent updates home the cursor instead
        sys.stdout.write("\033[2J")
        sys.stdout.flush()

    # ------------------------------------------------------------------

    def update(self, model: "CivModel") -> None:
        from agents.city import CityAgent

        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────────
        lines.append(
            f"{_BOLD}Civilization Simulation{_RESET}  tick={_BOLD}{model.steps}{_RESET}"
            f"  running={model.running}"
        )
        lines.append("")

        # ── Stats table ───────────────────────────────────────────────────
        col = f"{'Name':<14}{'Pop':>8}{'Military':>10}{'Food':>8}{'Territory':>11}{'Techs':>7}  {'Actions (top 3)'}"
        lines.append(f"{_DIM}{col}{_RESET}")
        lines.append(_DIM + "─" * 72 + _RESET)

        for civ in model.civilizations:
            cities = [
                a for a in model.agents
                if isinstance(a, CityAgent) and a.civ.civ_id == civ.civ_id
            ]
            total_pop  = sum(c.population  for c in cities)
            total_mil  = sum(c.military    for c in cities)
            total_food = sum(c.food_stock  for c in cities)
            territory  = model.grid.territory_count(civ.civ_id)
            techs      = len(civ.discovered_techs)

            action_counts = Counter(c.last_action for c in cities if c.last_action)
            top_actions   = "  ".join(
                f"{act}×{n}" for act, n in action_counts.most_common(3)
            ) or "—"

            color  = _FG[civ.civ_id]
            status = "" if civ.alive else f"  {_DIM}(extinct){_RESET}"

            lines.append(
                f"{color}{_BOLD}{civ.name[:14]:<14}{_RESET}"
                f"{total_pop:>8.0f}"
                f"{total_mil:>10.0f}"
                f"{total_food:>8.0f}"
                f"{territory:>11}"
                f"{techs:>7}"
                f"  {top_actions}"
                f"{status}"
            )

        lines.append("")

        # ── Map ───────────────────────────────────────────────────────────
        from world.resources import ResourceType
        ownership = np.array(model.grid.ownership)
        food = np.asarray(model.grid.layers[ResourceType.FOOD].data)  # host copy, works for both numpy and cupy
        max_r = float(model.config.resource_max)

        # Index city positions for O(1) lookup — stores (civ_id, population)
        city_at: dict[tuple[int, int], tuple[int, float]] = {}
        for a in model.agents:
            if isinstance(a, CityAgent):
                city_at[(a.x, a.y)] = (a.civ.civ_id, a.population)

        for row in range(self.map_rows):
            chars: list[str] = []
            for col in range(self.map_cols):
                # Grid coordinates for the centre of this block
                gx = min(col * self.scale_x + self.scale_x // 2, model.config.width  - 1)
                gy = min(row * self.scale_y + self.scale_y // 2, model.config.height - 1)

                # Check whether any city falls in this block
                city_info: tuple[int, float] | None = None
                for dy in range(self.scale_y):
                    for dx in range(self.scale_x):
                        cx = col * self.scale_x + dx
                        cy = row * self.scale_y + dy
                        if (cx, cy) in city_at:
                            city_info = city_at[(cx, cy)]
                            break
                    if city_info is not None:
                        break

                owner = int(ownership[gx, gy])
                if city_info is not None:
                    city_civ, city_pop = city_info
                    chars.append(
                        f"{_BRITE[city_civ]}{_BOLD}{_city_char(city_civ, city_pop)}{_RESET}"
                    )
                else:
                    chars.append(
                        f"{_FG[owner]}{_tile_char(food[gx, gy], max_r)}{_RESET}"
                    )

            lines.append("".join(chars))

        # ── Legend ────────────────────────────────────────────────────────
        lines.append("")
        civ_legend = "   ".join(
            f"{_FG[c.civ_id]}{_TILE[c.civ_id]}{_RESET}/{_BRITE[c.civ_id]}{_CITY}{_RESET} {c.name}"
            for c in model.civilizations
        )
        lines.append(
            f"  {_FG[-1]}{_TILE[-1]}{_RESET} unclaimed   {civ_legend}"
        )

        # ── Population sparklines ─────────────────────────────────────────
        h = model.history
        if h["tick"]:
            lines.append("")
            max_pop = max(
                (max(h[f"pop_{i}"], default=1) for i in range(len(model.civilizations))),
                default=1,
            )
            for i, civ in enumerate(model.civilizations):
                recent = h[f"pop_{i}"][-40:]
                bar_w  = min(40, len(recent))
                bar    = _bar(recent[-1] if recent else 0, max_pop, bar_w)
                color  = _FG[civ.civ_id]
                lines.append(
                    f"  {color}{civ.name[:8]:<8}{_RESET} pop {color}{bar}{_RESET}"
                    f" {recent[-1] if recent else 0:.0f}"
                )

        # ── Flush ─────────────────────────────────────────────────────────
        sys.stdout.write("\033[H")          # home cursor — no blank-screen flash
        sys.stdout.write("\n".join(lines))
        sys.stdout.write("\033[J")          # erase below (clears leftover lines)
        sys.stdout.flush()

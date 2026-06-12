"""Terminal-based renderer using ANSI escape codes — no display server required.

Works over SSH and in headless environments. Redraws state in-place each tick
using cursor positioning to minimise flicker.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from civ_sim.simulation.model import CivModel

logger = logging.getLogger(__name__)

# ANSI helpers
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

_FG = {-1: "\033[32m", 0: "\033[34m", 1: "\033[31m"}  # green / blue / red
_BRITE = {-1: "\033[92m", 0: "\033[94m", 1: "\033[91m"}  # bright variants
_ORANGE = "\033[33m"  # famine indicator
_YELLOW = "\033[93m"  # disease indicator
_GREY = "\033[37m"  # minerals

_GRADIENT = ("·", "░", "▒", "▓", "█")
_SPARK = "▁▂▃▄▅▆▇█"


def _tile_char(food: float, max_food: float) -> str:
    frac = food / max(float(max_food), 1.0)
    return _GRADIENT[min(int(frac * 5), 4)]


_CITY_GLYPHS: dict[int, tuple[str, str]] = {
    0: ("◦", "●"),
    1: ("◇", "◆"),
}
_CITY_THRESHOLD: float = 100.0


def _city_char(civ_id: int, population: float) -> str:
    small, large = _CITY_GLYPHS.get(civ_id, _CITY_GLYPHS[0])
    return large if population >= _CITY_THRESHOLD else small


def _sparkline(values: list[float], width: int = 24) -> str:
    """Unicode trend sparkline for the most recent `width` values."""
    if not values:
        return "─" * width
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo or 1.0
    chars = [_SPARK[min(7, int((v - lo) / span * 8))] for v in recent]
    pad = width - len(chars)
    return " " * pad + "".join(chars)


class TerminalRenderer:
    """Redraws simulation state each tick using ANSI escape codes."""

    def __init__(self, model: "CivModel"):
        try:
            term_cols, term_rows = os.get_terminal_size()
        except OSError:
            term_cols, term_rows = 120, 40

        n = len(model.civilizations)
        # Rows reserved outside the map:
        #   above: header(1) + blank(1) + col-header(2) + civs(n) + blank(1) = 5+n
        #   below: blank(1) + legend(1) + blank(1) + ts-header(1) + civ-rows(n) +
        #          blank(1) + res-header(1) + res-rows(3) + blank(1) +
        #          epi-header(1) + epi-rows(≤5) + blank(1) = 17+n
        reserved = (5 + n) + (17 + n) + 3
        map_rows = max(6, term_rows - reserved)
        map_cols = min(term_cols - 2, model.config.width)

        self.scale_x = max(1, model.config.width // map_cols)
        self.scale_y = max(1, model.config.height // map_rows)
        self.map_cols = model.config.width // self.scale_x
        self.map_rows = model.config.height // self.scale_y

        logger.info(
            "Renderer initialised: width=%d height=%d",
            model.config.width,
            model.config.height,
        )

        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

    # ------------------------------------------------------------------

    def update(self, model: "CivModel") -> None:
        lines: list[str] = []

        # ── Header ────────────────────────────────────────────────────────
        lines.append(
            f"{_BOLD}Civilization Simulation{_RESET}  tick={_BOLD}{model.steps}{_RESET}"
            f"  running={model.running}"
        )
        lines.append("")

        # ── Stats table ───────────────────────────────────────────────────
        col = (
            f"{'Name':<14}{'Pop':>8}{'Military':>10}{'Cities':>7}"
            f"{'Tiles':>7}{'FoodStk':>8}{'WoodStk':>8}{'MinStk':>8}{'Techs':>6}  {'Actions (top 3)'}"
        )
        lines.append(f"{_DIM}{col}{_RESET}")
        lines.append(_DIM + "─" * 96 + _RESET)

        for civ in model.civilizations:
            cities = [
                a
                for a in model.agents
                if hasattr(a, "civ") and a.civ.civ_id == civ.civ_id
            ]
            total_pop = sum(c.population for c in cities)
            total_mil = sum(c.military for c in cities)
            total_food = sum(c.food_stock for c in cities)
            total_wood = sum(getattr(c, "wood_stock", 0.0) for c in cities)
            total_min = sum(getattr(c, "mineral_stock", 0.0) for c in cities)
            territory = model.grid.territory_count(civ.civ_id)
            techs = len(civ.discovered_techs)
            n_cities = len(cities)

            has_disease = any(getattr(c, "_disease_hit_ticks", 0) > 0 for c in cities)
            has_famine = any(c.food_stock < 0.1 and c.population > 0 for c in cities)
            has_wood_short = any(
                getattr(c, "wood_stock", 1.0) < 1.0 and c.population > 0 for c in cities
            )
            has_min_short = any(getattr(c, "mineral_stock", 1.0) < 1.0 for c in cities)

            action_counts = Counter(c.last_action for c in cities if c.last_action)
            top_actions = (
                "  ".join(f"{act}×{cnt}" for act, cnt in action_counts.most_common(3))
                or "—"
            )

            color = _FG[civ.civ_id]
            tags = "" if civ.alive else f"  {_DIM}(extinct){_RESET}"
            if has_disease:
                tags += f"  {_YELLOW}⚕disease{_RESET}"
            if has_famine:
                tags += f"  {_ORANGE}⚠famine{_RESET}"
            if has_wood_short:
                tags += f"  {_ORANGE}⚠wood{_RESET}"
            if has_min_short:
                tags += f"  {_ORANGE}⚠min{_RESET}"

            lines.append(
                f"{color}{_BOLD}{civ.name[:14]:<14}{_RESET}"
                f"{total_pop:>8.0f}"
                f"{total_mil:>10.0f}"
                f"{n_cities:>7}"
                f"{territory:>7}"
                f"{total_food:>8.0f}"
                f"{total_wood:>8.0f}"
                f"{total_min:>8.0f}"
                f"{techs:>6}"
                f"  {top_actions}"
                f"{tags}"
            )

        lines.append("")

        # ── Map ───────────────────────────────────────────────────────────
        from civ_sim.world.resources import ResourceType

        ownership = np.array(model.grid.ownership)
        food = np.asarray(model.grid.layers[ResourceType.FOOD].data)
        max_r = float(model.config.resource_max)

        # (civ_id, population, is_disease, is_famine)
        city_at: dict[tuple[int, int], tuple[int, float, bool, bool]] = {}
        for a in model.agents:
            if hasattr(a, "civ"):
                city_at[(a.x, a.y)] = (
                    a.civ.civ_id,
                    a.population,
                    getattr(a, "_disease_hit_ticks", 0) > 0,
                    a.food_stock < 0.1 and a.population > 0,
                )

        for row in range(self.map_rows):
            chars: list[str] = []
            for col in range(self.map_cols):
                gx = min(col * self.scale_x + self.scale_x // 2, model.config.width - 1)
                gy = min(
                    row * self.scale_y + self.scale_y // 2, model.config.height - 1
                )

                city_info: tuple[int, float, bool, bool] | None = None
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
                    city_civ, city_pop, is_disease, is_famine = city_info
                    glyph = _city_char(city_civ, city_pop)
                    if is_disease:
                        c = _YELLOW
                    elif is_famine:
                        c = _ORANGE
                    else:
                        c = _BRITE[city_civ]
                    chars.append(f"{c}{_BOLD}{glyph}{_RESET}")
                else:
                    chars.append(
                        f"{_FG[owner]}{_tile_char(food[gx, gy], max_r)}{_RESET}"
                    )

            lines.append("".join(chars))

        # ── Legend ────────────────────────────────────────────────────────
        lines.append("")
        civ_legend = "   ".join(
            f"{_FG[c.civ_id]}█{_RESET} "
            f"{_BRITE[c.civ_id]}{_city_char(c.civ_id, 50)}{_RESET}/"
            f"{_BRITE[c.civ_id]}{_city_char(c.civ_id, 150)}{_RESET} {c.name}"
            for c in model.civilizations
        )
        lines.append(
            f"  {_FG[-1]}█{_RESET} unclaimed   {civ_legend}"
            f"   {_YELLOW}●{_RESET}=disease  {_ORANGE}●{_RESET}=famine"
            f"   tile: ·░▒▓█=food"
        )

        # ── Time-series sparklines ─────────────────────────────────────────
        h = model.history
        if h["tick"]:
            lines.append("")
            lines.append(
                f"{_DIM}  {'':8}  {'─── population ───':^26}  "
                f"{'─── military ───':^26}  "
                f"{'─ cities ─':^12}  "
                f"{'── food/territory ──':^26}{_RESET}"
            )
            for i, civ in enumerate(model.civilizations):
                color = _FG[civ.civ_id]
                pop_h = h.get(f"pop_{i}", [])
                mil_h = h.get(f"mil_{i}", [])
                cit_h = h.get(f"cities_{i}", [])
                food_h = h.get(f"food_civ_{i}", [])
                sp_pop = _sparkline(pop_h)
                sp_mil = _sparkline(mil_h)
                sp_cit = _sparkline(cit_h, width=12)
                sp_food = _sparkline(food_h)
                cur_pop = pop_h[-1] if pop_h else 0
                cur_mil = mil_h[-1] if mil_h else 0
                cur_cit = cit_h[-1] if cit_h else 0
                cur_food = food_h[-1] if food_h else 0
                lines.append(
                    f"  {color}{civ.name[:8]:<8}{_RESET}"
                    f"  {color}{sp_pop}{_RESET} {cur_pop:>6.0f}"
                    f"  {color}{sp_mil}{_RESET} {cur_mil:>6.0f}"
                    f"  {color}{sp_cit}{_RESET} {cur_cit:>3.0f}"
                    f"  {color}{sp_food}{_RESET} {cur_food:>6.0f}"
                )

            # ── Grid resource totals ───────────────────────────────────────
            lines.append("")
            lines.append(f"{_DIM}  Grid resource totals{_RESET}")
            for key, label, rc in [
                ("food_total", "food    ", "\033[32m"),
                ("minerals_total", "minerals", _GREY),
                ("wood_total", "wood    ", _ORANGE),
            ]:
                vals = h.get(key, [])
                sp = _sparkline(vals)
                cur = vals[-1] if vals else 0
                lines.append(f"  {rc}{label}{_RESET}  {rc}{sp}{_RESET}  {cur:>9.0f}")

        # ── Epidemic events ────────────────────────────────────────────────
        epi = getattr(model, "_epidemic_log", [])
        if epi:
            lines.append("")
            lines.append(f"{_DIM}  Epidemic events  (β = transmission rate){_RESET}")
            for tick, beta, deaths in epi[-5:]:
                if beta < 1.0:
                    sev, sc = "mild", _DIM
                elif beta < 2.0:
                    sev, sc = "severe", _YELLOW
                else:
                    sev, sc = "catastrophic", "\033[91m"
                lines.append(
                    f"  tick {tick:>4}  β={beta:.2f}  deaths={deaths:>6,}"
                    f"  {sc}[{sev}]{_RESET}"
                )

        # ── Flush ─────────────────────────────────────────────────────────
        sys.stdout.write("\033[H")
        sys.stdout.write("\n".join(lines))
        sys.stdout.write("\033[J")
        sys.stdout.flush()

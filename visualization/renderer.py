from __future__ import annotations
from typing import TYPE_CHECKING

import os
import numpy as np
import matplotlib
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":0"
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

if TYPE_CHECKING:
    from simulation.model import CivModel

# Territory map colors per civ_id (-1 = unclaimed)
_MAP_COLORS: dict[int, tuple] = {
    -1: (0.18, 0.42, 0.18),
     0: (0.20, 0.40, 0.85),
     1: (0.85, 0.25, 0.25),
     2: (0.25, 0.72, 0.30),
     3: (0.65, 0.25, 0.78),
}
# Line/scatter colors for up to 4 civs
_CIV_LINE_COLORS = ["#5599ff", "#ff5555", "#44cc55", "#cc44ff"]


class Renderer:
    """Live matplotlib dashboard updated each simulation tick."""

    def __init__(self, model: "CivModel"):
        self.model = model
        n = len(model.civilizations)

        self.fig = plt.figure(figsize=(24, 13))
        self.fig.patch.set_facecolor("#1a1a2e")

        # Layout: 3 rows × 3 cols; left column is the world map (spans all rows)
        gs = gridspec.GridSpec(
            3, 3, figure=self.fig,
            left=0.04, right=0.985, top=0.93, bottom=0.06,
            hspace=0.52, wspace=0.32,
            width_ratios=[1.5, 1, 1],
        )
        self.ax_map     = self.fig.add_subplot(gs[:, 0])   # world map
        self.ax_pop     = self.fig.add_subplot(gs[0, 1])   # population
        self.ax_mil     = self.fig.add_subplot(gs[1, 1])   # military
        self.ax_cities  = self.fig.add_subplot(gs[2, 1])   # city count
        self.ax_res     = self.fig.add_subplot(gs[0, 2])   # total grid resources
        self.ax_civfood = self.fig.add_subplot(gs[1, 2])   # per-civ food on owned territory
        self.ax_epi     = self.fig.add_subplot(gs[2, 2])   # epidemic log

        _all = [self.ax_map, self.ax_pop, self.ax_mil, self.ax_cities,
                self.ax_res, self.ax_civfood, self.ax_epi]
        for ax in _all:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white", labelsize=7)
            ax.xaxis.label.set_color("#aaa")
            ax.yaxis.label.set_color("#aaa")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        # ── World map ─────────────────────────────────────────────────
        rgb = self._build_rgb(model)
        self.im = self.ax_map.imshow(
            rgb.transpose(1, 0, 2), origin="lower",
            interpolation="nearest", aspect="equal",
        )
        self.ax_map.set_title("Territory  (brightness = food level)", fontsize=10)
        self.ax_map.set_xlabel("x", fontsize=8)
        self.ax_map.set_ylabel("y", fontsize=8)

        # City markers per civ
        self._scatters: dict[int, any] = {}
        for i, civ in enumerate(model.civilizations):
            col = _CIV_LINE_COLORS[i % len(_CIV_LINE_COLORS)]
            sc = self.ax_map.scatter([], [], s=[], c=col,
                                     edgecolors="white", linewidths=0.5,
                                     zorder=5, label=civ.name)
            self._scatters[i] = sc

        # Disease overlay (orange rings) — shown for 8 ticks after a hit
        self._sc_disease = self.ax_map.scatter(
            [], [], s=[], facecolors="none", edgecolors="#ff8800",
            linewidths=2.0, zorder=7, label="disease"
        )
        # Famine overlay (yellow rings)
        self._sc_famine = self.ax_map.scatter(
            [], [], s=[], facecolors="none", edgecolors="#ffee00",
            linewidths=2.0, zorder=7, label="famine"
        )
        self.ax_map.legend(loc="upper right", fontsize=7,
                           facecolor="#1a1a2e", labelcolor="white",
                           framealpha=0.7)

        # ── Population ────────────────────────────────────────────────
        self._pop_lines: dict[int, any] = {}
        self._epi_vlines: list = []   # vertical lines on pop chart for epidemics
        for i, civ in enumerate(model.civilizations):
            col = _CIV_LINE_COLORS[i % len(_CIV_LINE_COLORS)]
            ln, = self.ax_pop.plot([], [], color=col, label=civ.name, lw=1.5)
            self._pop_lines[i] = ln
        self.ax_pop.set_title("Population", fontsize=9)
        self.ax_pop.set_xlabel("tick", fontsize=7)
        self.ax_pop.set_ylabel("people", fontsize=7)
        self.ax_pop.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        # ── Military ──────────────────────────────────────────────────
        self._mil_lines: dict[int, any] = {}
        for i, civ in enumerate(model.civilizations):
            col = _CIV_LINE_COLORS[i % len(_CIV_LINE_COLORS)]
            ln, = self.ax_mil.plot([], [], color=col, label=civ.name, lw=1.5)
            self._mil_lines[i] = ln
        self.ax_mil.set_title("Military", fontsize=9)
        self.ax_mil.set_xlabel("tick", fontsize=7)
        self.ax_mil.set_ylabel("units", fontsize=7)
        self.ax_mil.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        # ── City count ────────────────────────────────────────────────
        self._city_lines: dict[int, any] = {}
        for i, civ in enumerate(model.civilizations):
            col = _CIV_LINE_COLORS[i % len(_CIV_LINE_COLORS)]
            ln, = self.ax_cities.plot([], [], color=col, label=civ.name, lw=1.5)
            self._city_lines[i] = ln
        self.ax_cities.set_title("City Count", fontsize=9)
        self.ax_cities.set_xlabel("tick", fontsize=7)
        self.ax_cities.set_ylabel("cities", fontsize=7)
        self.ax_cities.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        # ── Total grid resources ──────────────────────────────────────
        self._res_food,     = self.ax_res.plot([], [], color="#44bb44", label="food",     lw=1.2)
        self._res_minerals, = self.ax_res.plot([], [], color="#aaaaaa", label="minerals", lw=1.2)
        self._res_wood,     = self.ax_res.plot([], [], color="#aa7733", label="wood",     lw=1.2)
        self.ax_res.set_title("Total Grid Resources", fontsize=9)
        self.ax_res.set_xlabel("tick", fontsize=7)
        self.ax_res.set_ylabel("sum across grid", fontsize=7)
        self.ax_res.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        # ── Per-civ food on owned territory ──────────────────────────
        self._civfood_lines: dict[int, any] = {}
        for i, civ in enumerate(model.civilizations):
            col = _CIV_LINE_COLORS[i % len(_CIV_LINE_COLORS)]
            ln, = self.ax_civfood.plot([], [], color=col, label=civ.name, lw=1.5)
            self._civfood_lines[i] = ln
        self.ax_civfood.set_title("Food on Owned Territory", fontsize=9)
        self.ax_civfood.set_xlabel("tick", fontsize=7)
        self.ax_civfood.set_ylabel("food units", fontsize=7)
        self.ax_civfood.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white")

        # ── Epidemic log ──────────────────────────────────────────────
        # Each dot: x=tick, y=β; color encodes severity (green→red)
        self._epi_sc = self.ax_epi.scatter(
            [], [], s=[], c=[], cmap="RdYlGn_r", vmin=0.1, vmax=3.0,
            zorder=3, edgecolors="#555", linewidths=0.4,
        )
        self.ax_epi.set_title("Epidemic Events  (dot size ∝ β)", fontsize=9)
        self.ax_epi.set_xlabel("tick", fontsize=7)
        self.ax_epi.set_ylabel("β  (transmission rate)", fontsize=7)
        self.ax_epi.set_ylim(-0.1, 3.3)
        self.ax_epi.axhline(1.0, color="#555", lw=0.6, ls="--")
        self.ax_epi.axhline(2.0, color="#775", lw=0.6, ls="--")
        for y, label in [(0.08, "mild"), (0.36, "severe"), (0.69, "catastrophic")]:
            self.ax_epi.text(0.01, y, label, color="#888", fontsize=6,
                             transform=self.ax_epi.transAxes, va="bottom")
        # Colorbar for epidemic chart
        sm = ScalarMappable(cmap="RdYlGn_r", norm=Normalize(vmin=0.1, vmax=3.0))
        sm.set_array([])
        cb = self.fig.colorbar(sm, ax=self.ax_epi, pad=0.02, fraction=0.05)
        cb.ax.tick_params(colors="white", labelsize=6)
        cb.set_label("β", color="white", fontsize=7)

        self.title = self.fig.suptitle(
            "Civilization Simulation  tick=0", color="white", fontsize=13
        )
        plt.ion()
        plt.show()

    # ------------------------------------------------------------------

    def update(self, model: "CivModel") -> None:
        from agents.city import CityAgent

        # ── Map ───────────────────────────────────────────────────────
        self.im.set_data(self._build_rgb(model).transpose(1, 0, 2))

        all_cities = [a for a in model.agents if isinstance(a, CityAgent)]

        for i in range(len(model.civilizations)):
            cities = [c for c in all_cities if c.civ.civ_id == i]
            if cities:
                xs    = [c.x for c in cities]
                ys    = [c.y for c in cities]
                sizes = [max(25, c.population / 8) for c in cities]
                self._scatters[i].set_offsets(np.c_[xs, ys])
                self._scatters[i].set_sizes(sizes)
            else:
                self._scatters[i].set_offsets(np.empty((0, 2)))
                self._scatters[i].set_sizes([])

        def _apply_overlay(sc, cities):
            if cities:
                sc.set_offsets(np.c_[[c.x for c in cities], [c.y for c in cities]])
                sc.set_sizes([max(80, c.population / 4) for c in cities])
            else:
                sc.set_offsets(np.empty((0, 2)))
                sc.set_sizes([])

        _apply_overlay(self._sc_disease,
                       [c for c in all_cities if c._disease_hit_ticks > 0])
        _apply_overlay(self._sc_famine,
                       [c for c in all_cities if c.food_stock < 0.1 and c.population > 0])

        # ── Time-series charts ────────────────────────────────────────
        h = model.history
        if h["tick"]:
            ticks = h["tick"]
            n = len(model.civilizations)
            for i in range(n):
                self._pop_lines[i].set_data(ticks, h[f"pop_{i}"])
                self._mil_lines[i].set_data(ticks, h[f"mil_{i}"])
                self._city_lines[i].set_data(ticks, h[f"cities_{i}"])
                self._civfood_lines[i].set_data(ticks, h[f"food_civ_{i}"])

            self._res_food.set_data(ticks, h["food_total"])
            self._res_minerals.set_data(ticks, h["minerals_total"])
            self._res_wood.set_data(ticks, h["wood_total"])

            for ax in (self.ax_pop, self.ax_mil, self.ax_cities,
                       self.ax_res, self.ax_civfood):
                ax.relim()
                ax.autoscale_view()

        # ── Epidemic log ──────────────────────────────────────────────
        epi = model._epidemic_log
        if epi:
            et = [e[0] for e in epi]
            eb = [e[1] for e in epi]
            self._epi_sc.set_offsets(np.c_[et, eb])
            self._epi_sc.set_sizes([max(20, b * 35) for b in eb])
            self._epi_sc.set_array(np.array(eb))
            self.ax_epi.set_xlim(0, max(h["tick"]) if h["tick"] else 1)

        # Vertical epidemic markers on population chart (added incrementally)
        n_drawn = len(self._epi_vlines)
        for tick, beta in model._epidemic_log[n_drawn:]:
            alpha = min(0.75, 0.15 + beta / 3.5)
            color = plt.cm.RdYlGn_r(beta / 3.0)
            vl = self.ax_pop.axvline(tick, color=color, alpha=alpha, lw=0.8, ls=":")
            self._epi_vlines.append(vl)

        self.title.set_text(f"Civilization Simulation  tick={model.steps}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------

    def _build_rgb(self, model: "CivModel") -> np.ndarray:
        """(W, H, 3) float32: ownership color modulated by food brightness."""
        from world.resources import ResourceType
        food      = model.grid.layers[ResourceType.FOOD].data
        ownership = model.grid.ownership
        max_r     = model.config.resource_max

        rgb = np.zeros((model.config.width, model.config.height, 3), dtype=np.float32)
        for civ_id, color in _MAP_COLORS.items():
            mask       = ownership == civ_id
            brightness = np.where(mask, 0.35 + 0.65 * (food / max_r), 0.0)
            for c, base in enumerate(color):
                rgb[:, :, c] += mask * brightness * base
        return np.clip(rgb, 0, 1)

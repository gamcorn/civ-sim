from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # non-blocking interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation

if TYPE_CHECKING:
    from simulation.model import CivModel

# Civ colors: 0=blue, 1=red, -1=unclaimed (green-ish)
CIV_COLORS = {-1: (0.2, 0.5, 0.2), 0: (0.2, 0.4, 0.85), 1: (0.85, 0.25, 0.25)}


class Renderer:
    """Live matplotlib visualization updated each simulation tick."""

    def __init__(self, model: "CivModel"):
        self.model = model
        cfg = model.config

        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor("#1a1a2e")

        gs = self.fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_pop = self.fig.add_subplot(gs[0, 1])
        self.ax_mil = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_map, self.ax_pop, self.ax_mil):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")

        # Territory map image
        rgb = self._build_rgb(model)
        self.im = self.ax_map.imshow(
            rgb.transpose(1, 0, 2),
            origin="lower", interpolation="nearest",
            aspect="equal",
        )
        self.ax_map.set_title("Territory", color="white", fontsize=11)
        self.ax_map.set_xlabel("x", color="#aaa")
        self.ax_map.set_ylabel("y", color="#aaa")

        # City scatter plots
        self._scatters = {}
        colors = ["#5599ff", "#ff5555"]
        for i, civ in enumerate(model.civilizations):
            sc = self.ax_map.scatter([], [], s=[], c=colors[i],
                                     edgecolors="white", linewidths=0.4,
                                     zorder=5, label=civ.name)
            self._scatters[i] = sc
        self.ax_map.legend(loc="upper right", fontsize=8,
                           facecolor="#1a1a2e", labelcolor="white")

        # Population chart
        self._pop_lines = {}
        self._mil_lines = {}
        for i, civ in enumerate(model.civilizations):
            line_p, = self.ax_pop.plot([], [], color=colors[i], label=civ.name, lw=1.5)
            line_m, = self.ax_mil.plot([], [], color=colors[i], label=civ.name, lw=1.5)
            self._pop_lines[i] = line_p
            self._mil_lines[i] = line_m

        for ax, title, ylabel in [
            (self.ax_pop, "Population", "people"),
            (self.ax_mil, "Military", "units"),
        ]:
            ax.set_title(title, color="white", fontsize=10)
            ax.set_xlabel("tick", color="#aaa", fontsize=8)
            ax.set_ylabel(ylabel, color="#aaa", fontsize=8)
            ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

        self.title = self.fig.suptitle("Civilization Simulation  tick=0",
                                       color="white", fontsize=13)
        plt.ion()
        plt.show()

    # ------------------------------------------------------------------

    def update(self, model: "CivModel") -> None:
        # Territory map
        rgb = self._build_rgb(model)
        self.im.set_data(rgb.transpose(1, 0, 2))

        # City markers
        for i, civ in enumerate(model.civilizations):
            cities = [a for a in model.agents
                      if hasattr(a, 'civ') and a.civ.civ_id == i]
            if cities:
                xs = [c.x for c in cities]
                ys = [c.y for c in cities]
                sizes = [max(20, c.population / 10) for c in cities]
                self._scatters[i].set_offsets(np.c_[xs, ys])
                self._scatters[i].set_sizes(sizes)
            else:
                self._scatters[i].set_offsets(np.empty((0, 2)))
                self._scatters[i].set_sizes([])

        # Line charts
        h = model.history
        if h["tick"]:
            ticks = h["tick"]
            for i in range(len(model.civilizations)):
                self._pop_lines[i].set_data(ticks, h[f"pop_{i}"])
                self._mil_lines[i].set_data(ticks, h[f"mil_{i}"])
            self.ax_pop.relim(); self.ax_pop.autoscale_view()
            self.ax_mil.relim(); self.ax_mil.autoscale_view()

        self.title.set_text(f"Civilization Simulation  tick={model.steps}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # ------------------------------------------------------------------

    def _build_rgb(self, model: "CivModel") -> np.ndarray:
        """Build (W, H, 3) RGB array: ownership color modulated by food level."""
        from world.resources import ResourceType
        food = model.grid.layers[ResourceType.FOOD].data  # (W, H)
        ownership = model.grid.ownership                  # (W, H)
        max_r = model.config.resource_max

        rgb = np.zeros((model.config.width, model.config.height, 3), dtype=np.float32)
        for civ_id, color in CIV_COLORS.items():
            mask = ownership == civ_id
            brightness = np.where(mask, 0.4 + 0.6 * (food / max_r), 0.0)
            for c, base in enumerate(color):
                rgb[:, :, c] += mask * brightness * base

        return np.clip(rgb, 0, 1)

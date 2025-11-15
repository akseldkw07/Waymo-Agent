from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
else:
    Axes = t.Any  # type: ignore
    Figure = t.Any  # type: ignore

from ..dataclasses import RequestStatus, VehicleState, VehicleStatus
from .interface import GraphMixinInterface


class RideShareRenderingMixin(GraphMixinInterface):
    """Rendering isolated into its own mixin."""

    def render(self):
        mode = self.render_mode or "human"
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")
        if mode == "human":
            return self._render_matplotlib()
        if mode == "ansi":
            return self._render_text()
        return None

    def save_render(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prev_mode = self.render_mode
        try:
            if prev_mode != "human":
                self.render_mode = "human"
            self._render_matplotlib(save_path=path)
        finally:
            self.render_mode = prev_mode

    def close(self):
        if self._figure is not None and self._mpl is not None:
            self._mpl.close(self._figure)
        self._figure = None
        self._axes = None
        self._mpl = None
        self._line_cls = None

    def _render_matplotlib(self, save_path: Path | None = None):
        try:
            if self._mpl is None:
                import matplotlib.pyplot as plt
                from matplotlib.lines import Line2D

                plt.ion()
                self._mpl = plt
                self._line_cls = Line2D
            elif self._line_cls is None:
                from matplotlib.lines import Line2D

                self._line_cls = Line2D
            span_x = float(np.max(self.node_coords[:, 0]) - np.min(self.node_coords[:, 0]))
            span_y = float(np.max(self.node_coords[:, 1]) - np.min(self.node_coords[:, 1]))
            aspect = span_y / max(span_x, 1e-6)
            fig_w = max(6.0, min(18.0, span_x * 3.0))
            fig_h = max(8.0, min(32.0, fig_w * aspect))
            need_new = (
                self._figure is None
                or self._axes is None
                or not self._mpl.fignum_exists(self._figure.number)
                or tuple(self._figure.get_size_inches()) != (fig_w, fig_h)
            )
            if need_new:
                self._figure, self._axes = self._mpl.subplots(figsize=(fig_w, fig_h))
        except ImportError as exc:
            raise ImportError("Matplotlib is required for human render mode.") from exc

        ax = t.cast(Axes, self._axes)
        line_cls = self._line_cls
        if line_cls is None:
            raise RuntimeError("Matplotlib Line2D class not initialized.")
        ax.clear()
        node_x = self.node_coords[:, 0]
        node_y = self.node_coords[:, 1]
        node_scale = float(np.clip(4000.0 / max(1, self.num_nodes), 4.0, 40.0))
        charging_scale = node_scale * 1.6

        for src, dst in self.graph_edges:
            ax.plot(
                [node_x[src], node_x[dst]],
                [node_y[src], node_y[dst]],
                color="#dddddd",
                linewidth=1.0,
                zorder=0,
            )

        ax.scatter(node_x, node_y, c="#444444", s=node_scale, zorder=2)
        if self.charging_nodes:
            ax.scatter(
                node_x[self.charging_nodes],
                node_y[self.charging_nodes],
                c="#2ca02c",
                s=charging_scale,
                marker="s",
                zorder=3,
            )

        status_colors: dict[VehicleStatus, str] = {
            VehicleStatus.IDLE: "#1f77b4",
            VehicleStatus.TO_PICKUP: "#ff7f0e",
            VehicleStatus.WITH_PASSENGER: "#d62728",
            VehicleStatus.REPOSITIONING: "#9467bd",
            VehicleStatus.CHARGING: "#17becf",
        }

        for idx, vehicle in enumerate(self.vehicles):
            color = status_colors.get(vehicle.status, "#7f7f7f")
            pos = self._vehicle_position(vehicle)
            ax.scatter(pos[0], pos[1], c=color, s=70, edgecolors="black", linewidths=0.6, zorder=4)
            ax.text(
                pos[0],
                pos[1] + 0.2,
                f"V{idx} {vehicle.battery:.0%}",
                fontsize=7,
                ha="center",
                va="bottom",
                zorder=5,
            )

        for request in self.pending_requests:
            if request is None:
                continue
            if request.status == RequestStatus.CANCELLED:
                continue
            pickup = self.node_coords[request.pickup_node]
            dropoff = self.node_coords[request.dropoff_node]
            if request.status == RequestStatus.AWAITING_PRICE:
                color = "#ffbb78"
            elif request.status == RequestStatus.ACCEPTED:
                color = "#2ca02c"
            elif request.status == RequestStatus.ASSIGNED:
                color = "#1f77b4"
            else:
                color = "#c7c7c7"
            ax.scatter(pickup[0], pickup[1], c=color, s=60, marker="^", zorder=6)
            ax.plot(
                [pickup[0], dropoff[0]],
                [pickup[1], dropoff[1]],
                color=color,
                linewidth=1.2,
                alpha=0.5,
                zorder=1,
            )
            ax.text(
                pickup[0],
                pickup[1] - 0.25,
                f"${request.price:.1f} / {request.wait_time:.0f}m",
                fontsize=6,
                ha="center",
                va="top",
                zorder=7,
            )

        for ride in self.active_rides.values():
            pickup = self.node_coords[ride.pickup_node]
            dropoff = self.node_coords[ride.dropoff_node]
            ax.plot(
                [pickup[0], dropoff[0]],
                [pickup[1], dropoff[1]],
                color="#d62728",
                linewidth=2.0,
                alpha=0.7,
                zorder=1,
            )

        padding_x = max(0.5, span_x * 0.05)
        padding_y = max(0.5, span_y * 0.03)
        ax.set_xlim(node_x.min() - padding_x, node_x.max() + padding_x)
        ax.set_ylim(node_y.min() - padding_y, node_y.max() + padding_y)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        legend_handles = [
            line_cls(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#444444",
                markeredgecolor="black",
                markersize=6,
                label="Node",
            ),
            line_cls(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="#2ca02c",
                markeredgecolor="black",
                markersize=6,
                label="Charging",
            ),
            line_cls(
                [0],
                [0],
                marker="^",
                color="w",
                markerfacecolor="#ffbb78",
                markeredgecolor="black",
                markersize=6,
                label="Request",
            ),
            line_cls([0], [0], color="#d62728", linewidth=2.0, label="Active Ride"),
        ]
        for status, color in status_colors.items():
            legend_handles.append(
                line_cls(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=6,
                    label=status.name.replace("_", " ").title(),
                )
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, frameon=False)

        metrics = self.metrics
        ax.text(
            ax.get_xlim()[0] + 0.1,
            ax.get_ylim()[1] - 0.1,
            f"Completed: {metrics.get('completed_rides', 0):.0f}\n"
            f"Rejected: {metrics.get('rejected_requests', 0):.0f}\n"
            f"Cancelled: {metrics.get('cancelled_requests', 0):.0f}\n"
            f"Revenue: ${metrics.get('earned_revenue', 0):.1f}",
            fontsize=8,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.4},
        )

        ax.set_title(
            f"Step {self.current_step} | t={self.time_of_day:.2f}h | Weather={self.WEATHER_STATES[self.weather_idx]}"
        )
        fig = t.cast(Figure, self._figure)
        fig.canvas.draw()
        fig.canvas.flush_events()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300)
        self._mpl.pause(0.001)
        return None

    def _render_text(self) -> str:
        lines = [
            f"Step {self.current_step} | time={self.time_of_day:.2f}h | day={self.day_of_week} | weather={self.WEATHER_STATES[self.weather_idx]}",
            f"Pending: {sum(1 for r in self.pending_requests if r is not None)} | Active rides: {len(self.active_rides)}",
            f"Metrics: {self.metrics}",
        ]
        for idx, vehicle in enumerate(self.vehicles):
            lines.append(
                f"V{idx} node={vehicle.node} battery={vehicle.battery:.2f} "
                f"status={vehicle.status.name} target={vehicle.target_node} remaining={vehicle.remaining_distance:.2f}"
            )
        return "\n".join(lines)

    def _vehicle_position(self, vehicle: VehicleState) -> np.ndarray:
        if vehicle.target_node is not None and vehicle.origin_node is not None and vehicle.travel_distance_total > 1e-6:
            start = self.node_coords[vehicle.origin_node]
            end = self.node_coords[vehicle.target_node]
            progress = 1.0 - (vehicle.remaining_distance / max(vehicle.travel_distance_total, 1e-6))
            progress = float(np.clip(progress, 0.0, 1.0))
            return start + progress * (end - start)
        return self.node_coords[vehicle.node]

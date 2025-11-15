#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

import numpy as np


@dataclass
class AvenueDef:
    key: str
    label: str
    base_position: float
    min_alpha: float = 0.0
    max_alpha: float = 1.0
    excluded_ranges: list[tuple[float, float]] = field(default_factory=list)
    curve_points: list[tuple[float, float]] | None = None
    charger_slots: int = 0
    highway_speed: bool = False
    park_safe: bool = False

    def is_active(self, alpha: float) -> bool:
        if alpha < self.min_alpha or alpha > self.max_alpha:
            return False
        return not any(start <= alpha <= end for start, end in self.excluded_ranges)

    def position_at(self, alpha: float) -> float:
        if not self.curve_points:
            return self.base_position
        xs, ys = zip(*self.curve_points)
        return float(np.interp(alpha, xs, ys))


@dataclass
class ManhattanCsvBuilder:
    total_length_km: float = 21.6
    num_cross_streets: int = 190
    seed: int = 7

    def __post_init__(self):
        self.avenues: list[AvenueDef] = [
            AvenueDef("W_HWY", "West Side Hwy", 0.0, charger_slots=30, highway_speed=True, park_safe=True),
            AvenueDef("11AVE", "11th Ave", 0.08, min_alpha=0.14),
            AvenueDef("10AVE", "10th Ave", 0.16, min_alpha=0.12),
            AvenueDef("09AVE", "9th Ave", 0.23, min_alpha=0.10),
            AvenueDef("08AVE", "8th Ave / CPW", 0.30, min_alpha=0.08, park_safe=True),
            AvenueDef("07AVE", "7th Ave", 0.37, min_alpha=0.06, excluded_ranges=[(0.44, 0.68)]),
            AvenueDef(
                "BWAY",
                "Broadway",
                0.43,
                min_alpha=0.02,
                curve_points=[(0.0, 0.32), (0.25, 0.45), (0.55, 0.52), (0.85, 0.65), (1.0, 0.7)],
            ),
            AvenueDef("06AVE", "6th Ave", 0.45, min_alpha=0.05, excluded_ranges=[(0.44, 0.68)]),
            AvenueDef("05AVE", "5th Ave", 0.52, min_alpha=0.03, park_safe=True),
            AvenueDef("MAD", "Madison Ave", 0.58, min_alpha=0.10, excluded_ranges=[(0.44, 0.68)]),
            AvenueDef("PARK", "Park Ave", 0.62, min_alpha=0.10, excluded_ranges=[(0.44, 0.68)]),
            AvenueDef("LEX", "Lexington Ave", 0.67, min_alpha=0.09, park_safe=True),
            AvenueDef("03AVE", "3rd Ave", 0.73, min_alpha=0.09),
            AvenueDef("02AVE", "2nd Ave", 0.80, min_alpha=0.11),
            AvenueDef("01AVE", "1st Ave", 0.88, min_alpha=0.11, park_safe=True),
            AvenueDef("E_HWY", "FDR / Harlem River", 1.0, charger_slots=30, highway_speed=True, park_safe=True),
        ]
        # Estimated west/east boundaries (km) along normalized north-south axis.
        self.west_profile = [
            (0.0, -0.25),
            (0.05, -0.45),
            (0.12, -0.85),
            (0.2, -1.1),
            (0.3, -1.25),
            (0.45, -1.5),
            (0.6, -1.65),
            (0.75, -1.75),
            (0.9, -1.82),
            (1.0, -1.9),
        ]
        self.east_profile = [
            (0.0, 0.3),
            (0.08, 0.6),
            (0.15, 1.05),
            (0.25, 1.3),
            (0.35, 1.5),
            (0.45, 1.75),
            (0.55, 1.8),
            (0.65, 1.72),
            (0.75, 1.65),
            (0.9, 1.58),
            (1.0, 1.55),
        ]
        self.central_park_range = (0.44, 0.68)

    def build(self):
        nodes: list[dict[str, object]] = []
        node_lookup: dict[tuple[int, str], dict[str, object]] = {}
        nodes_by_id: dict[str, dict[str, object]] = {}
        street_rows: list[list[tuple[float, str, str]]] = []
        street_alphas = self._street_alphas(self.num_cross_streets)

        for s_idx, alpha in enumerate(street_alphas):
            y = alpha * self.total_length_km
            west = self._interp_profile(self.west_profile, alpha)
            east = self._interp_profile(self.east_profile, alpha)
            width = max(east - west, 0.2)
            row_nodes: list[tuple[float, str, str]] = []

            for avenue in self.avenues:
                if not avenue.is_active(alpha):
                    continue
                # Keep only border avenues through Central Park interior.
                if self.central_park_range[0] <= alpha <= self.central_park_range[1] and not avenue.park_safe:
                    continue
                position = np.clip(avenue.position_at(alpha), 0.0, 1.0)
                x = west + width * position
                node_id = f"S{s_idx:03d}_{avenue.key}"
                is_charger = avenue.charger_slots > 0 and s_idx % 18 == 0
                node = {
                    "node_id": node_id,
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "is_charger": "true" if is_charger else "false",
                    "num_chargers": avenue.charger_slots if is_charger else 0,
                    "zone": "manhattan_realistic",
                }
                nodes.append(node)
                node_lookup[(s_idx, avenue.key)] = node
                nodes_by_id[node_id] = node
                row_nodes.append((x, node_id, avenue.key))
            street_rows.append(sorted(row_nodes, key=lambda item: item[0]))

        edges = self._build_edges(node_lookup, nodes_by_id, street_rows)
        return nodes, edges

    def _build_edges(
        self,
        node_lookup: dict[tuple[int, str], dict[str, object]],
        nodes_by_id: dict[str, dict[str, object]],
        street_rows: list[list[tuple[float, str, str]]],
    ):
        edges: list[dict[str, object]] = []
        edge_idx = 0

        for street_idx in range(len(street_rows) - 1):
            for avenue in self.avenues:
                src = node_lookup.get((street_idx, avenue.key))
                dst = node_lookup.get((street_idx + 1, avenue.key))
                if not src or not dst:
                    continue
                self._add_edge_pair(edges, src, dst, avenue, edge_idx)
                edge_idx += 2

        # Horizontal connections (east-west).
        for street_idx, row in enumerate(street_rows):
            for idx in range(len(row) - 1):
                _, src_id, _ = row[idx]
                _, dst_id, _ = row[idx + 1]
                src = nodes_by_id.get(src_id)
                dst = nodes_by_id.get(dst_id)
                if not src or not dst:
                    continue
                length = self._distance(src, dst)
                edges.append(
                    self._edge_dict(
                        edge_idx,
                        src["node_id"],
                        dst["node_id"],
                        length,
                        32.0,
                        is_two_way="true",
                    )
                )
                edges.append(
                    self._edge_dict(
                        edge_idx + 1,
                        dst["node_id"],
                        src["node_id"],
                        length,
                        32.0,
                        is_two_way="true",
                    )
                )
                edge_idx += 2

        return edges

    def _add_edge_pair(self, edges, src, dst, avenue: AvenueDef, edge_idx: int):
        length = self._distance(src, dst)
        speed = 55.0 if avenue.highway_speed else 38.0
        edges.append(
            self._edge_dict(
                edge_idx,
                src["node_id"],
                dst["node_id"],
                length,
                speed,
                is_two_way="true",
            )
        )
        edges.append(
            self._edge_dict(
                edge_idx + 1,
                dst["node_id"],
                src["node_id"],
                length,
                speed,
                is_two_way="true",
            )
        )

    def _edge_dict(self, edge_idx: int, src_id: str, dst_id: str, length: float, speed: float, *, is_two_way: str):
        return {
            "edge_id": f"E{edge_idx}",
            "src": src_id,
            "dst": dst_id,
            "length_km": round(length, 4),
            "speed_limit_kmh": speed,
            "is_two_way": is_two_way,
        }

    def _distance(self, src: dict[str, object], dst: dict[str, object]) -> float:
        dx = float(dst["x"]) - float(src["x"])
        dy = float(dst["y"]) - float(src["y"])
        return float(np.hypot(dx, dy))

    def _interp_profile(self, profile: Sequence[tuple[float, float]], alpha: float) -> float:
        xs, ys = zip(*profile)
        return float(np.interp(alpha, xs, ys))

    def _street_alphas(self, count: int) -> np.ndarray:
        base = np.linspace(0.0, 1.0, count)
        eased = 0.6 * np.power(base, 0.85) + 0.4 * np.power(base, 1.25)
        eased /= eased[-1]
        return eased

    def write_csv(self, nodes_path: Path, edges_path: Path, nodes, edges):
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        with nodes_path.open("w", newline="") as node_file:
            writer = csv.DictWriter(
                node_file,
                fieldnames=["node_id", "x", "y", "is_charger", "num_chargers", "zone"],
            )
            writer.writeheader()
            writer.writerows(nodes)
        with edges_path.open("w", newline="") as edge_file:
            writer = csv.DictWriter(
                edge_file,
                fieldnames=["edge_id", "src", "dst", "length_km", "speed_limit_kmh", "is_two_way"],
            )
            writer.writeheader()
            writer.writerows(edges)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Generate Manhattan-like map CSVs.")
    parser.add_argument(
        "--nodes-path",
        type=Path,
        default=Path("maps/manhattan_shaped_nodes.csv"),
        help="Output CSV for nodes (default: maps/manhattan_shaped_nodes.csv)",
    )
    parser.add_argument(
        "--edges-path",
        type=Path,
        default=Path("maps/manhattan_shaped_edges.csv"),
        help="Output CSV for edges (default: maps/manhattan_shaped_edges.csv)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    builder = ManhattanCsvBuilder()
    nodes, edges = builder.build()
    builder.write_csv(args.nodes_path, args.edges_path, nodes, edges)
    print(
        f"Wrote {len(nodes)} nodes to {args.nodes_path} and {len(edges)} edges to {args.edges_path}",
    )


if __name__ == "__main__":
    main()

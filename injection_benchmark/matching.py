"""Sparse maximum-cardinality, minimum-cost bipartite matching helpers."""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment


def maximum_cardinality_min_cost_matching(
    n_left: int,
    n_right: int,
    edges: list[tuple[int, int, float]],
) -> dict[int, int]:
    """Match as many left/right vertices as possible, then minimize edge cost.

    ``edges`` contains only legal matches. Disconnected components are solved
    independently so large injection campaigns do not allocate one dense
    truth-by-event matrix for the entire run.
    """
    if n_left < 0 or n_right < 0:
        raise ValueError("Bipartite vertex counts must be non-negative")
    if n_left == 0 or n_right == 0 or not edges:
        return {}

    edge_cost: dict[tuple[int, int], float] = {}
    left_to_right: dict[int, set[int]] = {}
    right_to_left: dict[int, set[int]] = {}
    for left, right, cost in edges:
        left = int(left)
        right = int(right)
        cost = float(cost)
        if not (0 <= left < n_left and 0 <= right < n_right):
            raise IndexError(f"Matching edge ({left}, {right}) is out of bounds")
        if not np.isfinite(cost):
            continue
        key = (left, right)
        if key not in edge_cost or cost < edge_cost[key]:
            edge_cost[key] = cost
        left_to_right.setdefault(left, set()).add(right)
        right_to_left.setdefault(right, set()).add(left)

    matched: dict[int, int] = {}
    visited_left: set[int] = set()
    for start_left in sorted(left_to_right):
        if start_left in visited_left:
            continue

        component_left: set[int] = set()
        component_right: set[int] = set()
        queue = deque([("left", start_left)])
        while queue:
            side, vertex = queue.popleft()
            if side == "left":
                if vertex in component_left:
                    continue
                component_left.add(vertex)
                visited_left.add(vertex)
                queue.extend(("right", right) for right in left_to_right.get(vertex, ()))
            else:
                if vertex in component_right:
                    continue
                component_right.add(vertex)
                queue.extend(("left", left) for left in right_to_left.get(vertex, ()))

        left_ids = sorted(component_left)
        right_ids = sorted(component_right)
        left_pos = {vertex: index for index, vertex in enumerate(left_ids)}
        right_pos = {vertex: index for index, vertex in enumerate(right_ids)}

        component_costs = [
            edge_cost[(left, right)]
            for left in left_ids
            for right in left_to_right.get(left, ())
            if right in component_right and (left, right) in edge_cost
        ]
        minimum = min(component_costs)
        shift = -minimum if minimum < 0 else 0.0
        maximum = max(cost + shift for cost in component_costs)
        max_matches = min(len(left_ids), len(right_ids))
        dummy_penalty = (maximum + 1.0) * (max_matches + 1)
        invalid_penalty = dummy_penalty * (len(left_ids) + 2)

        # Every left vertex receives its own dummy option. Paying one dummy
        # penalty is more expensive than every possible real-edge cost
        # difference in this component, so cardinality is optimized first.
        cost_matrix = np.full(
            (len(left_ids), len(right_ids) + len(left_ids)),
            invalid_penalty,
            dtype=np.float64,
        )
        cost_matrix[:, len(right_ids):] = dummy_penalty
        for left in left_ids:
            for right in left_to_right.get(left, ()):
                key = (left, right)
                if right in right_pos and key in edge_cost:
                    cost_matrix[left_pos[left], right_pos[right]] = edge_cost[key] + shift

        rows, columns = linear_sum_assignment(cost_matrix)
        for row, column in zip(rows, columns):
            if column >= len(right_ids):
                continue
            left = left_ids[int(row)]
            right = right_ids[int(column)]
            if (left, right) in edge_cost:
                matched[left] = right

    return matched

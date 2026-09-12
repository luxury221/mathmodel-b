from __future__ import annotations

import itertools
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array


def neighborhood_costs(centers, radii):
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or radii.shape != (len(centers),):
        raise ValueError('Invalid neighborhood shapes')
    if not np.isfinite(centers).all() or not np.isfinite(radii).all() or np.any(radii < 0):
        raise ValueError('Invalid neighborhood values')
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    return np.maximum(0.0, distances - radii[:, None] - radii[None, :])


def components(node_count, edges, selected):
    adjacency = [set() for _node in range(node_count)]
    for (first, second), chosen in zip(edges, selected):
        if chosen > 0.5:
            adjacency[first].add(second)
            adjacency[second].add(first)
    remaining = set(range(node_count))
    groups = []
    while remaining:
        reached = {min(remaining)}
        queue = list(reached)
        while queue:
            node = queue.pop()
            fresh = adjacency[node] - reached
            reached.update(fresh)
            queue.extend(sorted(fresh))
        groups.append(reached)
        remaining -= reached
    return groups


def open_path_lower_bound(costs, time_limit=20.0):
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or len(costs) < 2:
        raise ValueError('At least two nodes in a square cost matrix are required')
    if not np.isfinite(costs).all() or np.any(costs < 0) or not np.allclose(costs, costs.T):
        raise ValueError('Nonnegative finite symmetric costs are required')
    real_count = len(costs)
    dummy = real_count
    node_count = real_count + 1
    edges = list(itertools.combinations(range(node_count), 2))
    objective = np.array([costs[first, second] if second != dummy else 0.0 for first, second in edges])
    lower = np.zeros(len(edges))
    lower[edges.index((0, dummy))] = 1.0
    row_ids, column_ids = [], []
    for edge_index, (first, second) in enumerate(edges):
        row_ids.extend((first, second))
        column_ids.extend((edge_index, edge_index))
    values = [1.0] * len(row_ids)
    constraint_lower = [2.0] * node_count
    constraint_upper = [2.0] * node_count
    cuts = set()
    rounds = []
    best_dual = 0.0
    started = time.perf_counter()
    connected_optimum = False
    while time.perf_counter() - started < time_limit:
        matrix = coo_array((values, (np.asarray(row_ids, dtype=np.int32), np.asarray(column_ids, dtype=np.int32))),
                           shape=(len(constraint_lower), len(edges))).tocsc()
        remaining_time = max(0.01, time_limit - (time.perf_counter() - started))
        result = milp(objective, integrality=np.ones(len(edges)), bounds=Bounds(lower, np.ones(len(edges))),
                      constraints=LinearConstraint(matrix, constraint_lower, constraint_upper),
                      options={'time_limit': remaining_time, 'mip_rel_gap': 0.0})
        dual = getattr(result, 'mip_dual_bound', None)
        primal = getattr(result, 'fun', None)
        if dual is not None and np.isfinite(dual):
            best_dual = max(best_dual, float(dual))
        groups = components(node_count, edges, result.x) if result.x is not None else []
        rounds.append({'status': int(result.status), 'message': result.message,
                       'dual': float(dual) if dual is not None and np.isfinite(dual) else None,
                       'primal': float(primal) if primal is not None and np.isfinite(primal) else None,
                       'component_sizes': [len(group) for group in groups],
                       'elapsed_seconds': time.perf_counter() - started})
        if result.status == 2:
            raise ValueError('A complete graph path relaxation cannot be infeasible')
        if len(groups) == 1:
            connected_optimum = result.status == 0
            break
        if not groups:
            break
        new_cuts = 0
        for group in groups:
            complement = set(range(node_count)) - group
            selected = min(tuple(sorted(group)), tuple(sorted(complement)))
            if selected in cuts:
                continue
            cuts.add(selected)
            row_index = len(constraint_lower)
            for edge_index, (first, second) in enumerate(edges):
                if (first in group) != (second in group):
                    row_ids.append(row_index)
                    column_ids.append(edge_index)
                    values.append(1.0)
            constraint_lower.append(2.0)
            constraint_upper.append(np.inf)
            new_cuts += 1
        if not new_cuts:
            break
    return {'lower_bound_meters': max(0.0, best_dual - 1e-4), 'raw_solver_dual': best_dual,
            'connected_relaxation_optimal': connected_optimum, 'subtour_cuts': len(cuts),
            'wall_seconds': time.perf_counter() - started, 'rounds': rounds}

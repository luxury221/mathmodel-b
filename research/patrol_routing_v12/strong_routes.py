from __future__ import annotations

from collections import Counter, defaultdict, deque
import time

import numpy as np
from geometry import open_route as reference_route
from geometry import route_length
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def point_indices(route, points):
    indices = defaultdict(deque)
    for index, point in enumerate(points):
        indices[tuple(point)].append(index)
    return np.array([indices[tuple(point)].popleft() for point in route], dtype=int)


def order_cost(order, distances, start_index):
    previous = np.r_[start_index, order[:-1]]
    return float(distances[previous, order].sum()) if len(order) else 0.0


def exact_order(distances, count):
    if count == 0:
        return np.empty(0, dtype=int)
    costs = np.full((1 << count, count), np.inf)
    parents = np.full((1 << count, count), -1, dtype=np.int16)
    for last in range(count):
        costs[1 << last, last] = distances[count, last]
    for mask in range(1, 1 << count):
        visited = [index for index in range(count) if mask & (1 << index)]
        for last in visited:
            previous_mask = mask ^ (1 << last)
            if not previous_mask:
                continue
            previous = [index for index in visited if index != last]
            proposals = costs[previous_mask, previous] + distances[previous, last]
            selected = int(np.argmin(proposals))
            costs[mask, last] = proposals[selected]
            parents[mask, last] = previous[selected]
    mask = (1 << count) - 1
    last = int(np.argmin(costs[mask]))
    reverse = []
    while last >= 0:
        reverse.append(last)
        previous = int(parents[mask, last])
        mask ^= 1 << last
        last = previous
    return np.asarray(reverse[::-1], dtype=int)


def two_opt(order, distances, start_index):
    order = np.asarray(order, dtype=int).copy()
    count = len(order)
    for _iteration in range(40):
        improved = False
        for left in range(count - 1):
            rights = np.arange(left + 1, count)
            previous = start_index if left == 0 else order[left - 1]
            change = distances[previous, order[rights]] - distances[previous, order[left]]
            if len(rights) > 1:
                change[:-1] += distances[order[left], order[rights[:-1] + 1]] - distances[order[rights[:-1]], order[rights[:-1] + 1]]
            selected = int(np.argmin(change))
            if change[selected] < -1e-8:
                right = int(rights[selected])
                order[left:right + 1] = order[left:right + 1][::-1]
                improved = True
        if not improved:
            break
    return order


def relocate(order, distances, start_index):
    order = order.copy()
    for _iteration in range(32):
        best = None
        for index, node in enumerate(order):
            previous = start_index if index == 0 else order[index - 1]
            removal = -distances[previous, node]
            if index + 1 < len(order):
                following = order[index + 1]
                removal += distances[previous, following] - distances[node, following]
            remaining = np.delete(order, index)
            starts = np.r_[start_index, remaining[:-1]]
            insertion = distances[starts, node] + distances[node, remaining] - distances[starts, remaining]
            insertion = np.r_[insertion, distances[remaining[-1], node]]
            gap = int(np.argmin(insertion))
            change = float(removal + insertion[gap])
            if change < -1e-8 and (best is None or change < best[0]):
                best = change, remaining, node, gap
        if best is None:
            break
        _change, remaining, node, gap = best
        order = np.insert(remaining, gap, node)
    return order


def multistart_order(distances, initial):
    count = len(initial)
    if count <= 10:
        return exact_order(distances, count)
    seeds = [initial, initial[::-1]]
    sorted_first = np.argsort(distances[count, :count], kind='stable')
    first_indices = sorted_first[np.linspace(0, count - 1, min(count, 32), dtype=int)]
    for first in first_indices:
        available = np.ones(count, dtype=bool)
        available[first] = False
        order = [int(first)]
        while available.any():
            candidates = np.flatnonzero(available)
            selected = int(candidates[np.argmin(distances[order[-1], candidates])])
            order.append(selected)
            available[selected] = False
        seeds.append(np.asarray(order))
    proposals = [two_opt(seed, distances, count) for seed in seeds]
    proposals.sort(key=lambda order: order_cost(order, distances, count))
    finalists = []
    for proposal in proposals[:4]:
        for _iteration in range(3):
            refined = two_opt(relocate(proposal, distances, count), distances, count)
            if order_cost(refined, distances, count) >= order_cost(proposal, distances, count) - 1e-8:
                break
            proposal = refined
        finalists.append(proposal)
    return min([initial, *finalists], key=lambda order: order_cost(order, distances, count))


def guided_order(distances, initial, solution_limit=48):
    count = len(initial)
    if count <= 10:
        return initial, {'status': 'exact_small', 'solutions': 0}
    costs = np.zeros((count + 2, count + 2), dtype=np.int64)
    costs[:count + 1, :count + 1] = np.rint(distances * 1000).astype(np.int64)
    manager = pywrapcp.RoutingIndexManager(count + 2, 1, [count], [count + 1])
    routing = pywrapcp.RoutingModel(manager)
    def transit(first, second):
        return int(costs[manager.IndexToNode(first), manager.IndexToNode(second)])
    callback = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(callback)
    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.solution_limit = solution_limit
    parameters.time_limit.FromMilliseconds(2000)
    routing.CloseModelWithParameters(parameters)
    assignment = routing.ReadAssignmentFromRoutes([initial.tolist()], True)
    result = routing.SolveFromAssignmentWithParameters(assignment, parameters) if assignment is not None else None
    solutions = routing.solver().Solutions()
    metadata = {'status': int(routing.status()), 'solutions': solutions, 'solution_limit': solution_limit}
    if result is None or solutions < solution_limit:
        metadata['status'] = 'deterministic_fallback'
        return initial, metadata
    order = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        if node < count:
            order.append(node)
        index = result.Value(routing.NextVar(index))
    if len(order) != count or sorted(order) != list(range(count)):
        raise ValueError('Routing solver did not visit exactly the input multiset')
    order = np.asarray(order, dtype=int)
    return min((initial, order), key=lambda candidate: order_cost(candidate, distances, count)), metadata


class StrongRouter:
    def __init__(self, mode='multistart'):
        if mode not in ('multistart', 'guided'):
            raise ValueError('Unknown routing mode')
        self.mode = mode
        self.events = []
        self.cache = {}

    def __call__(self, points, start=(0.0, 0.0)):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        start = np.asarray(start, dtype=float)
        if start.shape != (2,) or not np.all(np.isfinite(points)) or not np.all(np.isfinite(start)):
            raise ValueError('Invalid routing coordinates')
        key = (tuple(start), tuple(map(tuple, points)))
        if key in self.cache:
            return self.cache[key].copy()
        began = time.perf_counter()
        reference = reference_route(points, start)
        if len(points) <= 1:
            self.cache[key] = reference.copy()
            return reference
        coordinates = np.vstack((points, start))
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=2)
        initial = point_indices(reference, points)
        order = multistart_order(distances, initial)
        metadata = {'status': 'multistart', 'solutions': 0}
        if self.mode == 'guided':
            order, metadata = guided_order(distances, order)
        route = points[order].copy()
        old_cost, new_cost = route_length(reference, start), route_length(route, start)
        if new_cost > old_cost + 1e-7 or Counter(map(tuple, route)) != Counter(map(tuple, points)):
            raise ValueError('Strong router violated the no-worse complete-visit invariant')
        self.cache[key] = route.copy()
        self.events.append({'points': len(points), 'original_meters': old_cost, 'selected_meters': new_cost,
                            'seconds': time.perf_counter() - began, 'solver': metadata})
        return route

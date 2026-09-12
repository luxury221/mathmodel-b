from __future__ import annotations

import copy

import numpy as np
from geometry import clip_polygon, minimum_circle, open_route, route_length
from plan_geometry import hull_vertices
from shapely.geometry import MultiPoint, Polygon


BOX = np.array([[-6000.0, -6000.0], [6000.0, -6000.0], [6000.0, 6000.0], [-6000.0, 6000.0]])


class AdaptiveCoverPlanner:
    def __init__(self, policy):
        self.policy = policy
        self.certificates_checked = 0
        self.relocations = 0
        self.prunes = 0

    def complete(self, coverage, points):
        trial = copy.deepcopy(coverage)
        for point in points:
            trial.observe_absence(point)
        self.certificates_checked += 1
        return trial.empty

    def price(self, points):
        policy = self.policy
        targets = [state.near_position if state.status == 'NEAR' else policy.center_radius(state)[0]
                   for state in policy.states.values() if state.status in ('DETECTED', 'NEAR')]
        source_route = open_route(np.asarray(targets), policy.position) if targets else np.empty((0, 2))
        start = source_route[-1] if len(source_route) else policy.position
        scan_route = open_route(np.asarray(points), start) if points else np.empty((0, 2))
        travel = route_length(source_route, policy.position) + route_length(scan_route, start)
        return float(travel / 5 + 6 * len(policy.unknown_channels()) * len(points))

    def territory(self, coverage, points, index):
        vertices = BOX.copy()
        site = points[index]
        for other_index, other in enumerate(points):
            if other_index == index:
                continue
            normal = np.asarray(other) - site
            norm = float(np.linalg.norm(normal))
            if norm < 1e-6:
                continue
            bound = float(normal @ (np.asarray(other) + site) / 2)
            vertices = clip_polygon(vertices, normal / norm, bound / norm)
            if not len(vertices):
                return Polygon()
        cell = MultiPoint(vertices).convex_hull
        return coverage.region.intersection(cell)

    def prune(self, coverage, points):
        points = [np.asarray(point).copy() for point in points]
        for index in range(len(points) - 1, -1, -1):
            candidate = points[:index] + points[index + 1:]
            if self.complete(coverage, candidate) and self.price(candidate) < self.price(points) - 5:
                points = candidate
                self.prunes += 1
        return points

    def refine(self, coverage, points, allow_relocation):
        if not self.complete(coverage, points):
            raise ValueError('Input scanning plan lacks complete continuous coverage')
        points = self.prune(coverage, points)
        if not allow_relocation or not points:
            return points
        targets = [state.near_position if state.status == 'NEAR' else self.policy.center_radius(state)[0]
                   for state in self.policy.states.values() if state.status in ('DETECTED', 'NEAR')]
        attractors = [self.policy.position.copy(), *targets]
        for _iteration in range(2):
            any_change = False
            for index in range(len(points)):
                part = self.territory(coverage, points, index)
                if part.is_empty:
                    continue
                center, radius = minimum_circle(hull_vertices(part))
                slack = 999.5 - radius
                if slack < 0:
                    continue
                proposals = [center]
                for attractor in sorted(attractors, key=lambda point: float(np.linalg.norm(point - center)))[:2]:
                    vector = np.asarray(attractor) - center
                    proposals.append(center + vector * min(1.0, slack / max(1e-9, float(np.linalg.norm(vector)))))
                current_cost = self.price(points)
                selected = None
                for point in proposals:
                    if np.linalg.norm(point - points[index]) < 1:
                        continue
                    if any(np.linalg.norm(point - prior) < 25 for prior in coverage.observations):
                        continue
                    candidate = points.copy()
                    candidate[index] = np.asarray(point).copy()
                    cost = self.price(candidate)
                    if cost >= current_cost - 5 or (selected is not None and cost >= selected[0]):
                        continue
                    if self.complete(coverage, candidate):
                        selected = cost, candidate
                if selected is not None:
                    _cost, points = selected
                    self.relocations += 1
                    any_change = True
            if not any_change:
                break
        points = self.prune(coverage, points)
        if not self.complete(coverage, points):
            raise ValueError('Refined scan plan lost its continuous certificate')
        return points

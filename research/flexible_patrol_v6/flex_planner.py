from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import shapely
from geometry import disk_vertices, open_route, route_length
from patrol_planner import CertifiedPatrol
from plan_geometry import hull_vertices, initial_outer_region
from shapely.geometry import Polygon


def clip_halfplane(vertices, normal, bound):
    if len(vertices) == 0:
        return np.empty((0, 2))
    normal = np.asarray(normal, dtype=float)
    scale = float(np.linalg.norm(normal))
    if scale == 0:
        return np.asarray(vertices).copy() if bound >= 0 else np.empty((0, 2))
    normal = normal / scale
    bound = bound / scale
    clipped = []
    previous = vertices[-1]
    previous_distance = float(np.dot(normal, previous) - bound)
    for current in vertices:
        current_distance = float(np.dot(normal, current) - bound)
        if (previous_distance <= 0) != (current_distance <= 0):
            fraction = previous_distance / (previous_distance - current_distance)
            if not 0 <= fraction <= 1:
                raise ValueError('Halfplane crossing extrapolates outside its edge')
            clipped.append(previous + fraction * (current - previous))
        if current_distance <= 0:
            clipped.append(current)
        previous, previous_distance = current, current_distance
    return np.asarray(clipped).reshape(-1, 2)


@dataclass
class Layout:
    indices: list
    points: np.ndarray
    old_meters: float
    new_meters: float
    distance_limit: float
    halfplane_violation: float
    constraint_cells: int


class CellAssignment:
    def __init__(self, coverage, stations, problem, assignment='greedy'):
        self.coverage = coverage
        self.stations = np.asarray(stations)
        self.problem = problem
        self.assignment = assignment
        self.vertices = [[] for _station in stations]
        self.halfplanes = [[] for _station in stations]
        self.cells = 0
        self.maximum_depth = 0
        self.disks = [disk_vertices(point, 999.9, 128) for point in stations]
        self.observation_disks = [disk_vertices(point, 999.9, 128) for point in coverage.observations]
        self.closer = {}
        if assignment == 'nearest' and problem == 'q4':
            boundary = hull_vertices(initial_outer_region())
            for index, station in enumerate(self.stations):
                for other_index, other in enumerate(self.stations):
                    if index == other_index:
                        continue
                    normal = station - other
                    offset = (np.dot(station, station) - np.dot(other, other)) / 2
                    vertices = clip_halfplane(boundary, normal, offset - 1e-5 * np.linalg.norm(normal))
                    self.closer[index, other_index] = Polygon(vertices) if len(vertices) >= 3 else Polygon()

    def record(self, index, region, normals=()):
        if region.is_empty:
            return
        vertices = hull_vertices(region)
        self.vertices[index].extend(vertices)
        for normal in normals:
            self.halfplanes[index].append((normal, float(np.max(vertices @ normal))))
        self.cells += 1

    def omni(self):
        boundary = hull_vertices(initial_outer_region())
        for index, station in enumerate(self.stations):
            vertices = boundary.copy()
            for other_index, other in enumerate(self.stations):
                if other_index != index:
                    normal = other - station
                    offset = (np.dot(other, other) - np.dot(station, station)) / 2
                    vertices = clip_halfplane(vertices, normal, offset + 1e-6 * np.linalg.norm(normal))
                    if len(vertices) < 3:
                        break
            if len(vertices) >= 3:
                self.record(index, self.coverage.region.intersection(Polygon(vertices)))
        return True

    @staticmethod
    def sector(vertices, station, normals):
        for normal in normals:
            vertices = clip_halfplane(vertices, normal, float(np.dot(station, normal)) - 1e-6)
            if len(vertices) < 3:
                return Polygon()
        return Polygon(vertices)

    def directional_interval(self, region, lower, upper, depth=0):
        self.maximum_depth = max(self.maximum_depth, depth)
        normals = [np.array([math.cos(angle), math.sin(angle)]) for angle in (lower, upper)]
        for point, disk in zip(self.coverage.observations, self.observation_disks):
            region = region.difference(self.sector(disk.copy(), point, normals))
        cuts = [self.sector(disk.copy(), point, normals) for point, disk in zip(self.stations, self.disks)]
        if self.assignment == 'nearest':
            for index, cut in enumerate(cuts):
                cell = region.intersection(cut)
                for other_index, other in enumerate(cuts):
                    if index != other_index and not cell.is_empty:
                        removed = other.intersection(self.closer[index, other_index])
                        cell = cell.difference(removed)
                self.record(index, cell, normals)
            region = region.difference(shapely.union_all(cuts))
        else:
            for index, cut in enumerate(cuts):
                self.record(index, region.intersection(cut), normals)
                region = region.difference(cut)
                if region.is_empty:
                    return True
        if region.is_empty:
            return True
        if depth >= 7:
            return False
        middle = (lower + upper) / 2
        return (self.directional_interval(region, lower, middle, depth + 1)
                and self.directional_interval(region, middle, upper, depth + 1))

    def build(self):
        if self.problem == 'q3':
            return self.omni()
        for index in range(36):
            lower, upper = np.deg2rad([index * 10 - 5, index * 10 + 5])
            if not self.directional_interval(self.coverage.region, lower, upper):
                return False
        return True


class FlexiblePatrol(CertifiedPatrol):
    def __init__(self, policy):
        super().__init__(policy)
        self.flex_revision = None
        self.layout_log = []

    @staticmethod
    def path_meters(stations, targets, start):
        points = np.vstack((stations, targets)) if len(targets) else np.asarray(stations)
        return route_length(open_route(points, start), start) if len(points) else 0.0

    def propose(self, coverage, targets):
        policy = self.policy
        indices = policy.remaining_stations.copy()
        if not indices:
            return None
        stations = policy.route[indices]
        assignment = CellAssignment(coverage, stations, policy.problem, getattr(policy, 'flex_assignment', 'greedy'))
        if not assignment.build():
            policy.stats['flex_assignment_rejections'] += 1
            return None
        retained = [index for index, vertices in enumerate(assignment.vertices) if vertices]
        if not retained:
            return None
        kept_indices = [indices[index] for index in retained]
        initial = stations[retained]
        variables = cp.Variable((len(retained), 2))
        variables.value = initial / 1000
        constraints = []
        vertex_banks = []
        plane_banks = []
        edge_angles = (np.arange(128) + 0.5) * 2 * np.pi / 128
        edge_normals = np.column_stack((np.cos(edge_angles), np.sin(edge_angles)))
        polygon_inradius = 999.9 * math.cos(math.pi / 128)
        for local, original in enumerate(retained):
            vertices = hull_vertices(shapely.MultiPoint(assignment.vertices[original]).convex_hull)
            edge_bounds = np.max(vertices @ edge_normals.T, axis=0) - polygon_inradius
            constraints.append(edge_normals @ variables[local] >= (edge_bounds + 1e-5) / 1000)
            vertex_banks.append(vertices)
            planes = [*assignment.halfplanes[original], *zip(edge_normals, edge_bounds)]
            plane_banks.append(planes)
            directional_planes = assignment.halfplanes[original]
            if directional_planes:
                normals = np.array([plane[0] for plane in directional_planes])
                bounds = np.array([plane[1] for plane in directional_planes]) / 1000
                constraints.append(normals @ variables[local] >= bounds + 1e-9)
        destinations = np.vstack((initial, targets)) if len(targets) else initial
        itinerary = open_route(destinations, policy.position)
        used = set()
        expressions = []
        for point in itinerary:
            ordered = np.argsort(np.linalg.norm(destinations - point, axis=1))
            index = next(int(index) for index in ordered if int(index) not in used)
            used.add(index)
            expressions.append(variables[index] if index < len(initial) else point / 1000)
        previous = policy.position / 1000
        costs = []
        for expression in expressions:
            costs.append(cp.norm(expression - previous))
            previous = expression
        objective = cp.sum(cp.hstack(costs)) + 1e-6 * cp.sum_squares(variables - initial / 1000)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        policy.stats['flex_solver_calls'] += 1
        try:
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter('always')
                problem.solve(solver='CLARABEL', max_iter=120, tol_gap_abs=1e-8, tol_feas=1e-9, canon_backend=cp.SCIPY_CANON_BACKEND)
            policy.stats['flex_solver_warnings'] += len(recorded)
        except (cp.error.SolverError, ValueError):
            policy.stats['flex_solver_rejections'] += 1
            return None
        if problem.status not in ('optimal', 'optimal_inaccurate') or variables.value is None:
            policy.stats['flex_solver_rejections'] += 1
            return None
        proposal = np.asarray(variables.value) * 1000
        if not np.all(np.isfinite(proposal)) or np.any(np.linalg.norm(proposal, axis=1) >= 4000):
            return None
        maximum_distance = 0.0
        halfplane_violation = 0.0
        for point, vertices, planes in zip(proposal, vertex_banks, plane_banks):
            maximum_distance = max(maximum_distance, float(np.linalg.norm(vertices - point, axis=1).max()))
            for normal, bound in planes:
                halfplane_violation = max(halfplane_violation, float(bound - np.dot(normal, point)))
        if maximum_distance > 999.90005 or halfplane_violation > 1e-7:
            policy.stats['flex_constraint_rejections'] += 1
            return None
        old_cost = self.path_meters(stations, targets, policy.position)
        new_cost = self.path_meters(proposal, targets, policy.position)
        if new_cost >= old_cost - 0.5:
            return None
        trial = copy.deepcopy(coverage)
        try:
            for point in proposal:
                trial.observe_absence(point)
        except (ValueError, shapely.errors.GEOSException):
            policy.stats['flex_certificate_rejections'] += 1
            return None
        if not trial.empty:
            policy.stats['flex_certificate_rejections'] += 1
            return None
        return Layout(kept_indices, proposal, old_cost, new_cost, maximum_distance, halfplane_violation, assignment.cells)

    def install(self, layout, extra_scan=False):
        policy = self.policy
        removed = len(policy.remaining_stations) - len(layout.indices)
        policy.remaining_stations[:] = layout.indices
        policy.route[layout.indices] = layout.points
        self.matrix = np.array([policy.search_planner.visible(point) for point in policy.route])
        self.revision = None
        policy.stats['flex_layouts_accepted'] += 1
        policy.stats['flex_stations_removed'] += removed
        self.layout_log.append({'observed': len(policy.coverage.observations), 'indices': layout.indices.copy(),
                                'points': layout.points.tolist(), 'old_meters': layout.old_meters,
                                'new_meters': layout.new_meters, 'max_distance': layout.distance_limit,
                                'halfplane_violation': layout.halfplane_violation,
                                'cells': layout.constraint_cells, 'extra_scan': extra_scan})

    def reduce(self, targets):
        super().reduce(targets)
        policy = self.policy
        revision = (len(policy.coverage.observations), len(policy.coverage.clear_observations), tuple(policy.remaining_stations))
        if revision == self.flex_revision or not policy.remaining_stations:
            return
        layout = self.propose(policy.coverage, targets)
        if layout is not None:
            self.install(layout)
        self.flex_revision = (len(policy.coverage.observations), len(policy.coverage.clear_observations), tuple(policy.remaining_stations))

    def replace_from_current_position(self):
        if super().replace_from_current_position():
            return True
        policy = self.policy
        if policy.discovery_done or not policy.unknown_channels() or not policy.remaining_stations:
            return False
        if any(np.linalg.norm(policy.position - point) < 100 for point in policy.coverage.observations):
            return False
        trial = copy.deepcopy(policy.coverage)
        trial.observe_absence(policy.position)
        targets = [policy.center_radius(state)[0] for state in policy.states.values() if state.status == 'DETECTED']
        layout = self.propose(trial, targets)
        if layout is None:
            return False
        fee_meters = len(policy.unknown_channels()) * 6 * 5
        if layout.old_meters - layout.new_meters <= fee_meters + 100:
            return False
        policy.scan_unknown(policy.position, force=True)
        self.install(layout, extra_scan=True)
        policy.stats['flex_extra_scans'] += 1
        return True

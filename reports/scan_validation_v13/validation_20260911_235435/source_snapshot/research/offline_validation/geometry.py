from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import linprog
from shapely.geometry import Point, Polygon

TARGET_RADIUS = 1800.0
MIN_RANGE = 1000.0
MAX_RANGE = 1500.0
SAFE_ERROR_DEG = 1.005


def ring(radius, count, offset_deg=0.0):
    angles = np.deg2rad(offset_deg + np.arange(count) * 360.0 / count)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def disk_vertices(center, radius, count=256, outer=False):
    scale = radius / math.cos(math.pi / count) if outer else radius
    return ring(scale, count) + np.asarray(center)


def seven_network():
    return np.vstack((np.zeros((1, 2)), ring(1125.0, 6)))


def dual_ring_network(inner_radius=1000.0, outer_radius=1870.0):
    return np.vstack(
        (np.zeros((1, 2)), ring(inner_radius, 8), ring(outer_radius, 12, 15.0))
    )


def triangular_network(side=990.0):
    basis = np.array([[side, 0.0], [side / 2.0, math.sqrt(3.0) * side / 2.0]])
    shift = 0.1 * basis.sum(axis=0)
    lattice_indices = set()
    selected_triangles = []
    for first_index in range(-5, 6):
        for second_index in range(-5, 6):
            base = np.array([first_index, second_index])
            for offsets in (
                np.array([[0, 0], [1, 0], [0, 1]]),
                np.array([[1, 0], [0, 1], [1, 1]]),
            ):
                indices = base + offsets
                triangle = indices @ basis + shift
                if Polygon(triangle).distance(Point(0.0, 0.0)) <= TARGET_RADIUS:
                    lattice_indices.update(map(tuple, indices.tolist()))
                    selected_triangles.append(triangle)
    points = np.asarray(sorted(lattice_indices)) @ basis + shift
    return points, selected_triangles


def diameter(points):
    points = np.asarray(points, dtype=float)
    if len(points) <= 1:
        return 0.0
    return float(np.linalg.norm(points[:, None] - points[None, :], axis=2).max())


def pair_circle(first, second):
    center = (first + second) / 2.0
    return center, float(np.linalg.norm(first - center))


def triple_circle(first, second, third):
    points = np.array([first, second, third])
    for first_index, second_index in itertools.combinations(range(3), 2):
        center, radius = pair_circle(points[first_index], points[second_index])
        if np.linalg.norm(points - center, axis=1).max() <= radius + 1e-9:
            return center, radius
    differences = points[1:] - points[0]
    if abs(np.linalg.det(differences)) < 1e-12:
        raise ValueError('Numerically degenerate three-point circle')
    offset = np.linalg.solve(2.0 * differences, np.sum(differences**2, axis=1))
    return points[0] + offset, float(np.linalg.norm(offset))


def minimum_circle(points):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        raise ValueError('Empty feasible set has no clearance certificate')
    points = points[np.random.default_rng(2718).permutation(len(points))]
    center = points[0].copy()
    radius = 0.0
    for first_index, first in enumerate(points):
        if np.linalg.norm(first - center) <= radius + 1e-8:
            continue
        center, radius = first.copy(), 0.0
        for second_index, second in enumerate(points[:first_index]):
            if np.linalg.norm(second - center) <= radius + 1e-8:
                continue
            center, radius = pair_circle(first, second)
            for third in points[:second_index]:
                if np.linalg.norm(third - center) > radius + 1e-8:
                    center, radius = triple_circle(first, second, third)
    radius = float(np.linalg.norm(points - center, axis=1).max())
    return center, radius


def enumerate_circle(points):
    points = np.asarray(points, dtype=float)
    candidates = [(point, 0.0) for point in points]
    candidates.extend(pair_circle(*pair) for pair in itertools.combinations(points, 2))
    candidates.extend(triple_circle(*triple) for triple in itertools.combinations(points, 3))
    valid = [
        (center, radius)
        for center, radius in candidates
        if np.linalg.norm(points - center, axis=1).max() <= radius + 1e-7
    ]
    return min(valid, key=lambda candidate: candidate[1])


def bearing_halfplanes(position, bearing_deg, error_deg=SAFE_ERROR_DEG):
    lower_angle, upper_angle = np.deg2rad(
        [bearing_deg - error_deg, bearing_deg + error_deg]
    )
    normals = np.array(
        [[math.sin(lower_angle), -math.cos(lower_angle)],
         [-math.sin(upper_angle), math.cos(upper_angle)]]
    )
    return normals, normals @ np.asarray(position)


def clip_polygon(vertices, normal, bound):
    if len(vertices) == 0:
        return np.empty((0, 2))
    clipped = []
    previous = vertices[-1]
    previous_distance = float(np.dot(normal, previous) - bound)
    for current in vertices:
        current_distance = float(np.dot(normal, current) - bound)
        previous_inside = previous_distance <= 1e-10
        current_inside = current_distance <= 1e-10
        if previous_inside != current_inside:
            fraction = previous_distance / (previous_distance - current_distance)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous, previous_distance = current, current_distance
    return np.asarray(clipped).reshape(-1, 2)


def update_bearing_polygon(vertices, position, bearing_deg, error_deg=SAFE_ERROR_DEG):
    normals, bounds = bearing_halfplanes(position, bearing_deg, error_deg)
    for normal, bound in zip(normals, bounds):
        vertices = clip_polygon(vertices, normal, bound)
    angles = (np.arange(64) + 0.5) * (2.0 * math.pi / 64.0)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    for normal in normals:
        vertices = clip_polygon(vertices, normal, np.dot(normal, position) + MAX_RANGE)
    return vertices


def initial_region():
    return disk_vertices((0.0, 0.0), TARGET_RADIUS, 128, outer=True)


def halfplane_status(normals, bounds):
    options = {'A_ub': np.asarray(normals), 'b_ub': np.asarray(bounds),
               'bounds': [(None, None), (None, None)], 'method': 'highs'}
    feasibility = linprog([0.0, 0.0], **options)
    if feasibility.status == 2:
        return 'empty'
    if not feasibility.success:
        raise RuntimeError(feasibility.message)
    for objective in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        result = linprog(objective, **options)
        if result.status == 3:
            return 'unbounded'
        if not result.success:
            raise RuntimeError(result.message)
    return 'bounded'


def directional_gap(position, stations, receiving_radius=MIN_RANGE):
    differences = np.asarray(stations) - np.asarray(position)
    distances = np.linalg.norm(differences, axis=1)
    if np.any(distances <= 1e-12):
        return 0.0
    nearby = differences[distances <= receiving_radius]
    if len(nearby) < 2:
        return 360.0
    angles = np.sort(np.mod(np.arctan2(nearby[:, 1], nearby[:, 0]), 2.0 * math.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2.0 * math.pi])
    return float(np.rad2deg(gaps.max()))


def route_length(points, start=(0.0, 0.0)):
    points = np.asarray(points)
    if len(points) == 0:
        return 0.0
    return float(np.linalg.norm(np.diff(np.vstack((start, points)), axis=0), axis=1).sum())


def open_route(points, start=(0.0, 0.0)):
    points = np.asarray(points)
    remaining = list(range(len(points)))
    ordered = []
    current = np.asarray(start)
    while remaining:
        selected = min(remaining, key=lambda index: np.linalg.norm(points[index] - current))
        ordered.append(selected)
        remaining.remove(selected)
        current = points[selected]
    route = points[ordered].copy()
    for _ in range(100):
        improved = False
        for left in range(len(route) - 1):
            for right in range(left + 1, len(route)):
                previous = np.asarray(start) if left == 0 else route[left - 1]
                before = np.linalg.norm(previous - route[left])
                after = np.linalg.norm(previous - route[right])
                if right + 1 < len(route):
                    before += np.linalg.norm(route[right] - route[right + 1])
                    after += np.linalg.norm(route[left] - route[right + 1])
                if after < before - 1e-8:
                    route[left:right + 1] = route[left:right + 1][::-1]
                    improved = True
        if not improved:
            break
    return route


def rectangle_clear_cover(vertices, bearing_deg, start):
    angle = math.radians(bearing_deg)
    basis = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
    local = np.asarray(vertices) @ basis.T
    lower, upper = local.min(axis=0), local.max(axis=0)
    counts = np.maximum(1, np.ceil((upper - lower) / 28.0).astype(int))
    step = (upper - lower) / counts
    candidates = []
    for reverse_rows in (False, True):
        for reverse_first in (False, True):
            centers = []
            rows = list(range(int(counts[1])))
            if reverse_rows:
                rows.reverse()
            for row_index, row in enumerate(rows):
                columns = list(range(int(counts[0])))
                if bool(row_index % 2) != reverse_first:
                    columns.reverse()
                for column in columns:
                    centers.append(lower + (np.array([column, row]) + 0.5) * step)
            candidates.append(np.asarray(centers) @ basis)
    route = min(candidates, key=lambda candidate: route_length(candidate, start))
    return route, float(np.linalg.norm(step) / 2.0)

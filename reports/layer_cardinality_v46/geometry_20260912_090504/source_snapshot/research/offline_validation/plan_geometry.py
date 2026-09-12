from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import shapely
from geometry import (
    bearing_halfplanes,
    clip_polygon,
    disk_vertices,
    initial_region,
    minimum_circle,
    rectangle_clear_cover,
    route_length,
    update_bearing_polygon,
)
from scipy.optimize import minimize
from shapely.geometry import MultiPoint, Point, Polygon

CLEAR_SUPPORT_RADIUS = 19.95
CLEAR_POLYGON_RADIUS = 19.99
RECEPTION_GUARD = 999.9


@dataclass
class ClearPlan:
    centers: np.ndarray
    worst_seconds: float
    certified: bool
    certificate_kind: str


@dataclass
class MeasurementPlan:
    position: np.ndarray
    predicted_radius: float
    positive_proxy_seconds: float
    immediate_seconds: float
    max_receiving_distance: float


def hull_vertices(region):
    hull = region.convex_hull
    if isinstance(hull, Polygon):
        return np.asarray(hull.exterior.coords[:-1])
    if hasattr(hull, 'coords'):
        return np.asarray(hull.coords)
    raise ValueError('Empty or unsupported feasible geometry')


def disk_polygon(center, radius, sides=128):
    return Polygon(disk_vertices(center, radius, sides))


def positive_update(region, position, bearing):
    vertices = update_bearing_polygon(initial_region(), position, bearing)
    constraint = MultiPoint(vertices).convex_hull
    updated = region.intersection(constraint)
    if updated.is_empty or not updated.is_valid:
        raise ValueError('Positive observations yield an empty/invalid feasible set')
    return updated


def clear_failure_update(region, position):
    updated = region.difference(disk_polygon(position, CLEAR_POLYGON_RADIUS))
    if updated.is_empty:
        raise ValueError('Clear failure contradicts the guaranteed feasible region')
    return updated


def region_frame(region):
    vertices = hull_vertices(region)
    if len(vertices) < 2:
        return np.eye(2)
    distances = np.linalg.norm(vertices[:, None] - vertices[None, :], axis=2)
    first_index, second_index = np.unravel_index(np.argmax(distances), distances.shape)
    major = vertices[second_index] - vertices[first_index]
    major /= np.linalg.norm(major)
    return np.array([major, [-major[1], major[0]]])


def radical_inverse(indices, base):
    indices = np.asarray(indices).copy()
    result = np.zeros(len(indices))
    factor = 1.0 / base
    while np.any(indices):
        result += factor * (indices % base)
        indices //= base
        factor /= base
    return result


def sample_region(region, count=48):
    basis = region_frame(region)
    vertices = hull_vertices(region)
    local = vertices @ basis.T
    lower, upper = local.min(axis=0), local.max(axis=0)
    indices = np.arange(1, count * 24 + 1)
    unit = np.column_stack((radical_inverse(indices, 2), radical_inverse(indices, 3)))
    candidates = (lower + unit * (upper - lower)) @ basis
    mask = shapely.covers(region, shapely.points(candidates))
    candidates = candidates[mask]
    if len(candidates) >= count:
        return candidates[np.linspace(0, len(candidates) - 1, count).astype(int)]
    representative = np.asarray(region.representative_point().coords[0])
    if len(candidates) == 0:
        return representative[None, :]
    return np.vstack((candidates, representative))


def conservative_cover_check(region, centers):
    if len(centers) == 0:
        return False
    coverage = shapely.union_all([disk_polygon(center, CLEAR_POLYGON_RADIUS) for center in centers])
    return bool(coverage.is_valid and coverage.covers(region) and region.difference(coverage).is_empty)


def best_clear_order(region, centers, start):
    if not conservative_cover_check(region, centers):
        return None
    best = None
    for permutation in itertools.permutations(range(len(centers))):
        ordered = np.asarray(centers)[list(permutation)]
        remaining = region
        previous = np.asarray(start)
        elapsed_movement = 0.0
        worst = 0.0
        for index, center in enumerate(ordered):
            elapsed_movement += np.linalg.norm(center - previous) / 5.0
            receiving = disk_polygon(center, CLEAR_POLYGON_RADIUS)
            if not remaining.intersection(receiving).is_empty:
                worst = max(worst, elapsed_movement + index * 3 + 5)
            remaining = remaining.difference(receiving)
            previous = center
        if remaining.is_empty and (best is None or worst < best.worst_seconds):
            best = ClearPlan(ordered, float(worst), True, 'inner_disk_union_covers_outer_feasible_set')
    return best


def partition_centers(region, columns, rows, basis):
    vertices = hull_vertices(region)
    local = vertices @ basis.T
    lower, upper = local.min(axis=0), local.max(axis=0)
    step = (upper - lower) / np.array([columns, rows])
    centers = []
    for column, row in itertools.product(range(columns), range(rows)):
        cell_lower = lower + np.array([column, row]) * step
        cell_upper = cell_lower + step
        rectangle = np.array([
            cell_lower, [cell_upper[0], cell_lower[1]], cell_upper,
            [cell_lower[0], cell_upper[1]],
        ]) @ basis
        part = region.intersection(Polygon(rectangle))
        if part.is_empty:
            continue
        center, radius = minimum_circle(hull_vertices(part))
        if radius > CLEAR_SUPPORT_RADIUS:
            return None
        centers.append(center)
    return np.asarray(centers)


def small_clear_plan(region, start, max_circles=4):
    vertices = hull_vertices(region)
    center, radius = minimum_circle(vertices)
    if radius <= CLEAR_SUPPORT_RADIUS:
        return ClearPlan(np.array([center]), float(np.linalg.norm(center - start) / 5 + 5), True, 'mec')
    if radius > max_circles * 20.0:
        return None
    rectangle = region.minimum_rotated_rectangle
    if isinstance(rectangle, Polygon):
        corners = np.asarray(rectangle.exterior.coords)
        edges = np.diff(corners, axis=0)
        major = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        major /= np.linalg.norm(major)
        basis = np.array([major, [-major[1], major[0]]])
    else:
        basis = region_frame(region)
    candidates = []
    for columns, rows in ((2, 1), (3, 1), (4, 1), (2, 2), (1, 2), (1, 3), (1, 4)):
        if columns * rows > max_circles:
            continue
        centers = partition_centers(region, columns, rows, basis)
        if centers is not None and len(centers):
            plan = best_clear_order(region, centers, start)
            if plan is not None:
                candidates.append(plan)
    return min(candidates, key=lambda plan: plan.worst_seconds) if candidates else None


def fallback_plan(region, bearing, start):
    centers, cell_radius = rectangle_clear_cover(hull_vertices(region), bearing, start)
    assert cell_radius < CLEAR_POLYGON_RADIUS * math.cos(math.pi / 128)
    worst = route_length(centers, start) / 5 + 3 * (len(centers) - 1) + 5
    return ClearPlan(centers, worst, True, 'rectangle_partition_circumradius_bound')


def sampled_clear_cost(plan, positions, start, weights=None):
    distances = np.linalg.norm(positions[:, None] - plan.centers[None, :], axis=2)
    covered = distances <= 20.0
    if not np.all(covered.any(axis=1)):
        return plan.worst_seconds
    first_hit = np.argmax(covered, axis=1)
    travel = np.cumsum(np.linalg.norm(np.diff(np.vstack((start, plan.centers)), axis=0), axis=1)) / 5
    costs = travel[first_hit] + first_hit * 3 + 5
    mean = float(np.average(costs, weights=weights))
    return 0.8 * mean + 0.2 * float(costs.max())


def clipped_prediction(vertices, position, bearing):
    normals, bounds = bearing_halfplanes(position, bearing)
    for normal, bound in zip(normals, bounds):
        vertices = clip_polygon(vertices, normal, bound)
    return vertices


def minimax_measurement(region, last_positive, start, visited, optimize=True):
    vertices = hull_vertices(region)
    center, radius = minimum_circle(vertices)
    radial = center - last_positive
    norm = np.linalg.norm(radial)
    radial = radial / norm if norm > 1e-9 else np.array([1.0, 0.0])
    tangent = np.array([-radial[1], radial[0]])
    offsets = (min(550, max(60, 0.7 * radius)), min(350, max(30, 0.35 * radius)))
    candidates = [center + sign * offset * tangent for offset, sign in itertools.product(offsets, (-1, 1))]
    candidates.extend([center + sign * 40 * radial for sign in (-1, 1)])

    def eligible(candidate):
        return (
            np.linalg.norm(vertices - candidate, axis=1).max() <= RECEPTION_GUARD + 1e-8
            and all(np.linalg.norm(candidate - point) > 1e-5 for point in visited)
        )

    def local_proxy(candidate):
        first = vertices - last_positive
        second = vertices - candidate
        first_ranges = np.linalg.norm(first, axis=1)
        second_ranges = np.linalg.norm(second, axis=1)
        product = np.maximum(first_ranges * second_ranges, 1e-10)
        sine = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / product
        cosine = np.abs(np.sum(first * second, axis=1)) / product
        radii = math.tan(math.radians(1.005)) * np.sqrt(
            first_ranges**2 + second_ranges**2 + 2 * product * cosine
        ) / np.maximum(sine, 1e-6)
        return float(radii.max() + 0.002 * np.linalg.norm(candidate - start))

    candidates = [candidate for candidate in candidates if eligible(candidate)]
    if not candidates:
        if eligible(center):
            candidates = [center]
        else:
            return None
    candidates.sort(key=local_proxy)
    if optimize:
        optimized = minimize(
            local_proxy, candidates[0], method='SLSQP',
            constraints={'type': 'ineq', 'fun': lambda point: RECEPTION_GUARD**2 - np.sum((vertices - point)**2, axis=1)},
            options={'maxiter': 24, 'ftol': 0.01},
        )
        if np.all(np.isfinite(optimized.x)) and eligible(optimized.x):
            candidates.append(optimized.x)
    candidates.sort(key=local_proxy)
    probes = sample_region(region, 5)
    if len(vertices) <= 8:
        probes = np.vstack((probes, vertices))
    else:
        probes = np.vstack((probes, vertices[np.linspace(0, len(vertices) - 1, 8).astype(int)]))
    best = None
    for candidate in candidates[:2]:
        worst_radius = 0.0
        positive_costs = []
        for source in probes:
            vector = source - candidate
            if np.linalg.norm(vector) <= 5:
                positive_costs.append(5.0)
                continue
            true_bearing = math.degrees(math.atan2(vector[1], vector[0]))
            for error in (-1.005, 0.0, 1.005):
                posterior = clipped_prediction(vertices, candidate, true_bearing + error)
                if len(posterior) == 0:
                    continue
                future_center, future_radius = minimum_circle(posterior)
                worst_radius = max(worst_radius, future_radius)
                residual_cells = max(1, math.ceil(future_radius / CLEAR_SUPPORT_RADIUS))
                positive_costs.append(np.linalg.norm(candidate - future_center) / 5 + 5 + (residual_cells - 1) * 11)
        immediate = float(np.linalg.norm(candidate - start) / 5 + 6)
        future_cost = float(np.mean(positive_costs)) if positive_costs else 1e6
        score = worst_radius + 0.04 * immediate
        if best is None or score < best[0]:
            best = (score, MeasurementPlan(
                candidate, worst_radius, future_cost, immediate,
                float(np.linalg.norm(vertices - candidate, axis=1).max()),
            ))
    return best[1] if best is not None else None


def source_proxy_hypotheses(region, positives, negatives, directional, sample_count=40):
    positions = sample_region(region, sample_count)
    angles = np.arange(24) * 2 * math.pi / 24
    headings = np.column_stack((np.cos(angles), np.sin(angles)))
    directions = np.vstack((np.zeros((1, 2)), headings)) if directional else np.zeros((1, 2))
    tiled_positions = np.repeat(positions, len(directions), axis=0)
    tiled_headings = np.tile(directions, (len(positions), 1))
    is_omni = np.linalg.norm(tiled_headings, axis=1) == 0
    lower = np.full(len(tiled_positions), 1000.0)
    upper = np.full(len(tiled_positions), 1500.0)
    valid = np.ones(len(tiled_positions), dtype=bool)
    for observation_position, _bearing in positives:
        vectors = observation_position - tiled_positions
        distances = np.linalg.norm(vectors, axis=1)
        valid &= is_omni | (np.sum(vectors * tiled_headings, axis=1) >= 0)
        lower = np.maximum(lower, distances)
    for observation_position in negatives:
        vectors = observation_position - tiled_positions
        visible = is_omni | (np.sum(vectors * tiled_headings, axis=1) >= 0)
        distances = np.linalg.norm(vectors, axis=1)
        upper = np.minimum(upper, np.where(visible, distances - 1e-6, 1500.0))
    valid &= lower <= upper
    tiled_positions = tiled_positions[valid]
    tiled_headings = tiled_headings[valid]
    lower, upper, is_omni = lower[valid], upper[valid], is_omni[valid]
    if len(lower) == 0:
        return None
    positions_output = np.repeat(tiled_positions, 2, axis=0)
    headings_output = np.repeat(tiled_headings, 2, axis=0)
    radii = np.column_stack((lower, upper)).ravel()
    prior = np.where(is_omni, 0.5, 0.5 / 24) if directional else np.ones(len(lower))
    weights = np.repeat(prior, 2)
    weights /= weights.sum()
    return positions_output, headings_output, radii, weights


def reception_proxy(hypotheses, position):
    if hypotheses is None:
        return 0.0
    source_positions, headings, radii, weights = hypotheses
    vectors = np.asarray(position) - source_positions
    visible = (np.linalg.norm(headings, axis=1) == 0) | (np.sum(vectors * headings, axis=1) >= 0)
    visible &= np.linalg.norm(vectors, axis=1) <= radii
    return float(weights[visible].sum())


def geometry_contains(region, position, tolerance=1e-6):
    return region.distance(Point(position)) <= tolerance


def initial_outer_region():
    return Polygon(initial_region())


def points_hull(points):
    return MultiPoint(points).convex_hull

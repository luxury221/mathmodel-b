from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from geometry import (
    bearing_halfplanes,
    diameter,
    directional_gap,
    disk_vertices,
    dual_ring_network,
    enumerate_circle,
    halfplane_status,
    minimum_circle,
    open_route,
    rectangle_clear_cover,
    ring,
    route_length,
    seven_network,
    triangular_network,
)
from scipy.optimize import minimize
from shapely.geometry import Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'offline_review'


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(name, value):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=serializable),
        encoding='utf-8',
    )


def audit_q1():
    generator = np.random.default_rng(20260911)
    records = []
    for count in range(1, 10):
        for _ in range(30):
            points = generator.normal(size=(count, 2)) * generator.uniform(1, 1800)
            center, radius = minimum_circle(points)
            _reference_center, reference_radius = enumerate_circle(points)
            maximum_distance = diameter(points)
            assert abs(radius - reference_radius) < 1e-6
            assert np.linalg.norm(points - center, axis=1).max() <= radius + 1e-8
            assert radius >= maximum_distance / 2 - 1e-7
            assert radius <= maximum_distance / math.sqrt(3) + 1e-7
            records.append(radius)
    degenerate_sets = (
        [[0, 0]], [[1, 1], [1, 1]], [[-20, 0], [0, 0], [20, 0]],
        [[-20, 0], [0, 1e-10], [20, 0]],
    )
    for points in degenerate_sets:
        center, radius = minimum_circle(points)
        assert abs(radius - enumerate_circle(points)[1]) < 1e-7
    triangle = np.array([[0.0, 0.0], [40.0, 0.0], [20.0, 20.0 * math.sqrt(3)]])
    center, radius = minimum_circle(triangle)
    classifications = {
        'empty': halfplane_status([[1, 0], [-1, 0]], [-1, -1]),
        'unbounded_single_wedge': halfplane_status(*bearing_halfplanes((0, 0), 0, 1)),
        'bounded_square': halfplane_status([[1, 0], [-1, 0], [0, 1], [0, -1]], [1] * 4),
        'bounded_segment': halfplane_status([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, 1, 0, 0]),
    }
    assert list(classifications.values()) == ['empty', 'unbounded', 'bounded', 'bounded']
    inclusion_checks = 0
    for bearing in [-180.0, -1.0, 0.0, 1.0, 90.0, 179.99, 359.99]:
        position = generator.uniform(-1800, 1800, 2)
        for error in [-1.0, 0.0, 1.0]:
            target = position + ring(1234.0, 1, bearing + error)[0]
            normals, bounds = bearing_halfplanes(position, bearing, 1.0)
            assert np.max(normals @ target - bounds) < 1e-9
            inclusion_checks += 1
    actual_angle = -0.0049
    displayed_angle = round(actual_angle + 1.0, 2)
    quantized_target = ring(1000.0, 1, actual_angle)[0]
    quantization = {}
    for error_bound in (1.0, 1.005):
        normals, bounds = bearing_halfplanes((0, 0), displayed_angle, error_bound)
        quantization[str(error_bound)] = bool(np.max(normals @ quantized_target - bounds) <= 1e-9)
    return {
        'random_cases': len(records), 'degenerate_cases': len(degenerate_sets),
        'bearing_inclusion_cases': inclusion_checks, 'halfplane_classifications': classifications,
        'diameter_40_equilateral_counterexample': {'diameter': diameter(triangle), 'mec_radius': radius},
        'rounding_after_error_stress_only': {
            'true_angle': actual_angle, 'displayed_angle': displayed_angle,
            'inclusion_by_error_bound': quantization,
        },
    }


def wedge_intersection_case(first_range, second_range, crossing_deg, errors=(0, 0)):
    first_position = np.array([-first_range, 0.0])
    second_position = -ring(second_range, 1, crossing_deg)[0]
    first_normals, first_bounds = bearing_halfplanes(first_position, errors[0], 1.0)
    second_normals, second_bounds = bearing_halfplanes(second_position, crossing_deg + errors[1], 1.0)
    normals = np.vstack((first_normals, second_normals))
    bounds = np.r_[first_bounds, second_bounds]
    status = halfplane_status(normals, bounds)
    if status != 'bounded':
        return {'status': status}
    vertices = []
    for indices in itertools.combinations(range(4), 2):
        selected = normals[list(indices)]
        if abs(np.linalg.det(selected)) < 1e-12:
            continue
        vertex = np.linalg.solve(selected, bounds[list(indices)])
        if np.all(normals @ vertex <= bounds + 1e-7):
            vertices.append(vertex)
    vertices = np.asarray(vertices)
    return {'status': status, 'diameter': diameter(vertices), 'mec_radius': minimum_circle(vertices)[1]}


def local_diameter(first_range, second_range, crossing_deg):
    angle = math.radians(crossing_deg)
    return 2 * math.tan(math.radians(1)) / abs(math.sin(angle)) * math.sqrt(
        first_range**2 + second_range**2 + 2 * first_range * second_range * abs(math.cos(angle))
    )


def audit_q2():
    errors = (-1.0, -0.5, 0.0, 0.5, 1.0)
    records = []
    unsafe = []
    for first_range, second_range in [(500, 500), (800, 800), (809, 809), (1000, 800), (1500, 1000)]:
        for crossing in (1.5, 3.0, 10.0, 30.0, 60.0, 90.0):
            local_radius = local_diameter(first_range, second_range, crossing) / 2
            for first_error, second_error in itertools.product(errors, repeat=2):
                exact = wedge_intersection_case(first_range, second_range, crossing, (first_error, second_error))
                record = {
                    'ranges': [first_range, second_range], 'crossing_deg': crossing,
                    'errors_deg': [first_error, second_error], 'local_radius': local_radius, **exact,
                }
                records.append(record)
                if local_radius <= 20 and exact.get('mec_radius', 0.0) > 20:
                    unsafe.append(record)
    save_json('q2_error_stress.json', records)
    angles = np.linspace(-1.0, 1.0, 2001)
    boundary = np.vstack([
        radius * np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
        for radius in (5.0001, 1500.0)
    ])
    candidates = {
        'line_lens_tip': np.array([752.5, math.sqrt(1000**2 - 747.5**2)]),
        'line_proxy_optimum': np.array([824.7137132538311, 572.7734526366644]),
    }
    candidate_results = {}
    for name, candidate in candidates.items():
        distances = np.linalg.norm(boundary - candidate, axis=1)
        worst_index = np.argmax(distances)
        first_distances = np.linalg.norm(boundary, axis=1)
        minimum_consistent_radii = np.maximum(1000.0, first_distances)
        reception_excess = distances - minimum_consistent_radii
        missed_index = np.argmax(reception_excess)
        candidate_results[name] = {
            'position': candidate, 'max_distance': float(distances[worst_index]),
            'conservative_domain_counterexample_source': boundary[worst_index],
            'inside_conservative_1000m_domain': bool(distances.max() <= 1000),
            'consistent_actual_missed_reception_example': {
                'source': boundary[missed_index], 'first_distance': first_distances[missed_index],
                'fixed_source_radius': minimum_consistent_radii[missed_index],
                'second_distance': distances[missed_index], 'excess_meters': reception_excess[missed_index],
            },
        }
    target_ranges = np.linspace(5, 1500, 5001)

    def proxy_objective(candidate):
        horizontal, vertical = candidate
        second_ranges = np.hypot(target_ranges - horizontal, vertical)
        cosine = np.abs(target_ranges - horizontal) / second_ranges
        return float(np.max(2 * math.tan(math.radians(1)) * second_ranges / vertical * np.sqrt(
            target_ranges**2 + second_ranges**2 + 2 * target_ranges * second_ranges * cosine
        )))

    extreme_sources = np.array([
        radius * np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        for radius, angle in itertools.product((5.0, 1500.0), (-1.0, 1.0))
    ])
    refinements = []
    for initial in ([752.5, 500.0], [820.0, 550.0], [700.0, 550.0]):
        result = minimize(
            proxy_objective, initial, method='SLSQP', bounds=[(5, 1500), (1, 900)],
            constraints={'type': 'ineq', 'fun': lambda point: 999.9**2 - np.sum((extreme_sources - point)**2, axis=1)},
            options={'maxiter': 500, 'ftol': 1e-9},
        )
        if np.linalg.norm(boundary - result.x, axis=1).max() <= 999.90001:
            refinements.append(result)
    best = min(refinements, key=lambda result: result.fun)
    return {
        'error_stress_cases': len(records),
        'unbounded_untruncated_wedge_cases': sum(record['status'] == 'unbounded' for record in records),
        'local_clearance_false_certificate_count': len(unsafe),
        'local_clearance_counterexamples': unsafe[:6],
        'full_wedge_reception_candidates': candidate_results,
        'safe_feasible_proxy_refinement_not_full_minimax': {
            'position': best.x, 'proxy_diameter': best.fun,
            'full_wedge_max_distance': float(np.linalg.norm(boundary - best.x, axis=1).max()),
            'optimizer_success': bool(best.success),
        },
    }


def coverage_certificate(points, receiving_radius=1000.0, target_radius=1806.0, resolution=256):
    receivers = [Polygon(disk_vertices(point, receiving_radius, resolution)) for point in points]
    cores = []
    for indices in itertools.combinations(range(len(points)), 3):
        core = Polygon(points[list(indices)])
        if core.area < 1e-9:
            continue
        for index in indices:
            core = core.intersection(receivers[index])
        if not core.is_empty:
            cores.append(core)
    coverage = unary_union(cores)
    target = Polygon(disk_vertices((0, 0), target_radius, 720, outer=True))
    residual = target.difference(coverage)
    analytic_center_patch = Polygon(disk_vertices((0, 0), 900.0, 128))
    witnesses = []
    residual_parts = list(residual.geoms) if hasattr(residual, 'geoms') else [residual]
    for part in sorted(residual_parts, key=lambda part: part.area, reverse=True)[:5]:
        if part.is_empty:
            continue
        representative = np.array(part.representative_point().coords[0])
        witnesses.append({
            'position': representative, 'source_radius': np.linalg.norm(representative),
            'exact_circle_gap_deg': directional_gap(representative, points, receiving_radius),
            'component_area': part.area,
        })
    return {
        'receiving_radius': receiving_radius, 'target_radius': target_radius,
        'polygon_sides': resolution, 'nonempty_triple_cores': len(cores),
        'residual_area': residual.area, 'residual_is_empty': residual.is_empty,
        'coverage_covers_target': coverage.covers(target),
        'coverage_is_valid': coverage.is_valid, 'residual_bounds': residual.bounds,
        'residual_witnesses': witnesses,
        'nominal_residual_after_analytic_center_patch_is_empty': (
            residual.difference(analytic_center_patch).is_empty if receiving_radius == 1000.0 else None
        ),
    }


def structured_positions(stations):
    positions = [ring(radius, 4096) for radius in (1800.0, 1799.999999, 1000.0)]
    for station in stations:
        positions.append(ring(1000.0, 512) + station)
        for offset in (1e-7, 0.001, 0.1, 1.0):
            positions.append(ring(offset, 32) + station)
    for first, second in itertools.combinations(stations, 2):
        distance = np.linalg.norm(second - first)
        if distance == 0 or distance > 2000:
            continue
        midpoint = (first + second) / 2
        tangent = np.array([first[1] - second[1], second[0] - first[0]]) / distance
        height = math.sqrt(max(0.0, 1000.0**2 - distance**2 / 4))
        for sign in (-1, 1):
            intersection = midpoint + sign * height * tangent
            positions.append(np.array([intersection]))
            positions.append(ring(0.001, 16) + intersection)
    positions = np.vstack(positions)
    return positions[np.linalg.norm(positions, axis=1) <= 1800.0 + 1e-10]


def angular_audit(stations, seed, random_count=100000):
    generator = np.random.default_rng(seed)
    radii = 1800.0 * np.sqrt(generator.random(random_count))
    angles = generator.uniform(0, 2 * math.pi, random_count)
    random_points = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
    structured = structured_positions(stations)
    maximum = 0.0
    failures = []
    for category, positions in [('random', random_points), ('structured', structured)]:
        for position in positions:
            gap = directional_gap(position, stations)
            maximum = max(maximum, gap)
            if gap > 180.0 + 1e-7:
                failures.append({'category': category, 'position': position, 'gap_deg': gap})
    return {
        'seed': seed, 'random_points': random_count, 'structured_points': len(structured),
        'max_angular_gap_deg': maximum, 'failure_count': len(failures), 'failure_examples': failures[:10],
    }


def audit_networks():
    seven = seven_network()
    worst_points = ring(1800, 6, 30)
    worst_distances = np.linalg.norm(worst_points[:, None] - seven[None, :], axis=2).min(axis=1)
    analytic = max(1125 / math.sqrt(3), math.sqrt(1800**2 + 1125**2 - math.sqrt(3) * 1800 * 1125))
    assert np.allclose(worst_distances, analytic, atol=1e-9)
    grid, triangles = triangular_network()
    assert len(grid) == 25
    assert all(diameter(triangle) <= 990 + 1e-9 for triangle in triangles)
    grid_route = open_route(grid)
    dual = dual_ring_network()
    inner_route = ring(1000, 8)
    outer_route = ring(1870, 12, 315)
    dual_explicit_route = np.vstack(([0.0, 0.0], inner_route, outer_route))
    certificate_cases = []
    for receiving_radius, target_radius, resolution in [
        (1000, 1800, 256), (1000, 1806, 256), (1000, 1806.25, 256),
        (1000, 1806.30, 256), (1000, 1806, 128), (1000, 1806, 512),
        (999.99, 1800, 256), (999.0, 1800, 256),
    ]:
        certificate_cases.append(coverage_certificate(dual, receiving_radius, target_radius, resolution))
    perturbed = dual.copy()
    perturbed[1:9] *= 1.00001
    perturbation_position = np.array([0.001, 0.0])
    perturbation_gap = directional_gap(perturbation_position, perturbed)
    assert perturbation_gap > 180
    summary = {
        'q3': {
            'analytic_cover_radius': analytic, 'worst_boundary_points': worst_points,
            'receiving_radius_margin': 1000 - analytic,
            'discovery_distance': 6750.0, 'full_scan_virtual_seconds': 2183.0,
            'directional_negative_control_gap_deg': directional_gap((1800, 0), seven),
        },
        'grid25': {
            'node_count': len(grid), 'intersecting_triangles': len(triangles),
            'triangle_edge_margin': 10.0, 'nn_2opt_distance': route_length(grid_route),
            'angular_audit': angular_audit(grid, 424242),
        },
        'dual21': {
            'node_count': len(dual), 'explicit_route_distance': route_length(dual_explicit_route),
            'nn_2opt_distance': route_length(open_route(dual)),
            'outer_polygon_inradius': 1870 * math.cos(math.radians(15)),
            'certificates': certificate_cases,
            'angular_audit': angular_audit(dual, 121212),
            'one_centimeter_inner_ring_outward_perturbation': {
                'source': perturbation_position, 'gap_deg': perturbation_gap,
                'heading_away_from_origin_deg': 0,
            },
        },
    }
    save_json('network_nodes.json', {'seven': seven, 'grid25': grid, 'dual21': dual,
                                    'grid25_route': grid_route, 'dual21_explicit_route': dual_explicit_route})
    return summary


def audit_fallback():
    halfwidth = 1500 * math.sin(math.radians(1.005))
    rectangle = np.array([[0, -halfwidth], [1500, -halfwidth], [1500, halfwidth], [0, halfwidth]])
    route, cell_radius = rectangle_clear_cover(rectangle, 0, (0, 0))
    assert cell_radius < 20 and len(route) == 108
    grid_cell_width = 1500 / 54
    worst_distance_to_first = math.hypot(1500 - grid_cell_width / 2, halfwidth / 2)
    in_grid_length = 2 * (1500 - grid_cell_width) + halfwidth
    per_source_seconds = (worst_distance_to_first + in_grid_length) / 5 + (len(route) - 1) * 3 + 5
    full_task_upper_bound = 35000 / 5 + 25 * 20 * 6 + 16 * ((3100 + 10000) / 5 + 107 * 3 + 5)
    adaptive_upper_bound = full_task_upper_bound + 16 * 4 * (10000 / 5 + 6)
    return {
        'error_bound_deg': 1.005, 'clearance_centers': len(route), 'cell_circumradius': cell_radius,
        'single_first_bearing_rectangle': {'length': 1500, 'width': 2 * halfwidth},
        'standalone_first_detection_fallback_seconds_bound': per_source_seconds,
        'full_task_loose_bound_seconds': full_task_upper_bound,
        'full_task_loose_bound_hours': full_task_upper_bound / 3600,
        'four_extra_measures_bound_hours': adaptive_upper_bound / 3600,
        'bound_assumptions': '25-node route <=35km; all action points have norm <5000m; within-cover path <=3100m; 16 sources; <=108 clear centers per source',
    }


def main():
    started = time.perf_counter()
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'shapely', 'pandas')}
    proposal = ROOT / 'research' / 'input' / 'B题_高上限建模与算法总方案.md'
    results = {'metadata': {
        'date': '2026-09-11', 'python': sys.version, 'platform': platform.platform(),
        'packages': versions, 'geos': shapely.geos_version_string,
        'proposal_sha256': hashlib.sha256(proposal.read_bytes()).hexdigest(),
        'upstream_commit': 'd9749c43e5668e56e645c2e06928efeb8b652b41',
        'official_simulator_calls': 0, 'kind': 'independent_offline_geometry_audit',
    }}
    for name, audit in [('q1', audit_q1), ('q2', audit_q2), ('networks', audit_networks), ('fallback', audit_fallback)]:
        section_started = time.perf_counter()
        results[name] = audit()
        results[name]['runtime_seconds'] = time.perf_counter() - section_started
        save_json('geometry_audit.json', results)
        print(name, 'completed', round(results[name]['runtime_seconds'], 3), 'seconds', flush=True)
    results['runtime_seconds'] = time.perf_counter() - started
    save_json('geometry_audit.json', results)


if __name__ == '__main__':
    main()

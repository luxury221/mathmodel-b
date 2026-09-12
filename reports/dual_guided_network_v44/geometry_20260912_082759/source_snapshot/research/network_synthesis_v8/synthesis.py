from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/offline_validation')]
from coverage_certificate import TriangleCoverage
from geometry import dual_ring_network, open_route, ring, route_length
from plan_geometry import hull_vertices, initial_outer_region
import cvxpy as cp
import numpy as np
import shapely


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def source_hashes():
    folders = (HERE, ROOT / 'research/joint_search_v5', ROOT / 'research/offline_validation')
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in folders for path in folder.glob('*.py')}


def initial_witnesses():
    boundary = hull_vertices(initial_outer_region())
    axis = np.arange(-1800, 1801, 300)
    interior = np.array([(horizontal, vertical) for horizontal in axis for vertical in axis
                         if 20**2 <= horizontal**2 + vertical**2 <= 1800**2])
    positions = np.vstack((interior, boundary, ring(20.1, 36), ring(999.8, 48), ring(1000.2, 48)))
    headings = ring(1, 36)
    return np.repeat(positions, len(headings), axis=0), np.tile(headings, (len(positions), 1))


def visibility_violation(receivers, positions, headings, radius=999.6, heading_margin=0.1):
    vectors = np.asarray(receivers)[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(vectors, axis=2) - radius
    projection = heading_margin - np.sum(vectors * headings[None, :, :], axis=2)
    return np.maximum(distance, projection)


def weakest_heading(position, receivers, radius=999.6):
    vectors = np.asarray(receivers) - position
    distances = np.linalg.norm(vectors, axis=1)
    if np.min(distances) < 1e-8:
        return None
    vectors = vectors[distances <= radius]
    if len(vectors) == 0:
        return np.array([1.0, 0.0])
    angles = np.sort(np.mod(np.arctan2(vectors[:, 1], vectors[:, 0]), 2 * np.pi))
    gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
    index = int(np.argmax(gaps))
    if gaps[index] < np.pi - 1e-8:
        return None
    angle = angles[index] + gaps[index] / 2
    return np.array([math.cos(angle), math.sin(angle)])


def certify(receivers):
    coverage = TriangleCoverage()
    coverage.observe_clear_absence([0, 0])
    for point in receivers:
        coverage.observe_absence(point)
    return coverage


def counterexamples(region, receivers, limit=32, receiving_radius=999.6):
    parts = list(region.geoms) if hasattr(region, 'geoms') else [region]
    candidates = []
    for part in sorted(parts, key=lambda part: part.area, reverse=True)[:limit]:
        if part.is_empty:
            continue
        representative = np.asarray(part.representative_point().coords[0])
        vertices = hull_vertices(part)
        chosen = vertices[np.linspace(0, len(vertices) - 1, min(6, len(vertices))).astype(int)]
        for position in np.vstack((representative, chosen)):
            heading = weakest_heading(position, receivers, radius=receiving_radius)
            if heading is not None:
                violation = float(visibility_violation(receivers, position[None, :], heading[None, :]).min())
                candidates.append((violation, position, heading))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for item in candidates:
        if any(np.linalg.norm(item[1] - previous[1]) < 0.01 and np.dot(item[2], previous[2]) > 0.99999 for previous in selected):
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def optimize_step(receivers, positions, headings, penalty, trust_radius=350.0):
    violations = visibility_violation(receivers, positions, headings)
    assignment = np.argmin(violations, axis=0)
    variables = cp.Variable(receivers.shape)
    variables.value = receivers / 1000
    radial_slack = cp.Variable(len(receivers), nonneg=True)
    heading_slack = cp.Variable(len(receivers), nonneg=True)
    constraints = [variables[0] == 0, cp.norm(variables, axis=1) <= 2.6,
                   cp.norm(variables - receivers / 1000, axis=1) <= trust_radius / 1000]
    assigned_counts = []
    for index in range(len(receivers)):
        mask = assignment == index
        assigned_counts.append(int(np.sum(mask)))
        if not mask.any():
            constraints.extend((radial_slack[index] == 0, heading_slack[index] == 0))
            continue
        witnesses = positions[mask]
        normals = headings[mask]
        vertices = hull_vertices(shapely.MultiPoint(witnesses).convex_hull)
        constraints.append(cp.norm(variables[index] - vertices / 1000, axis=1) <= 0.9996 + radial_slack[index])
        unique_normals, inverse = np.unique(normals, axis=0, return_inverse=True)
        bounds = np.full(len(unique_normals), -np.inf)
        np.maximum.at(bounds, inverse, np.sum(witnesses * normals, axis=1))
        constraints.append(unique_normals @ variables[index] >= (bounds + 0.1) / 1000 - heading_slack[index])
    itinerary = open_route(receivers[1:], [0, 0])
    available = set(range(1, len(receivers)))
    order = [0]
    for point in itinerary:
        index = min(available, key=lambda index: np.linalg.norm(receivers[index] - point))
        order.append(index)
        available.remove(index)
    costs = [cp.norm(variables[second] - variables[first]) for first, second in zip(order[:-1], order[1:])]
    objective = cp.sum(cp.hstack(costs)) + penalty * cp.sum(radial_slack + heading_slack)
    objective += 0.01 * cp.sum_squares(variables - receivers / 1000)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    started = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            problem.solve(solver='CLARABEL', max_iter=150, tol_feas=1e-8, canon_backend=cp.SCIPY_CANON_BACKEND)
    except (cp.error.SolverError, ValueError) as error:
        return None, {'status': 'solver_error', 'failure': str(error), 'seconds': time.perf_counter() - started}
    metadata = {'status': problem.status, 'seconds': time.perf_counter() - started,
                'warnings': [str(warning.message) for warning in captured], 'assigned_counts': assigned_counts}
    if problem.status not in ('optimal', 'optimal_inaccurate') or variables.value is None:
        return None, metadata
    proposal = np.asarray(variables.value) * 1000
    proposal[0] = [0, 0]
    metadata['slack_meters'] = float(np.sum(radial_slack.value + heading_slack.value) * 1000)
    metadata['witness_violation_meters'] = float(visibility_violation(proposal, positions, headings).min(axis=0).max())
    return proposal, metadata


def layouts(seed):
    original = dual_ring_network()
    generator = np.random.default_rng(seed)
    result = [('original21', original)]
    for name, removed in (('drop_inner20', [1]), ('drop_outer20', [9]), ('drop_two_inner19', [1, 5])):
        result.append((name, np.delete(original, removed, axis=0)))
    for repetition in range(3):
        points = original.copy()
        points[1:] += generator.normal(0, 100, points[1:].shape)
        if repetition:
            points = np.delete(points, [1] if repetition == 1 else [9], axis=0)
        result.append((f'perturbed_{repetition}_{len(points)}', points))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--layouts', nargs='+')
    parser.add_argument('--seed', type=int, default=120711200)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = source_hashes()
    save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                   'seed': args.seed, 'iterations': args.iterations, 'official_calls': 0,
                                   'goal': 'Q3<180 Q4<300 case-equal virtual seconds/source',
                                   'scope': 'Public-geometry network synthesis only, not source-case performance evidence'})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    initial_positions, initial_headings = initial_witnesses()
    save(output / 'initial_witnesses.json', {'positions': initial_positions.tolist(), 'headings': initial_headings.tolist()})
    baseline = dual_ring_network()
    baseline_coverage = certify(baseline)
    best = {'name': 'frozen_original21', 'points': baseline.tolist(), 'receivers': len(baseline), 'complete': baseline_coverage.empty,
            'route_meters': route_length(open_route(baseline[1:], [0, 0])), 'optical_origin_required': True}
    accepted = [best]
    records = []
    for name, receivers in layouts(args.seed):
        if args.layouts and name not in args.layouts:
            continue
        positions, headings = initial_positions.copy(), initial_headings.copy()
        for iteration in range(args.iterations):
            proposal, metadata = optimize_step(receivers, positions, headings, penalty=2000 + iteration * 1000)
            record = {'layout': name, 'iteration': iteration, 'receivers': len(receivers), **metadata}
            if proposal is None:
                records.append(record)
                save(output / 'iterations' / f'{name}_{iteration:02d}.json', record)
                break
            receivers = proposal
            record['points'] = receivers.tolist()
            record['route_meters'] = route_length(open_route(receivers[1:], [0, 0]))
            record['witness_count'] = len(positions)
            try:
                coverage = certify(receivers)
                record['complete'] = coverage.empty
                record['remaining_area'] = coverage.area
                record['geometry_error'] = None
                if coverage.empty:
                    accepted.append({'name': name, 'iteration': iteration, 'points': receivers.tolist(), 'receivers': len(receivers),
                                     'complete': True, 'route_meters': record['route_meters'], 'optical_origin_required': True})
                    if record['route_meters'] < best['route_meters']:
                        best = accepted[-1]
                else:
                    examples = counterexamples(coverage.region, receivers)
                    record['counterexamples'] = [{'violation': item[0], 'position': item[1].tolist(), 'heading': item[2].tolist()} for item in examples]
                    if examples:
                        positions = np.vstack((positions, [item[1] for item in examples]))
                        headings = np.vstack((headings, [item[2] for item in examples]))
            except (ValueError, shapely.errors.GEOSException) as error:
                record['complete'] = False
                record['geometry_error'] = str(error)
            records.append(record)
            save(output / 'iterations' / f'{name}_{iteration:02d}.json', record)
            save(output / 'accepted.json', accepted)
            save(output / 'best.json', best)
            print(name, iteration, 'complete', record['complete'], 'route', round(record['route_meters'], 2),
                  'area', round(record.get('remaining_area', -1), 4), 'slack', round(record.get('slack_meters', -1), 5), flush=True)
    save(output / 'summary.json', {'iterations': len(records), 'accepted_count': len(accepted), 'best': best,
                                 'source_hashes_unchanged': source_hashes() == frozen, 'official_calls': 0})


if __name__ == '__main__':
    main()

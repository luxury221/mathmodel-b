from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from synthesis import ROOT, HERE, certify, counterexamples, initial_witnesses, save, source_hashes
from geometry import dual_ring_network, open_route, ring, route_length
from ortools.sat.python import cp_model
import numpy as np


def candidate_pool():
    points = list(dual_ring_network())
    points.extend(point for radius in (750, 1000, 1250, 1500, 1750, 1900, 2050) for point in ring(radius, 16))
    axis = np.arange(-2000, 2001, 500)
    points.extend(np.array([horizontal, vertical], dtype=float) for horizontal in axis for vertical in axis
                  if horizontal**2 + vertical**2 <= 2100**2)
    unique = {}
    for point in points:
        unique.setdefault(tuple(np.round(point, 6)), np.asarray(point))
    return np.asarray(list(unique.values()))


def visible_sets(pool, positions, headings):
    matrix = []
    for point in pool:
        vectors = point - positions
        matrix.append((np.sum(vectors**2, axis=1) <= (999.9 + 1e-8)**2)
                      & (np.sum(vectors * headings, axis=1) >= -1e-8))
    matrix = np.asarray(matrix).T
    if not matrix.any(axis=1).all():
        raise ValueError('Candidate pool cannot cover a public-geometry witness')
    packed, indices = np.unique(np.packbits(matrix, axis=1), axis=0, return_index=True)
    return [np.flatnonzero(matrix[index]).tolist() for index in indices]


def solve(pool, groups, forbidden, seed, seconds, incumbent=None):
    count = len(pool)
    terminal = count
    model = cp_model.CpModel()
    selected = [model.NewBoolVar(f'scan_{index}') for index in range(count)]
    model.Add(selected[0] == 1)
    for group in groups:
        model.AddBoolOr([selected[index] for index in group])
    for bad in forbidden:
        members = set(bad)
        model.AddBoolOr([selected[index].Not() if index in members else selected[index] for index in range(1, count)])
    distances = np.linalg.norm(pool[:, None] - pool[None, :], axis=2)
    edge_pairs = set()
    for index in range(count):
        for other in np.argsort(distances[index])[1:31]:
            edge_pairs.add((index, int(other)))
            edge_pairs.add((int(other), index))
    original = dual_ring_network()
    original_indices = [int(np.argmin(np.linalg.norm(pool - point, axis=1))) for point in original]
    original_route = open_route(original[1:], [0, 0])
    original_order = [0] + [int(np.argmin(np.linalg.norm(pool - point, axis=1))) for point in original_route]
    for first, second in zip(original_order[:-1], original_order[1:]):
        edge_pairs.add((first, second))
        edge_pairs.add((second, first))
    if incumbent is not None:
        for first, second in zip(incumbent[:-1], incumbent[1:]):
            edge_pairs.add((first, second))
    arcs = [(index, index, selected[index].Not()) for index in range(1, count)]
    variables = {}
    costs = []
    for first, second in sorted(edge_pairs):
        if second == 0:
            continue
        active = model.NewBoolVar(f'edge_{first}_{second}')
        model.Add(active <= selected[first])
        model.Add(active <= selected[second])
        arcs.append((first, second, active))
        variables[first, second] = active
        costs.append(int(round(distances[first, second] * 100)) * active)
    for index in range(count):
        active = model.NewBoolVar(f'end_{index}')
        model.Add(active <= selected[index])
        variables[index, terminal] = active
        arcs.append((index, terminal, active))
    forced = model.NewBoolVar('terminal_to_origin')
    model.Add(forced == 1)
    arcs.append((terminal, 0, forced))
    model.AddCircuit(arcs)
    model.Minimize(sum(costs) + 30000 * sum(selected[1:]))
    hint_order = original_order if incumbent is None else incumbent
    hint_set = set(hint_order)
    hint_edges = set(zip(hint_order, hint_order[1:] + [terminal]))
    for index in range(count):
        model.AddHint(selected[index], int(index in hint_set))
    for edge, variable in variables.items():
        model.AddHint(variable, int(edge in hint_edges))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = seed
    started = time.perf_counter()
    status = solver.Solve(model)
    metadata = {'status': solver.StatusName(status), 'wall_seconds': time.perf_counter() - started,
                'candidate_sites': count, 'witness_groups': len(groups), 'arc_variables': len(variables)}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, metadata
    following = {first: second for (first, second), variable in variables.items() if solver.Value(variable)}
    order = [0]
    while following[order[-1]] != terminal:
        order.append(following[order[-1]])
        if len(order) > count:
            raise ValueError('Covering-tour solver returned a subtour')
    metadata.update({'objective_reference_meters': solver.ObjectiveValue() / 100,
                     'restricted_discrete_bound_meters': solver.BestObjectiveBound() / 100,
                     'order': order, 'receivers': len(order), 'route_meters': route_length(pool[order[1:]])})
    return order, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=20)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = source_hashes()
    save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                   'rounds': args.rounds, 'seconds_per_round': args.seconds, 'official_calls': 0,
                                   'scope': 'Public geometry only; objective uses ten residual channels as a fixed reference, not hidden case counts.'})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    pool = candidate_pool()
    positions, headings = initial_witnesses()
    save(output / 'candidate_pool.json', pool.tolist())
    forbidden = []
    accepted = []
    incumbent = None
    for iteration in range(args.rounds):
        groups = visible_sets(pool, positions, headings)
        order, record = solve(pool, groups, forbidden, 120811200 + iteration, args.seconds, incumbent)
        record['iteration'] = iteration
        if order is None:
            save(output / 'iterations' / f'{iteration:02d}.json', record)
            continue
        receivers = pool[order]
        coverage = certify(receivers)
        record.update({'points': receivers.tolist(), 'complete': coverage.empty, 'remaining_area': coverage.area})
        if coverage.empty:
            accepted.append(record)
            incumbent = order
        else:
            examples = counterexamples(coverage.region, receivers, limit=64, receiving_radius=999.9)
            record['counterexamples'] = [{'position': item[1].tolist(), 'heading': item[2].tolist()} for item in examples]
            if examples:
                positions = np.vstack((positions, [item[1] for item in examples]))
                headings = np.vstack((headings, [item[2] for item in examples]))
            forbidden.append(order)
        save(output / 'iterations' / f'{iteration:02d}.json', record)
        save(output / 'accepted.json', accepted)
        print(iteration, record['status'], 'points', len(order), 'complete', coverage.empty,
              'route', round(record['route_meters'], 2), 'area', round(coverage.area, 4), flush=True)
    save(output / 'summary.json', {'accepted_count': len(accepted), 'source_hashes_unchanged': source_hashes() == frozen,
                                  'best': min(accepted, key=lambda row: row['objective_reference_meters']) if accepted else None,
                                  'official_calls': 0})


if __name__ == '__main__':
    main()

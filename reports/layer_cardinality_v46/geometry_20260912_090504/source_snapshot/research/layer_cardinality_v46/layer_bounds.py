from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


def integer_certificate(groups, group_indices, point_indices, weights, denominator=100000000):
    if denominator <= 0 or len(group_indices) != len(weights):
        raise ValueError('Invalid rationalization configuration')
    if len(set(group_indices)) != len(group_indices) or len(set(point_indices)) != len(point_indices):
        raise ValueError('Duplicate input indices')
    loads = {int(point): 0 for point in point_indices}
    numerators = []
    for group_index, weight in zip(group_indices, weights):
        if not math.isfinite(float(weight)) or group_index < 0 or group_index >= len(groups):
            raise ValueError('Invalid weight or group index')
        group = set(groups[group_index])
        if not group or not group.issubset(loads):
            raise ValueError('Each weighted group must be nonempty and inside the interval')
        numerator = math.floor(max(0.0, float(weight)) * denominator)
        if numerator == 0:
            continue
        numerators.append([int(group_index), numerator])
        for point in group:
            loads[point] += numerator
    safe_denominator = max(denominator, max(loads.values(), default=0))
    total = sum(numerator for _group_index, numerator in numerators)
    return {
        'point_indices': sorted(loads), 'weighted_groups': numerators,
        'denominator': safe_denominator, 'total_numerator': total,
        'maximum_point_load': max(loads.values(), default=0),
        'fractional_lower': total / safe_denominator,
        'required_points': (total + safe_denominator - 1) // safe_denominator,
    }


def generate_interval_bounds(groups, layer_size=72, layer_count=9, seconds=3):
    all_sets = [set(group) for group in groups]
    cuts = []
    for first_layer in range(layer_count):
        for last_layer in range(first_layer, layer_count):
            points = list(range(1 + first_layer * layer_size, 1 + (last_layer + 1) * layer_size))
            point_set = set(points)
            eligible = [index for index, group in enumerate(all_sets) if group and group.issubset(point_set)]
            metadata = {'first_layer': first_layer, 'last_layer': last_layer, 'eligible_groups': len(eligible)}
            if not eligible:
                metadata.update({'lp_status': 'NO_INTERNAL_GROUP', 'lp_seconds': 0.0})
                metadata.update(integer_certificate(groups, [], points, []))
                cuts.append(metadata)
                continue
            point_to_column = {point: column for column, point in enumerate(points)}
            row_indices, column_indices = [], []
            for row_index, group_index in enumerate(eligible):
                for point in all_sets[group_index]:
                    row_indices.append(row_index)
                    column_indices.append(point_to_column[point])
            matrix = csr_matrix((-np.ones(len(row_indices)), (row_indices, column_indices)),
                                shape=(len(eligible), len(points)))
            started = time.perf_counter()
            result = linprog(np.ones(len(points)), A_ub=matrix, b_ub=-np.ones(len(eligible)),
                             bounds=(0, None), method='highs', options={'time_limit': seconds})
            metadata.update({'lp_status': int(result.status), 'lp_message': str(result.message),
                             'lp_seconds': time.perf_counter() - started})
            if result.success:
                metadata['lp_primal_value'] = float(result.fun)
                metadata.update(integer_certificate(groups, eligible, points, -result.ineqlin.marginals))
            else:
                metadata.update(integer_certificate(groups, [], points, []))
            cuts.append(metadata)
    return cuts


def disjoint_interval_lower(cuts, layer_count=9):
    prefix = [0] * (layer_count + 1)
    predecessors = [None] * (layer_count + 1)
    for stop in range(1, layer_count + 1):
        prefix[stop] = prefix[stop - 1]
        predecessors[stop] = (stop - 1, None)
        for index, cut in enumerate(cuts):
            first_layer, last_layer = cut['first_layer'], cut['last_layer']
            if not 0 <= first_layer <= last_layer < layer_count or cut['required_points'] < 0:
                raise ValueError('Invalid layer interval or lower bound')
            if last_layer + 1 == stop:
                candidate = prefix[first_layer] + cut['required_points']
                if candidate > prefix[stop]:
                    prefix[stop] = candidate
                    predecessors[stop] = (first_layer, index)
    selected = []
    cursor = layer_count
    while cursor:
        cursor, cut_index = predecessors[cursor]
        if cut_index is not None:
            selected.append(cut_index)
    return {'nonorigin_lower': prefix[-1], 'with_forced_origin_lower': prefix[-1] + 1,
            'disjoint_cut_indices': list(reversed(selected)), 'prefix_bounds': prefix}

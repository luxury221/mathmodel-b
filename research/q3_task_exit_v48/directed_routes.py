from __future__ import annotations

import numpy as np


def transition_costs(entries, exits, weights, start):
    entries = np.asarray(entries, dtype=float)
    if entries.ndim != 2 or entries.shape[1] != 2 or len(exits) != len(entries) or len(weights) != len(entries):
        raise ValueError('Each task needs one entry and an exit distribution')
    matrix = np.zeros((len(entries), len(entries)))
    for index, (points, probabilities) in enumerate(zip(exits, weights)):
        points, probabilities = np.asarray(points, dtype=float), np.asarray(probabilities, dtype=float)
        if points.shape != (len(probabilities), 2) or len(points) == 0:
            raise ValueError('Invalid task exit support')
        if not np.isfinite(points).all() or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
            raise ValueError('Exit distribution must be finite and nonnegative')
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
            raise ValueError('Exit weights must sum to one')
        matrix[index] = probabilities @ np.linalg.norm(points[:, None] - entries[None, :], axis=2)
    initial = np.linalg.norm(entries - np.asarray(start), axis=1)
    return initial, matrix


def route_cost(order, initial, matrix):
    if not order:
        return 0.0
    return float(initial[order[0]] + sum(matrix[first, second] for first, second in zip(order, order[1:])))


def directed_order(initial, matrix):
    initial, matrix = np.asarray(initial, dtype=float), np.asarray(matrix, dtype=float)
    count = len(initial)
    if matrix.shape != (count, count) or not np.isfinite(matrix).all() or not np.isfinite(initial).all():
        raise ValueError('Invalid directed routing matrix')
    if count == 0:
        return []
    if count <= 9:
        values = {(1 << index, index): (float(initial[index]), (index,)) for index in range(count)}
        for mask in range(1, 1 << count):
            for last in range(count):
                previous = values.get((mask, last))
                if previous is None:
                    continue
                cost, order = previous
                for following in range(count):
                    if mask & (1 << following):
                        continue
                    key = (mask | (1 << following), following)
                    candidate = (cost + float(matrix[last, following]), (*order, following))
                    if key not in values or candidate < values[key]:
                        values[key] = candidate
        return list(min(values[((1 << count) - 1, last)] for last in range(count))[1])
    best = None
    for first in np.argsort(initial, kind='stable')[:8]:
        order = [int(first)]
        remaining = set(range(count)) - set(order)
        while remaining:
            following = min(remaining, key=lambda index: (matrix[order[-1], index], index))
            order.append(following)
            remaining.remove(following)
        cost = route_cost(order, initial, matrix)
        for _iteration in range(8):
            improvement = None
            for left in range(count - 1):
                for right in range(left + 1, count):
                    candidate = order[:left] + order[left:right + 1][::-1] + order[right + 1:]
                    candidate_cost = route_cost(candidate, initial, matrix)
                    if candidate_cost < cost - 1e-7 and (improvement is None or candidate_cost < improvement[0]):
                        improvement = candidate_cost, candidate
            if improvement is None:
                break
            cost, order = improvement
        candidate = cost, tuple(order)
        if best is None or candidate < best:
            best = candidate
    return list(best[1])

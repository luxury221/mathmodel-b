from __future__ import annotations

import numpy as np


def bundle_order(entries, exits, start):
    entries = np.asarray(entries, dtype=float)
    exits = np.asarray(exits, dtype=float)
    start = np.asarray(start, dtype=float)
    count = len(entries)
    if entries.shape != exits.shape or entries.shape != (count, 2):
        raise ValueError('Bundle endpoints must have matching two-dimensional shapes')
    if count == 0:
        return []
    matrix = np.linalg.norm(exits[:, None] - entries[None, :], axis=2)
    initial = np.linalg.norm(entries - start, axis=1)
    if count <= 9:
        values = {(1 << index, index): (float(initial[index]), (index,)) for index in range(count)}
        for mask in range(1, 1 << count):
            for last in range(count):
                prior = values.get((mask, last))
                if prior is None:
                    continue
                cost, route = prior
                for following in range(count):
                    if mask & (1 << following):
                        continue
                    next_key = (mask | (1 << following), following)
                    candidate = (cost + float(matrix[last, following]), (*route, following))
                    if next_key not in values or candidate < values[next_key]:
                        values[next_key] = candidate
        return list(min(values[((1 << count) - 1, last)] for last in range(count))[1])

    def cost(route):
        return float(initial[route[0]] + matrix[route[:-1], route[1:]].sum())

    best = None
    for first in np.argsort(initial, kind='stable')[:8]:
        route = [int(first)]
        remaining = set(range(count)) - set(route)
        while remaining:
            following = min(remaining, key=lambda index: (matrix[route[-1], index], index))
            route.append(following)
            remaining.remove(following)
        previous_cost = cost(route)
        for _iteration in range(8):
            selected = None
            for left in range(count - 1):
                for right in range(left + 1, count):
                    candidate = route[:left] + route[left:right + 1][::-1] + route[right + 1:]
                    candidate_cost = cost(candidate)
                    if candidate_cost < previous_cost - 1e-7 and (selected is None or candidate_cost < selected[0]):
                        selected = candidate_cost, candidate
            if selected is None:
                break
            previous_cost, route = selected
        candidate = (previous_cost, tuple(route))
        if best is None or candidate < best:
            best = candidate
    return list(best[1])

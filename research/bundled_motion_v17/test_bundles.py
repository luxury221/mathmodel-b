import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bundles import evaluate, experiment
from bundle_routes import bundle_order


def route_cost(entries, exits, start, route):
    return float(np.linalg.norm(entries[route[0]] - start)
                 + sum(np.linalg.norm(exits[first] - entries[second]) for first, second in zip(route[:-1], route[1:])))


@pytest.mark.parametrize('count', [1, 4, 7])
def test_exact_route_agrees_with_exhaustive(count):
    generator = np.random.default_rng(291411200 + count)
    entries = generator.normal(size=(count, 2))
    exits = generator.normal(size=(count, 2))
    start = np.array([0.1, 0.3])
    route = bundle_order(entries, exits, start)
    best = min(route_cost(entries, exits, start, list(candidate)) for candidate in itertools.permutations(range(count)))
    assert abs(route_cost(entries, exits, start, route) - best) < 1e-9


def test_large_route_keeps_all_bundles():
    generator = np.random.default_rng(291411204)
    entries = generator.normal(size=(20, 2))
    exits = generator.normal(size=(20, 2))
    first = bundle_order(entries, exits, [0, 0])
    assert sorted(first) == list(range(20))
    assert first == bundle_order(entries, exits, [0, 0])


@pytest.mark.parametrize('mode', ['bundled', 'bundled_transit'])
def test_full_task_clearance_and_clock(mode):
    scene = experiment.make_scenes(291411300, 'unit_test', (10,))[6]
    row, _trace = evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    assert row['bundle_events']

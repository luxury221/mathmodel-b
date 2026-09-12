from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'research/joint_search_v4'),
                str(ROOT / 'research/offline_validation'), str(ROOT / 'research/policy_optimization_v3'),
                str(ROOT / 'research/q4_optimization_v2')]

import numpy as np
import math
from shapely.geometry import Point
from coverage_certificate import TriangleCoverage
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
from policy_v5 import ContinuousPolicy
from plan_geometry import disk_polygon, initial_outer_region, positive_update, geometry_contains
from probe_geometry import certified_v_probe


def test_full_network_and_optical_origin_prove_absence():
    coverage = TriangleCoverage()
    coverage.observe_clear_absence([0, 0])
    for point in dual_ring_network():
        coverage.observe_absence(point)
    assert coverage.empty


def test_single_negative_does_not_remove_disk():
    coverage = TriangleCoverage()
    coverage.observe_absence([0, 0])
    assert coverage.region.covers(Point([100, 0]))


def test_four_point_interior_handles_diagonal_intersection():
    coverage = TriangleCoverage()
    for point in ([500, 0], [0, 500], [-500, 0], [0, -500]):
        coverage.observe_absence(point)
    assert not coverage.region.covers(Point([0, 0]))


def test_negative_measurements_preserve_all_true_headings():
    generator = np.random.default_rng(105771912)
    for heading in (0, 45, 90, 179.9999999, 180, 225, 355, 359.9999999):
        source = np.array([850.0, -450.0])
        world = OfflineRuleWorld([Source(1, tuple(source), 1000.0, heading)], 105771912)
        coverage = TriangleCoverage()
        for receiver in generator.uniform(-2000, 2000, (35, 2)):
            observation = world.measure(receiver, 1)
            if observation['result'] == 'no_signal':
                coverage.observe_absence(receiver)
                assert coverage.region.covers(Point(source))


def test_origin_optical_sweep_clears_backside_sources():
    sources = [Source(channel, (0.001, -0.001), 1000.0, float(channel * 21)) for channel in range(1, 17)]
    world = OfflineRuleWorld(sources, 105771913)
    policy = ContinuousPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    result = policy.run()
    assert len(result['cleared_channels']) == 16
    assert not world.remaining
    assert result['virtual_seconds'] == world.virtual_seconds
    assert world.stats['measure_count'] == 0


def test_optical_certificate_only_cuts_small_disk():
    coverage = TriangleCoverage()
    coverage.observe_clear_absence([0, 0])
    assert not coverage.region.covers(Point([10, 0]))
    assert coverage.region.covers(Point([20, 0]))
    assert coverage.region.covers(Point([500, 0]))


def test_peripheral_first_network_has_valid_remainders():
    network = dual_ring_network()
    indices = [0, 15, 16, 17, 7, 8, 1, 2, 3, 4, 5, 13, 11, 9, 14, 20, 10, 12, 6, 18, 19]
    coverage = TriangleCoverage()
    coverage.observe_clear_absence([0, 0])
    for index in indices:
        coverage.observe_absence(network[index])
        assert coverage.region.is_valid
    assert coverage.empty


def test_network_certificate_survives_dynamic_observation_order():
    network = dual_ring_network()
    extra = [[-545.9267329767649, -1709.2843581487134],
             [-1435.9966129700226, -1056.3919201822164]]
    generator = np.random.default_rng(106991200)
    for _repeat in range(12):
        points = np.vstack((network, extra))
        generator.shuffle(points)
        coverage = TriangleCoverage()
        coverage.observe_clear_absence([0, 0])
        for point in points:
            coverage.observe_absence(point)
            assert coverage.region.is_valid
        assert coverage.empty


def test_v_probe_range_and_mirrored_reception_at_error_limits():
    negative_first = 0
    checked = 0
    for bearing in (0.0, 90.0, 179.999, 359.99):
        for error in (-1.005, 0.0, 1.005):
            angle = math.radians(bearing + error)
            direction = np.array([math.cos(angle), math.sin(angle)])
            for distance in (5.001, 19.999, 20.001, 999.999, 1000.001, 1499.999):
                position = distance * direction
                region = positive_update(initial_outer_region(), [0, 0], bearing)
                region = region.difference(disk_polygon([0, 0], 0.9 * distance))
                assert geometry_contains(region, position)
                probe = certified_v_probe(region, [0, 0], bearing)
                assert probe is not None
                assert probe.maximum_squared_range_change <= 1e-7
                assert np.all(np.linalg.norm(probe.candidates - position, axis=1) <= distance + 1e-7)
                for relative_heading in (90.000001, 120.0, 180.0, 240.0, 269.999999):
                    heading = (bearing + error + relative_heading) % 360
                    source = Source(1, tuple(position), 1000.0 if distance < 1000 else 1500.0, heading)
                    world = OfflineRuleWorld([source], 107111200)
                    assert world.measure([0, 0], 1)['result'] != 'no_signal'
                    responses = [world.measure(point, 1)['result'] for point in probe.candidates]
                    assert 'direction' in responses or 'near' in responses
                    if responses[0] == 'no_signal':
                        negative_first += 1
                        assert responses[1] != 'no_signal'
                    checked += 1
    assert checked == 360
    assert negative_first > 0

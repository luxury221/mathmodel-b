from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import ShapedProbePolicy
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import disk_polygon, geometry_contains, initial_outer_region, positive_update
from shaped_geometry import certified_shaped_probe


@pytest.mark.parametrize('fraction,aspect', [(0.7, 0.35), (0.85, 0.35), (0.95, 0.2), (0.95, 1.0)])
def test_mirrored_reception_and_range_at_cone_limits(fraction, aspect):
    negative_first = 0
    for bearing in (0.0, 90.0, 359.99):
        for error in (-1.005, 0.0, 1.005):
            angle = math.radians(bearing + error)
            direction = np.array([math.cos(angle), math.sin(angle)])
            for distance in (5.001, 20.001, 999.999, 1499.999):
                source_position = distance * direction
                region = positive_update(initial_outer_region(), [0, 0], bearing)
                region = region.difference(disk_polygon([0, 0], 0.9 * distance))
                probe = certified_shaped_probe(region, [0, 0], bearing, fraction, aspect)
                assert probe is not None
                assert probe.maximum_squared_range_change < 0
                assert probe.minimum_forward_gap > 0
                assert probe.minimum_cone_slack > 0
                assert geometry_contains(region, source_position)
                assert np.all(np.linalg.norm(probe.candidates - source_position, axis=1) <= distance + 1e-7)
                for relative_heading in (90.000001, 120.0, 180.0, 240.0, 269.999999):
                    heading = (bearing + error + relative_heading) % 360
                    world = OfflineRuleWorld([Source(1, tuple(source_position), max(1000, distance), heading)], 121411201)
                    assert world.measure([0, 0], 1)['result'] != 'no_signal'
                    responses = [world.measure(point, 1)['result'] for point in probe.candidates]
                    assert any(response in ('direction', 'near') for response in responses)
                    if responses[0] == 'no_signal':
                        negative_first += 1
                        assert responses[1] != 'no_signal'
    assert negative_first > 0


def test_invalid_or_uncertified_geometry_is_not_accepted():
    region = positive_update(initial_outer_region(), [0, 0], 0).difference(disk_polygon([0, 0], 100))
    assert certified_shaped_probe(region, [0, 0], 0, 0.85, 0.001) is None
    with pytest.raises(ValueError):
        certified_shaped_probe(region, [0, 0], 0, 1.1, 0.35)


def test_action_proposals_do_not_mutate_evidence_or_time():
    world = OfflineRuleWorld([Source(1, (850, 100), 1000, 180)], 121411202)
    policy = ShapedProbePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    policy.clear(1, [0, 0])
    policy.measure(1, [0, 0])
    state = policy.states[1]
    before = (state.region.wkb, state.revision, world.virtual_seconds, len(world.trace))
    action = policy.next_target_action(state)
    assert action[0] == 'v_probe'
    assert (state.region.wkb, state.revision, world.virtual_seconds, len(world.trace)) == before


def test_empty_scene_requires_real_complete_coverage():
    world = OfflineRuleWorld([], 121411203)
    policy = ShapedProbePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode='narrow85')
    result = policy.run()
    assert policy.coverage.empty
    assert result['virtual_seconds'] == world.virtual_seconds
    assert set(result['declared_absent']) == set(range(1, 21))
    assert world.stats['measure_count'] > 20

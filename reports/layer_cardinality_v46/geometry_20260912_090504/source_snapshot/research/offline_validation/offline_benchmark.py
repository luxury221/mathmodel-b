from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from geometry import (
    dual_ring_network,
    initial_region,
    minimum_circle,
    open_route,
    rectangle_clear_cover,
    seven_network,
    triangular_network,
    update_bearing_polygon,
)
from shapely.geometry import Point, Polygon

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'offline_review'


@dataclass(frozen=True)
class Source:
    channel: int
    position: tuple[float, float]
    radius: float
    heading_deg: float | None


@dataclass(frozen=True)
class ObservationPort:
    measure: Callable
    clear: Callable


class OfflineRuleWorld:
    def __init__(self, sources, seed, error_mode='spatial_uniform'):
        self.sources = {source.channel: source for source in sources}
        self.remaining = set(self.sources)
        self.position = np.zeros(2)
        self.receiver_channel = 1
        self.virtual_seconds = 0.0
        self.seed = seed
        self.error_mode = error_mode
        self.started = time.perf_counter()
        self.stats = {
            'move_meters': 0.0, 'measure_count': 0, 'switch_count': 0,
            'failed_clear_count': 0, 'successful_clear_count': 0,
        }
        self.trace = []

    def port(self):
        return ObservationPort(self.measure, self.clear)

    def move(self, position, channel):
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)) or np.any(np.abs(position) > 2e6):
            raise ValueError('Invalid position')
        if channel not in range(1, 21):
            raise ValueError('Invalid channel')
        distance = float(np.linalg.norm(position - self.position))
        self.position = position.copy()
        self.virtual_seconds += distance / 5.0
        self.stats['move_meters'] += distance

    def bearing_error(self, channel):
        payload = f'{self.seed}|{channel}|{self.position[0].hex()}|{self.position[1].hex()}'.encode()
        integer = int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), 'big')
        if self.error_mode == 'constant_extreme':
            return 1.0 if channel % 2 else -1.0
        if self.error_mode == 'spatial_extreme':
            return 1.0 if integer % 2 else -1.0
        return 2.0 * integer / (2**64 - 1) - 1.0

    def record(self, action, channel, response):
        self.trace.append({
            'action': action, 'channel': channel, 'position': self.position.tolist(),
            'response': response, 'virtual_seconds': self.virtual_seconds,
        })
        if self.virtual_seconds > 100 * 3600 or len(self.trace) > 5000:
            raise RuntimeError('Offline virtual/action budget exceeded')
        if time.perf_counter() - self.started > 1200:
            raise RuntimeError('Offline wall-clock budget exceeded')
        return response

    def measure(self, position, channel):
        self.move(position, channel)
        self.virtual_seconds += 5.0
        self.stats['measure_count'] += 1
        if channel != self.receiver_channel:
            self.virtual_seconds += 1.0
            self.stats['switch_count'] += 1
            self.receiver_channel = channel
        result = {'result': 'no_signal'}
        if channel in self.remaining:
            source = self.sources[channel]
            from_source = self.position - np.array(source.position)
            distance = float(np.linalg.norm(from_source))
            visible = distance <= source.radius
            if source.heading_deg is not None:
                heading = math.radians(source.heading_deg)
                direction = np.array([math.cos(heading), math.sin(heading)])
                visible = visible and float(np.dot(direction, from_source)) >= 0.0
            if visible and distance <= 5.0:
                result = {'result': 'near'}
            elif visible:
                angle = math.degrees(math.atan2(-from_source[1], -from_source[0]))
                displayed = round((angle + self.bearing_error(channel)) % 360.0, 2) % 360.0
                result = {'result': 'direction', 'bearing_deg': displayed}
        return self.record('measure', channel, result)

    def clear(self, position, channel):
        self.move(position, channel)
        success = channel in self.remaining and np.linalg.norm(
            self.position - np.array(self.sources[channel].position)
        ) <= 20.0
        if success:
            self.remaining.remove(channel)
            self.virtual_seconds += 5.0
            self.stats['successful_clear_count'] += 1
        else:
            self.virtual_seconds += 3.0
            self.stats['failed_clear_count'] += 1
        return self.record('clear', channel, {'result': 'success' if success else 'no_target_in_range'})


def next_measurement(region, last_positive, current, visited):
    center, radius = minimum_circle(region)
    radial = center - np.asarray(last_positive)
    norm = np.linalg.norm(radial)
    radial = radial / norm if norm > 1e-9 else np.array([1.0, 0.0])
    tangent = np.array([-radial[1], radial[0]])
    offset = min(550.0, max(60.0, 0.7 * radius))
    candidates = [center + sign * offset * tangent for sign in (-1, 1)]
    candidates += [center + sign * 0.6 * offset * tangent for sign in (-1, 1)]
    if radius <= 80:
        candidates.append(center)
    eligible = [
        candidate for candidate in candidates
        if all(np.linalg.norm(candidate - previous) > 1e-6 for previous in visited)
        and np.linalg.norm(region - candidate, axis=1).max() < 999.9
    ]
    if not eligible:
        eligible = [
            center + scale * tangent for scale in (0.0, 25.0, -25.0, 50.0)
            if all(np.linalg.norm(center + scale * tangent - previous) > 1e-6 for previous in visited)
        ]
    if not eligible:
        return None

    def score(candidate):
        first_vectors = region - last_positive
        second_vectors = region - candidate
        first_ranges = np.linalg.norm(first_vectors, axis=1)
        second_ranges = np.linalg.norm(second_vectors, axis=1)
        product = np.maximum(first_ranges * second_ranges, 1e-9)
        sine = np.abs(first_vectors[:, 0] * second_vectors[:, 1] - first_vectors[:, 1] * second_vectors[:, 0]) / product
        cosine = np.abs(np.sum(first_vectors * second_vectors, axis=1)) / product
        approximate_radius = math.tan(math.radians(1.005)) * np.sqrt(
            first_ranges**2 + second_ranges**2 + 2 * product * cosine
        ) / np.maximum(sine, 1e-6)
        return float(approximate_radius.max() + 0.005 * np.linalg.norm(candidate - current))

    return min(eligible, key=score)


def run_policy(port, stations, adaptive):
    position = np.zeros(2)
    unknown = set(range(1, 21))
    detections = {}
    route = open_route(stations)
    visited_stations = 0
    safe_circle_clears = 0
    fallback_clears = 0
    adaptive_measures = 0
    region_snapshots = []
    for station_index, station in enumerate(route):
        channels = sorted(unknown, reverse=bool(station_index % 2))
        for channel in channels:
            observation = port.measure(station, channel)
            position = station.copy()
            if observation['result'] == 'no_signal':
                continue
            record = {'position': station.copy(), 'observation': observation}
            if observation['result'] == 'direction':
                record['region'] = update_bearing_polygon(initial_region(), station, observation['bearing_deg'])
                region_snapshots.append((channel, record['region'].copy()))
            detections[channel] = record
            unknown.remove(channel)
            if len(detections) == 16:
                break
        visited_stations += 1
        if len(detections) == 16 or not unknown:
            break
    assert visited_stations == len(route) or len(detections) == 16 or not unknown
    absent = sorted(unknown)
    unresolved = set(detections)
    while unresolved:
        def destination(channel):
            record = detections[channel]
            if record['observation']['result'] == 'near':
                return record['position']
            return minimum_circle(record['region'])[0]

        channel = min(sorted(unresolved), key=lambda current_channel: np.linalg.norm(destination(current_channel) - position))
        record = detections[channel]
        observation = record['observation']
        if observation['result'] == 'near':
            position = record['position'].copy()
            assert port.clear(position, channel)['result'] == 'success'
            safe_circle_clears += 1
            unresolved.remove(channel)
            continue
        region = record['region'].copy()
        last_positive = record['position'].copy()
        measurement_positions = [last_positive.copy()]
        cleared = False
        for step in range(5 if adaptive else 1):
            center, radius = minimum_circle(region)
            if radius <= 19.95:
                position = center
                assert port.clear(position, channel)['result'] == 'success'
                safe_circle_clears += 1
                cleared = True
                break
            if not adaptive or step == 4:
                break
            candidate = next_measurement(region, last_positive, position, measurement_positions)
            if candidate is None:
                break
            observation_next = port.measure(candidate, channel)
            position = candidate.copy()
            measurement_positions.append(candidate.copy())
            adaptive_measures += 1
            if observation_next['result'] == 'near':
                assert port.clear(position, channel)['result'] == 'success'
                safe_circle_clears += 1
                cleared = True
                break
            if observation_next['result'] == 'direction':
                region = update_bearing_polygon(region, candidate, observation_next['bearing_deg'])
                if len(region) == 0:
                    raise RuntimeError('Inconsistent positive-bearing region')
                last_positive = candidate.copy()
                region_snapshots.append((channel, region.copy()))
        if not cleared:
            clearance_route, cell_radius = rectangle_clear_cover(region, observation['bearing_deg'], position)
            assert cell_radius < 20.0
            for candidate in clearance_route:
                response = port.clear(candidate, channel)
                position = candidate.copy()
                if response['result'] == 'success':
                    fallback_clears += 1
                    cleared = True
                    break
        if not cleared:
            raise RuntimeError('Exhausted a certified rectangular clearance cover')
        unresolved.remove(channel)
    return {
        'declared_absent': absent, 'detected_channels': sorted(detections),
        'visited_stations': visited_stations, 'safe_circle_clears': safe_circle_clears,
        'fallback_clears': fallback_clears, 'adaptive_measure_count': adaptive_measures,
        'region_snapshots': region_snapshots,
    }


def protocol_checks():
    checks = []
    world = OfflineRuleWorld([Source(2, (300.0, 0.0), 1000, None)], 123)
    world.measure((300.0, 400.0), 1)
    assert world.virtual_seconds == 105
    world.measure((300.0, 400.0), 2)
    assert world.virtual_seconds == 111
    world.clear((300.0, 0.0), 3)
    assert world.virtual_seconds == 194 and world.receiver_channel == 2
    assert world.measure((300.0, 0.0), 2)['result'] == 'near'
    assert world.virtual_seconds == 199
    checks.extend(['movement_5_mps', 'measure_5_seconds', 'switch_1_second', 'failed_clear_3_seconds', 'clear_preserves_channel'])
    world = OfflineRuleWorld([Source(1, (0.0, 0.0), 1000, 0.0)], 3)
    assert world.measure((-1, 0), 1)['result'] == 'no_signal'
    assert world.measure((5, 0), 1)['result'] == 'near'
    assert world.clear((-20, 0), 1)['result'] == 'success'
    assert world.clear((-20, 0), 1)['result'] == 'no_target_in_range'
    checks.extend(['backside_near_is_no_signal', 'near_inclusive_5m', 'clear_inclusive_20m_independent_heading', 'no_double_clear'])
    world = OfflineRuleWorld([Source(1, (500.0, 500.0), 1000, None)], 7)
    first = world.measure((0, 0), 1)
    assert first == world.measure((0, 0), 1)
    world.measure((12, 34), 1)
    assert first == world.measure((0, 0), 1)
    checks.extend(['fixed_location_fixed_error', 'returning_to_location_fixed_error'])
    world = OfflineRuleWorld([Source(1, (0.0, 0.0), 1000, 0.0)], 4)
    assert world.measure((1000, 0), 1)['result'] == 'direction'
    assert world.measure((1000.001, 0), 1)['result'] == 'no_signal'
    assert world.measure((0, 500), 1)['result'] == 'direction'
    checks.extend(['inclusive_1000m_range', 'outside_range_no_signal', 'closed_directional_halfplane'])
    return {'passed': len(checks), 'checks': checks}


def generate_scene(problem, count, profile, seed):
    generator = np.random.default_rng(seed)
    channels = generator.choice(np.arange(1, 21), count, replace=False)
    angles = generator.uniform(0, 2 * math.pi, count)
    if profile == 'random':
        radii = 1800 * np.sqrt(generator.random(count))
        positions = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        receiver_radii = generator.uniform(1000, 1500, count)
        error_mode = 'spatial_uniform'
    elif profile == 'boundary':
        angles = np.arange(count) * 2 * math.pi / count + generator.uniform(0, 2 * math.pi)
        positions = 1799.999999 * np.column_stack((np.cos(angles), np.sin(angles)))
        receiver_radii = np.full(count, 1000.0)
        error_mode = 'constant_extreme'
    elif profile == 'clustered':
        positions = np.array([1200.0, -600.0]) + generator.normal(0, 20, (count, 2))
        receiver_radii = np.full(count, 1000.0)
        error_mode = 'spatial_extreme'
    else:
        positions = generator.normal(0, 0.002, (count, 2))
        receiver_radii = np.full(count, 1000.0)
        error_mode = 'constant_extreme'
    sources = []
    for index, channel in enumerate(channels):
        heading = None
        if problem == 'q4':
            if profile == 'random':
                heading = float(generator.uniform(0, 360)) if index % 3 != 0 else None
            elif profile == 'boundary':
                heading = float(np.rad2deg(angles[index])) if index != 0 else None
            else:
                heading = float(generator.uniform(0, 360))
        sources.append(Source(int(channel), tuple(positions[index]), float(receiver_radii[index]), heading))
    return sources, error_mode


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    protocol = protocol_checks()
    print('Protocol checks:', protocol['passed'], flush=True)
    grid, _ = triangular_network()
    methods = {
        'q3': [('q3_grid7_cover', seven_network(), False), ('q3_grid7_active', seven_network(), True)],
        'q4': [('q4_grid25_cover', grid, False), ('q4_grid25_active', grid, True),
               ('q4_dual21_cover', dual_ring_network(), False), ('q4_dual21_active', dual_ring_network(), True)],
    }
    records = []
    all_scenes = []
    worst_traces = {}
    region_checks = 0
    for problem_index, problem in enumerate(('q3', 'q4')):
        for profile_index, profile in enumerate(('random', 'boundary', 'clustered', 'near_origin')):
            for count in range(10, 17):
                for repetition in range(2):
                    seed = 20260911 + problem_index * 10000 + profile_index * 1000 + count * 10 + repetition
                    sources, error_mode = generate_scene(problem, count, profile, seed)
                    scene_id = f'{problem}_{profile}_{count}_{repetition}'
                    all_scenes.append({
                        'scene_id': scene_id, 'seed': seed, 'problem': problem, 'profile': profile,
                        'error_mode': error_mode, 'sources': [source.__dict__ for source in sources],
                    })
                    for method, stations, adaptive in methods[problem]:
                        world = OfflineRuleWorld(sources, seed, error_mode)
                        run_started = time.perf_counter()
                        failure = None
                        policy_result = {}
                        try:
                            policy_result = run_policy(world.port(), stations, adaptive)
                            for channel, region in policy_result.pop('region_snapshots'):
                                distance = Polygon(region).distance(Point(world.sources[channel].position))
                                assert distance < 1e-6, f'True source excluded by feasible set: {distance}'
                                region_checks += 1
                            assert not world.remaining, f'Undiscovered/uncleared channels: {sorted(world.remaining)}'
                            assert not set(policy_result['declared_absent']) & set(world.sources)
                        except (AssertionError, RuntimeError, ValueError) as exception:
                            failure = f'{type(exception).__name__}: {exception}'
                            policy_result.pop('region_snapshots', None)
                        cleared = len(sources) - len(world.remaining)
                        record = {
                            'scene_id': scene_id, 'seed': seed, 'problem': problem, 'profile': profile,
                            'method': method, 'source_count': count, 'cleared_count': cleared,
                            'cleared_fraction': cleared / count, 'all_cleared': cleared == count and failure is None,
                            'virtual_seconds': world.virtual_seconds,
                            'seconds_per_cleared': world.virtual_seconds / cleared if cleared else None,
                            'wall_seconds': time.perf_counter() - run_started,
                            'action_count': len(world.trace), 'failure': failure,
                            **world.stats, **policy_result,
                        }
                        records.append(record)
                        if method not in worst_traces or record['seconds_per_cleared'] > worst_traces[method]['record']['seconds_per_cleared']:
                            worst_traces[method] = {'record': record, 'trace': world.trace}
                        if failure:
                            print('FAIL', scene_id, method, failure, flush=True)
            print(problem, profile, 'completed; cumulative runs', len(records), flush=True)
            pd.DataFrame(records).to_csv(OUTPUT / 'offline_runs.csv', index=False, encoding='utf-8-sig')
    frame = pd.DataFrame(records)
    frame.groupby(['method', 'profile']).agg(
        runs=('scene_id', 'size'), all_cleared=('all_cleared', 'sum'),
        mean_seconds_per_source=('seconds_per_cleared', 'mean'),
        worst_seconds_per_source=('seconds_per_cleared', 'max'),
    ).to_csv(OUTPUT / 'offline_by_profile.csv', encoding='utf-8-sig')
    summary = []
    for method, group in frame.groupby('method'):
        summary.append({
            'method': method, 'runs': len(group), 'all_cleared_runs': int(group.all_cleared.sum()),
            'mean_seconds_per_source': float(group.seconds_per_cleared.mean()),
            'median_seconds_per_source': float(group.seconds_per_cleared.median()),
            'p90_seconds_per_source': float(group.seconds_per_cleared.quantile(0.9)),
            'worst_seconds_per_source': float(group.seconds_per_cleared.max()),
            'mean_total_virtual_seconds': float(group.virtual_seconds.mean()),
            'worst_total_virtual_seconds': float(group.virtual_seconds.max()),
            'mean_move_meters': float(group.move_meters.mean()),
            'mean_measure_count': float(group.measure_count.mean()),
            'mean_failed_clear_count': float(group.failed_clear_count.mean()),
            'worst_wall_seconds': float(group.wall_seconds.max()),
            'max_action_count': int(group.action_count.max()),
        })
    paired = []
    for first_method, second_method in [
        ('q3_grid7_cover', 'q3_grid7_active'),
        ('q4_grid25_cover', 'q4_grid25_active'),
        ('q4_dual21_cover', 'q4_dual21_active'),
        ('q4_grid25_cover', 'q4_dual21_cover'),
        ('q4_grid25_active', 'q4_dual21_active'),
    ]:
        first = frame[frame.method == first_method].set_index('scene_id').seconds_per_cleared
        second = frame[frame.method == second_method].set_index('scene_id').seconds_per_cleared
        differences = first - second
        generator = np.random.default_rng(880011)
        means = generator.choice(differences.to_numpy(), (5000, len(differences)), replace=True).mean(axis=1)
        paired.append({
            'baseline': first_method, 'candidate': second_method, 'paired_scenes': len(differences),
            'candidate_faster_scenes': int((differences > 1e-6).sum()),
            'candidate_slower_scenes': int((differences < -1e-6).sum()),
            'mean_saved_seconds_per_source': float(differences.mean()),
            'relative_mean_reduction': float(1 - second.mean() / first.mean()),
            'synthetic_scene_bootstrap_interval_95': np.quantile(means, [0.025, 0.975]).tolist(),
        })
    failures = frame[~frame.all_cleared].to_dict('records')
    result = {
        'kind': 'synthetic_offline_rule_replacement_not_official_simulator',
        'official_simulator_calls': 0, 'scenario_count': len(all_scenes), 'run_count': len(records),
        'all_cleared_runs': int(frame.all_cleared.sum()), 'protocol_checks': protocol,
        'true_source_feasible_region_checks': region_checks, 'summary': summary,
        'paired_comparisons': paired, 'failures': failures,
        'runtime_seconds': time.perf_counter() - started,
        'limitations': [
            'Not the full proposed minimax/particles/POMDP/Clear-as-Sensing planner',
            'Errors generated by declared synthetic spatial functions; not official noise data',
            'Clustered and near-origin q4 profiles are all-directional stress cases',
            'Synthetic scenes are stratified, not sampled from an official distribution',
            'No HTTP latency, official score, official formal test, or simulator internals used',
            'Bootstrap interval only describes this synthetic benchmark, not competition outcomes',
        ],
    }
    for name, content in [('offline_summary.json', result), ('offline_scenes.json', all_scenes), ('worst_traces.json', worst_traces)]:
        (OUTPUT / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

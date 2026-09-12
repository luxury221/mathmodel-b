from __future__ import annotations

import csv
import hashlib
import json
import time

from geometry import dual_ring_network, seven_network, triangular_network
from offline_benchmark import OUTPUT, OfflineRuleWorld, Source, run_policy


def main():
    started = time.perf_counter()
    scenes = json.loads((OUTPUT / 'offline_scenes.json').read_text(encoding='utf-8'))
    scenes = {scene['scene_id']: scene for scene in scenes}
    with (OUTPUT / 'offline_runs.csv').open(encoding='utf-8-sig', newline='') as stream:
        records = list(csv.DictReader(stream))
    stations = {
        'grid7': seven_network(), 'grid25': triangular_network()[0], 'dual21': dual_ring_network(),
    }
    compared = 0
    for record in records:
        scene = scenes[record['scene_id']]
        sources = [Source(**source) for source in scene['sources']]
        world = OfflineRuleWorld(sources, scene['seed'], scene['error_mode'])
        _, network_name, strategy = record['method'].split('_')
        result = run_policy(world.port(), stations[network_name], strategy == 'active')
        assert not world.remaining
        assert world.virtual_seconds == float(record['virtual_seconds'])
        assert len(world.trace) == int(record['action_count'])
        for name, value in world.stats.items():
            assert value == float(record[name]), (record['scene_id'], record['method'], name)
        for name in ('visited_stations', 'safe_circle_clears', 'fallback_clears', 'adaptive_measure_count'):
            assert result[name] == int(record[name])
        compared += 1
    result = {
        'deterministic_replayed_runs': compared, 'all_non_wall_clock_metrics_exact_match': True,
        'wall_clock_time_excluded': True, 'official_simulator_calls': 0,
        'scene_file_sha256': hashlib.sha256((OUTPUT / 'offline_scenes.json').read_bytes()).hexdigest(),
        'runtime_seconds': time.perf_counter() - started,
    }
    (OUTPUT / 'replay_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/network_synthesis_v8'), str(ROOT / 'research/offline_validation')]
import numpy as np
import synthesis
from offline_benchmark import OfflineRuleWorld, Source
from shapely import from_wkt


def main():
    directory = ROOT / 'reports/nonuniform_repair_v37/repair_20260912_053044'
    records = [(path, json.loads(path.read_text(encoding='utf-8')))
               for path in (directory / 'iterations').glob('*.json')]
    path, record = min((item for item in records if 'remaining_area' in item[1]), key=lambda item: item[1]['remaining_area'])
    region = from_wkt(record['remaining_geometry'])
    parts = list(region.geoms) if hasattr(region, 'geoms') else [region]
    counterexample = None
    for part in sorted(parts, key=lambda value: value.area, reverse=True):
        position = np.asarray(part.representative_point().coords[0])
        if not 20 < np.linalg.norm(position) <= 1800:
            continue
        heading = synthesis.weakest_heading(position, np.asarray(record['stations']), radius=1000.0)
        if heading is None:
            continue
        heading_degrees = math.degrees(math.atan2(heading[1], heading[0])) % 360
        world = OfflineRuleWorld([Source(1, tuple(position), 1000.0, heading_degrees)], 361424001)
        port = world.port()
        optical = port.clear(np.zeros(2), 1)
        responses = [port.measure(np.asarray(station), 1) for station in record['stations']]
        if optical['result'] != 'no_target_in_range' or any(response['result'] != 'no_signal' for response in responses):
            raise ValueError('Geometric counterexample does not agree with the rule-world implementation')
        counterexample = {'source_position': position.tolist(), 'receiving_radius': 1000.0,
                          'heading_degrees': heading_degrees, 'radio_stations': len(responses),
                          'all_radio_results': 'no_signal', 'origin_optical_result': optical['result'],
                          'remaining_channels': sorted(world.remaining), 'trace': world.trace,
                          'noise_seed_correctness_fixture_only': 361424001}
        break
    output = ROOT / 'reports/layout_stage_v35_v37/public_geometry_counterexample.json'
    if output.exists():
        raise ValueError('Preserve the existing counterexample output')
    data = {'found': counterexample is not None, 'geometry_record': str(path),
            'geometry_record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'constructed_geometry_fixture_not_a_competition_case': True,
            'not_a_performance_test_or_holdout': True, 'official_calls': 0,
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'counterexample': counterexample}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: data[key] for key in ('found', 'constructed_geometry_fixture_not_a_competition_case', 'official_calls')}, indent=2))


if __name__ == '__main__':
    main()

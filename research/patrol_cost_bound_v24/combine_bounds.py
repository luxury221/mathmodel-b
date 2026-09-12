from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output file')
    station_path = args.input / 'station_bound.json'
    station = load(station_path)
    source_paths = [station_path, *sorted((args.input / 'records').glob('*.json'))]
    records = []
    for path in source_paths[1:]:
        row = load(path)
        for kind in ('record', 'trace'):
            if hashlib.sha256(Path(row[kind + '_path']).read_bytes()).hexdigest() != row[kind + '_sha256']:
                raise ValueError('An archived diagnostic input changed')
        combined_meters = max(station['lower_bound_meters'], row['solver']['lower_bound_meters'])
        travel_clear = combined_meters / 5 / row['count'] + 5
        records.append({'scene_id': row['scene_id'], 'profile': row['profile'], 'count': row['count'],
                        'combined_lower_bound_meters': combined_meters, 'travel_plus_clear': travel_clear,
                        'full_absent_scan': travel_clear + row['stricter_absent_measurement_floor']})
    summary = {str(count): {key: statistics.mean(row[key] for row in selected)
                            for key in ('travel_plus_clear', 'full_absent_scan')}
               for count in (10, 13, 'all')
               for selected in [[row for row in records if count == 'all' or row['count'] == count]]}
    data = {'scope': 'conditional fixed 21-station classes only, not all algorithms', 'official_calls': 0,
            'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_hashes': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
            'records': records, 'summary': summary}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

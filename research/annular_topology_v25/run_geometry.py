from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from topology import ROOT, certify_layout, configurations, dual_ring_network, geometry_price, make_layout, public_witnesses, uncovered_witnesses

HERE = Path(__file__).resolve().parent


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def hashes():
    paths = [*HERE.glob('*.py'), *[ROOT / relative for relative in
             ('research/joint_search_v5/coverage_certificate.py', 'research/offline_validation/geometry.py',
              'research/offline_validation/plan_geometry.py')]]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = hashes()
    save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                    'official_calls': 0, 'source_scenes_read': False, 'certificate_cap': 64})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    witnesses = public_witnesses()
    baseline = dual_ring_network()
    baseline_result = {**geometry_price(baseline), **certify_layout(baseline)}
    save(output / 'baseline.json', baseline_result)
    if not baseline_result['complete']:
        raise ValueError('Frozen baseline coverage failed')
    records = []
    started = time.perf_counter()
    for index, configuration in enumerate(configurations()):
        stations = make_layout(**configuration)
        missing = uncovered_witnesses(stations, witnesses)
        row = {'id': index, 'parameters': configuration, 'node_count': len(stations),
               'missing_witnesses': int(missing.sum()), 'screen_passed': not bool(missing.any())}
        if row['screen_passed']:
            row.update(geometry_price(stations))
        records.append(row)
        if (index + 1) % 1000 == 0:
            print('screened', index + 1, 'passed', sum(row['screen_passed'] for row in records), flush=True)
    save(output / 'screening.json', records)
    passing = sorted([row for row in records if row['screen_passed']], key=lambda row: (row['proxy_cost_meters'], row['id']))
    certified = []
    for row in passing[:64]:
        stations = make_layout(**row['parameters'])
        result = {**row, 'stations': stations.tolist(), **certify_layout(stations)}
        certified.append(result)
        save(output / 'certificates' / f"layout_{row['id']}.json", result)
        print('certified', row['id'], 'nodes', row['node_count'], 'complete', result['complete'],
              'remaining', result['remaining_area'], 'length', round(row['route_length_meters'], 3), flush=True)
    complete = [row for row in certified if row['complete']]
    eligible = [row for row in complete if row['proxy_cost_meters'] <= 0.99 * baseline_result['proxy_cost_meters']]
    selected = min(eligible, key=lambda row: (row['proxy_cost_meters'], row['id'])) if eligible else None
    save(output / 'selected.json', selected)
    summary = {'layouts_screened': len(records), 'finite_screen_passed': len(passing),
               'continuous_certificates_checked': len(certified), 'complete_layouts': len(complete),
               'eligible_layouts': len(eligible), 'selected_id': selected['id'] if selected else None,
               'baseline': baseline_result, 'wall_seconds': time.perf_counter() - started,
               'source_integrity': hashes() == frozen, 'not_a_global_optimality_proof': True}
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

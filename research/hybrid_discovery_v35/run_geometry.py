from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from geometry_hybrid import (ROOT, certify_hybrid, dual_ring_network, layouts,
                             mixed_price, public_witnesses, uncovered_witnesses, witness_patches)


HERE = Path(__file__).resolve().parent


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def hashes():
    paths = [HERE / name for name in ('geometry_hybrid.py', 'run_geometry.py', 'test_geometry.py', 'PROTOCOL.md')]
    paths.extend(ROOT / relative for relative in ('research/annular_topology_v25/topology.py',
                 'research/joint_search_v5/coverage_certificate.py', 'research/offline_validation/geometry.py',
                 'research/offline_validation/plan_geometry.py'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output directory')
    frozen = hashes()
    save(output / 'protocol.json', {'source_hashes': frozen, 'created_local': datetime.now().isoformat(),
                                  'official_calls': 0, 'source_scenes_read': False, 'certificate_cap': 128})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    baseline = certify_hybrid(dual_ring_network())
    if not baseline['complete'] or baseline['optical_centers']:
        raise ValueError('Frozen network no longer passes its original coverage check')
    save(output / 'baseline.json', baseline)
    witnesses = public_witnesses()
    records = []
    started = time.perf_counter()
    for identifier, parameters, stations in layouts():
        missing = uncovered_witnesses(stations, witnesses)
        patches = witness_patches(witnesses[missing])
        record = {'id': identifier, 'parameters': parameters, 'stations': stations.tolist(),
                  'radio_count': len(stations), 'missing_witnesses': int(missing.sum()),
                  'screen_passed': patches is not None}
        if patches is not None:
            record['witness_optical_count'] = len(patches)
            record['screen_price'] = mixed_price(stations, patches)
        records.append(record)
        if len(records) % 500 == 0:
            print('screened', len(records), 'passed', sum(row['screen_passed'] for row in records), flush=True)
    save(output / 'screening.json', records)
    passing = sorted([row for row in records if row['screen_passed']],
                     key=lambda row: (row['screen_price']['proxy_cost_meters'], row['id']))
    certified = []
    for row in passing[:128]:
        result = {**row, **certify_hybrid(row['stations'])}
        certified.append(result)
        save(output / 'certificates' / (row['id'] + '.json'), result)
        print('checked', row['id'], 'radio', row['radio_count'], 'complete', result['complete'],
              'patches', len(result['optical_centers']) if result['optical_centers'] is not None else None,
              'cost', result.get('proxy_cost_meters'), flush=True)
    complete = [row for row in certified if row['complete']]
    eligible = [row for row in complete if row['proxy_cost_meters'] <= baseline['proxy_cost_meters'] * 0.99]
    selected = min(eligible, key=lambda row: (row['proxy_cost_meters'], row['id'])) if eligible else None
    save(output / 'selected.json', selected)
    summary = {'layouts_screened': len(records), 'finite_patch_screen_passed': len(passing),
               'continuous_certificates_checked': len(certified), 'complete_hybrid_layouts': len(complete),
               'eligible_layouts': len(eligible), 'selected_id': selected['id'] if selected else None,
               'baseline_proxy_cost_meters': baseline['proxy_cost_meters'],
               'best_complete_proxy_cost_meters': min(row['proxy_cost_meters'] for row in complete) if complete else None,
               'wall_seconds': time.perf_counter() - started, 'source_integrity': frozen == hashes(),
               'official_calls': 0, 'not_a_global_optimality_proof': True}
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['source_integrity']:
        raise ValueError('Geometry source files changed during execution')


if __name__ == '__main__':
    main()

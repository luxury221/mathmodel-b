from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/dual_guided_network_v44'))
import search_network as search


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    directory = args.batch.resolve()
    output = directory / 'audit.json'
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive audit file')
    protocol = read(directory / 'protocol.json')
    for relative, digest in protocol['source_hashes'].items():
        for path in (ROOT / relative, directory / 'source_snapshot' / relative):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Source or snapshot changed: ' + relative)
    pool = np.asarray(read(directory / 'pool.json'))
    if not np.array_equal(pool, search.pool_points()):
        raise ValueError('Point pool differs from the registered generator')
    reference = search.dual_ring_network()
    indices = [int(np.argmin(np.linalg.norm(pool - point, axis=1))) for point in reference]
    if any(np.linalg.norm(pool[index] - point) > 1e-8 for index, point in zip(indices, reference)):
        raise ValueError('Original complete network is absent from the point pool')
    positions, headings = search.initial_witnesses()
    rows = [read(path) for path in sorted((directory / 'iterations').glob('*.json'))]
    if len(rows) != 1 or rows[0]['status'] != 'UNKNOWN' or 'points' in rows[0]:
        raise ValueError('This closeout expects the recorded first-round unknown, not an infeasibility proof')
    groups = search.visible_groups(pool, positions, headings)
    misses = sum(not set(group) & set(indices) for group in groups)
    if misses:
        raise ValueError('Conservative witness system rejects the known complete baseline')
    if len(groups) != rows[0]['witness_groups'] or len(positions) != rows[0]['source_witness_count']:
        raise ValueError('Witness counts differ')
    if not search.certify(reference).empty:
        raise ValueError('The original network no longer has a full continuous certificate')
    summary = read(directory / 'summary.json')
    if summary['returned_candidates'] or summary['eligible_count'] or summary['continuous_complete'] or summary['policy_runs']:
        raise ValueError('Unknown solver result was incorrectly promoted to a tested candidate')
    if not summary['source_hashes_unchanged'] or protocol['source_hashes'] != search.hashes():
        raise ValueError('Source inventory changed')
    result = {'status': 'passed_unknown_status_preserved', 'point_pool_size': len(pool),
              'witness_groups_checked': len(groups), 'known_21_point_reference_satisfies_all_groups': True,
              'known_reference_continuously_complete': True, 'solver_status': 'UNKNOWN',
              'solver_wall_seconds': rows[0]['wall_seconds'], 'solver_branches': rows[0]['branches'],
              'certified_new_networks': 0, 'new_policy_runs': 0, 'new_holdout': False,
              'official_calls': 0, 'formal_calls': 0, 'source_files': len(protocol['source_hashes']),
              'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'conclusion': 'No incumbent before the registered cutoff; feasibility remains unresolved, not disproved.'}
    search.provenance.save(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

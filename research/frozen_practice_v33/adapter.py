from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research/q3_scan_economy_v13'),
                str(ROOT / 'research/radial_patrol_v7'), str(ROOT / 'research/shaped_probe_v11'),
                str(ROOT / 'research/joint_search_v5')]
import run_scan_benchmark
from b2026_robot import session
from b2026_robot.policy import make_policy as original_factory
from scan_policy import ScanEconomyPolicy
from shaped_policy import ShapedProbePolicy


SELECTED = {'q3': 'station_only', 'q4': 'shaped_cost'}
MANIFEST = ROOT / 'reports/scan_validation_v13/validation_20260911_235435/selection.json'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def hashes():
    frozen = load(MANIFEST)['source_hashes']
    if any(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest for relative, digest in frozen.items()):
        raise ValueError('Permanently frozen source differs')
    paths = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md']
    return {**frozen, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def make_policy(client, problem, q4_network='dual21'):
    if problem not in SELECTED or q4_network != 'dual21':
        raise ValueError('Only the registered q3/grid7 and q4/dual21 candidates are allowed')
    verified = hashes()
    original, metadata = original_factory(client, problem, q4_network)
    selected_class = ScanEconomyPolicy if problem == 'q3' else ShapedProbePolicy
    adapted = type('FrozenPracticePolicy', (selected_class, type(original)), {})
    if adapted.account is not type(original).account:
        raise ValueError('Authoritative interface accounting was not preserved')
    policy = adapted(original.port, problem, original.stations, 'E_joint' if problem == 'q3' else 'F_route',
                     'grid7' if problem == 'q3' else 'dual21', mode=SELECTED[problem])
    metadata.update({'variant': SELECTED[problem], 'candidate_version': 'v13' if problem == 'q3' else 'v11',
                     'candidate_source_hashes': verified, 'ground_truth_access': False})
    return policy, metadata


def run_session(client, problem):
    with patch.object(session, 'make_policy', make_policy):
        return session.run_session(client, problem)


def require_validation(directory):
    directory = Path(directory).resolve()
    if directory.drive.upper() != 'D:' or not directory.is_relative_to(ROOT / 'reports/frozen_practice_v33'):
        raise ValueError('Invalid local validation directory')
    report = load(directory / 'verification.json')
    if not report['passed'] or report['official_calls'] != 0 or len(report['http_results']) != 8:
        raise ValueError('The isolated bridge has not passed all eight HTTP checks')
    if load(directory / 'protocol.json')['source_hashes'] != hashes():
        raise ValueError('Bridge sources changed since validation')
    return directory

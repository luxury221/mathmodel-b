from __future__ import annotations

import hashlib
import json
from pathlib import Path

from b2026_robot.client import RobotError
from b2026_robot.storage import ROOT

VALIDATION_ROOT = ROOT / 'reports' / 'candidate_interface_v1'
V3_ROOT = ROOT / 'research' / 'policy_optimization_v3'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_candidate():
    protocol = load(ROOT / 'reports' / 'policy_optimization_v3' / 'protocol.json')
    selection = load(ROOT / 'reports' / 'policy_optimization_v3' / 'selection.json')
    if selection['selected'].get('q3') != 'combined' or selection['hashes'] != protocol['hashes']:
        raise RobotError('Frozen q3 candidate selection changed')
    for name, expected in protocol['hashes']['candidate'].items():
        if file_hash(V3_ROOT / name) != expected:
            raise RobotError(f'Frozen v3 code changed: {name}')
    return protocol['hashes']


def source_hashes():
    paths = list((ROOT / 'src' / 'b2026_candidate').glob('*.py'))
    paths += list((ROOT / 'src' / 'b2026_robot').glob('*.py'))
    paths += list((ROOT / 'research' / 'candidate_interface_v1').glob('*.py'))
    paths += list((ROOT / 'research' / 'interface_validation').glob('*.py'))
    paths += [ROOT / 'src' / 'run_candidate_practice.py', ROOT / 'scripts' / 'run-q3-candidate-practice.ps1']
    paths += [V3_ROOT / 'candidate.py', V3_ROOT / 'experiments.py']
    paths += [ROOT / 'reports' / 'policy_optimization_v3' / name for name in ('protocol.json', 'selection.json')]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in sorted(paths)}


def require_validation():
    try:
        folder = Path((VALIDATION_ROOT / 'LATEST_SUCCESS.txt').read_text(encoding='utf-8-sig').strip()).resolve()
        if not folder.is_relative_to(VALIDATION_ROOT.resolve()) or folder == VALIDATION_ROOT.resolve():
            raise RobotError('Validation evidence must be in the candidate validation directory')
        verification = load(folder / 'verification.json')
        protocol = load(folder / 'protocol.json')
        if (verification.get('passed') is not True or verification.get('candidate_mode') != 'combined'
                or verification.get('problem') != 'q3' or verification.get('official_calls') != 0):
            raise RobotError('Candidate has no successful local integration record')
        if source_hashes() != protocol['source_hashes']:
            raise RobotError('Candidate interface changed since local validation')
        verify_candidate()
        return folder
    except (OSError, KeyError, ValueError) as error:
        raise RobotError(f'Candidate validation evidence unavailable: {error}') from error

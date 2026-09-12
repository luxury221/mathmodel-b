from __future__ import annotations

import importlib.util

from b2026_robot.client import RobotError
from b2026_robot.policy import make_policy as make_original_policy

from .validation import V3_ROOT, verify_candidate


def make_policy(client, problem, q4_network='dual21'):
    if problem != 'q3' or q4_network != 'dual21':
        raise RobotError('This isolated candidate adapter is authorized only for q3/grid7/combined')
    hashes = verify_candidate()
    original, metadata = make_original_policy(client, problem, q4_network)
    if hashes['baseline'] != metadata['frozen_source_hashes']:
        raise RobotError('Candidate and adapter reference different frozen baselines')
    specification = importlib.util.spec_from_file_location('b2026_frozen_candidate_v3', V3_ROOT / 'candidate.py')
    if specification is None or specification.loader is None:
        raise RobotError('Cannot load the frozen candidate module')
    candidate = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(candidate)

    class CandidateInterfacePolicy(candidate.CandidatePolicy, type(original)):
        pass

    policy = CandidateInterfacePolicy(original.port, 'q3', original.stations, 'E_joint', 'grid7', mode='combined')
    metadata.update({'variant': 'E_joint+v3_combined', 'candidate_mode': 'combined',
                     'candidate_source_hashes': hashes['candidate'], 'ground_truth_access': False})
    return policy, metadata

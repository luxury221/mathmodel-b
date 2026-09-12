from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'research' / 'interface_validation'))
sys.path.insert(0, str(ROOT / 'research' / 'policy_optimization_v3'))
client_module = importlib.import_module('b2026_robot.client')
original_policy = importlib.import_module('b2026_robot.policy')
original_session = importlib.import_module('b2026_robot.session')
candidate_policy = importlib.import_module('b2026_candidate.policy')
candidate_session = importlib.import_module('b2026_candidate.session')
validation = importlib.import_module('b2026_candidate.validation')
storage = importlib.import_module('b2026_robot.storage')
candidate_module = importlib.import_module('candidate')
frozen, frozen_hashes = original_policy.frozen_modules()
mock_module = importlib.import_module('mock_server')
OfflineRuleWorld = frozen['offline_benchmark'].OfflineRuleWorld
Source = frozen['offline_benchmark'].Source
MockArena = mock_module.MockArena
Fault = mock_module.Fault

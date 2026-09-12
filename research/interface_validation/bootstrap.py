import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
client_module = importlib.import_module('b2026_robot.client')
storage = importlib.import_module('b2026_robot.storage')
policy_module = importlib.import_module('b2026_robot.policy')
session_module = importlib.import_module('b2026_robot.session')
frozen, FROZEN_HASHES = policy_module.frozen_modules()
OfflineRuleWorld = frozen['offline_benchmark'].OfflineRuleWorld
Source = frozen['offline_benchmark'].Source

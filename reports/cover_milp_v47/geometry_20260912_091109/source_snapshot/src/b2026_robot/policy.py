from __future__ import annotations

import hashlib
import importlib
import json
import sys

from .client import BudgetStop, RobotError
from .storage import ROOT

FROZEN_ROOT = ROOT / 'research' / 'offline_validation'


def frozen_modules():
    protocol = json.loads((ROOT / 'reports' / 'plan_trials_v2' / 'protocol.json').read_text(encoding='utf-8'))
    for name, expected in protocol['source_hashes'].items():
        actual = hashlib.sha256((FROZEN_ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RobotError(f'Frozen v2 source changed: {name}; create a new validated version first')
    if str(FROZEN_ROOT) not in sys.path:
        sys.path.insert(0, str(FROZEN_ROOT))
    modules = {name: importlib.import_module(name) for name in ('geometry', 'plan_policy', 'offline_benchmark')}
    for name, module in modules.items():
        if module.__file__ != str(FROZEN_ROOT / f'{name}.py'):
            raise RobotError(f'Unexpected module import location: {name}')
    return modules, protocol['source_hashes']


class ObservationAdapter:
    def __init__(self, client):
        self._client = client

    def measure(self, position, channel):
        response = self._client.measure(position, channel)
        observation = {'result': response['measure_result']}
        if observation['result'] == 'direction':
            observation['bearing_deg'] = response['svd_deg']
        return observation

    def clear(self, position, channel):
        return {'result': self._client.clear(position, channel)['clear_result']}


def make_policy(client, problem, q4_network='dual21'):
    if problem not in ('q3', 'q4') or q4_network not in ('dual21', 'grid25'):
        raise ValueError('Choose q3/q4 and dual21/grid25')
    modules, hashes = frozen_modules()
    geometry, policy_module = modules['geometry'], modules['plan_policy']
    selected = json.loads((ROOT / 'reports' / 'plan_trials_v2' / 'selection.json').read_text(encoding='utf-8'))['selected']
    if selected != {'q3': 'E_joint', 'q4': 'F_route'}:
        raise RobotError('Frozen development selection was changed')
    network = 'grid7' if problem == 'q3' else q4_network
    stations = {'grid7': geometry.seven_network, 'dual21': geometry.dual_ring_network,
                'grid25': lambda: geometry.triangular_network()[0]}[network]()

    class InterfacePolicy(policy_module.PlanPolicy):
        def account(self, position, channel, operation, result):
            if client.exited:
                raise BudgetStop('Deadline watchdog ended the session')
            path, fields, response = client.last_action
            result_key = 'measure_result' if operation == 'measure' else 'clear_result'
            if (path != '/' + operation or fields['channel'] != channel or response[result_key] != result
                    or tuple(position) != client.position):
                raise RobotError('Policy and accepted interface action disagree')
            self.position = policy_module.np.asarray(client.position).copy()
            self.receiver_channel = client.receiver_channel
            self.virtual_seconds = client.virtual_seconds

    adapter = ObservationAdapter(client)
    port = modules['offline_benchmark'].ObservationPort(adapter.measure, adapter.clear)
    policy = InterfacePolicy(port, problem, stations, selected[problem], network)
    metadata = {'problem': problem, 'variant': selected[problem], 'network': network,
                'frozen_source_hashes': hashes, 'ground_truth_access': False}
    return policy, metadata

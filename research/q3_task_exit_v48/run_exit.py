from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'research/q3_scan_economy_v13'), str(ROOT / 'research/radial_patrol_v7'),
                str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/q3_action_scheduler_v41'),
                str(ROOT / 'research/local_rollout_v20'), str(ROOT / 'research/cover_milp_v47')]
import run_scan_benchmark as baseline
import milp_search as provenance
from scan_policy import ScanEconomyPolicy
from exit_policy import ExitDistributionPolicy, MODES


experiment = baseline.experiment


def hashes():
    frozen = provenance.source_hashes()
    prior_protocol = json.loads((ROOT / 'reports/cover_milp_v47/geometry_20260912_091109/protocol.json').read_text(encoding='utf-8'))
    for relative, expected in prior_protocol['source_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Previously frozen source changed: ' + relative)
    files = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md',
             ROOT / 'research/local_rollout_v20/local_model.py',
             ROOT / 'research/q3_action_scheduler_v41/action_policy.py',
             *sorted((ROOT / 'research/q3_task_exit_audit_v48').glob('*.py'))]
    return {**frozen, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def audit_first_actions(events, trace):
    checked = 0
    for event in events:
        if event['selected_kind'] != 'target':
            continue
        actions = [action for action in trace if event['start_seconds'] < action['virtual_seconds'] <= event['end_seconds']]
        expected = 'measure' if event['first_kind'] == 'probe' else 'clear'
        if not actions or actions[0]['position'] != event['first_position'] or actions[0]['action'] != expected:
            raise ValueError('Scheduled physical first action differs')
        if actions[0]['channel'] != event['selected_identifier'] or event['active_after'] - event['active_before'] > 8:
            raise ValueError('Task channel or original active budget differs')
        if actions[-1]['position'] != event['actual_exit']:
            raise ValueError('Task endpoint differs from actual physical trace')
        checked += 1
    return checked


def evaluate(scene, mode):
    holder = []
    def factory(*arguments, mode):
        policy = ScanEconomyPolicy(*arguments, mode='station_only') if mode == 'previous' else ExitDistributionPolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['task_events'] = getattr(holder[0], 'task_events', [])
    row['effective_target_radius'] = holder[0].options['target_radius']
    row['timing_audit'] = baseline.audit_trace(trace, row['virtual_seconds'])
    if row['success']:
        try:
            row['first_actions_checked'] = audit_first_actions(row['task_events'], trace)
        except ValueError as error:
            row.update({'success': False, 'seconds_per_source': None, 'failure': str(error)})
    return row, trace


def summarize(records, modes):
    reference = {row['scene_id']: row for row in records if row['mode'] == 'previous'}
    identity = [row for row in records if row['mode'] == 'identity']
    identity_ok = all(row['success'] and row['trace_sha256'] == reference[row['scene_id']]['trace_sha256'] for row in identity)
    summary = {}
    for mode in modes:
        rows = [row for row in records if row['mode'] == mode]
        valid = identity_ok and all(row['success'] for row in rows) and all(row['success'] for row in reference.values())
        mean = statistics.mean(row['seconds_per_source'] for row in rows) if valid else None
        saved = statistics.mean(reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source'] for row in rows) if valid else None
        worst = max((row['seconds_per_source'] / reference[row['scene_id']]['seconds_per_source'] - 1) * 100 for row in rows) if valid else None
        summary[mode] = {'runs': len(rows), 'successes': sum(row['success'] for row in rows), 'mean': mean,
                         'saved_seconds_per_source': saved, 'worst_regression_percent': worst,
                         'identity_passed': identity_ok,
                         'screen_gate_passed': bool(valid and saved >= 1 and worst <= 10)}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, default=[13])
    parser.add_argument('--modes', nargs='+', choices=('previous', *MODES), default=['previous', *MODES])
    parser.add_argument('--gate-batch', type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.drive.upper() != 'D:' or output.exists() or arguments.counts not in ([13], [10, 16]):
        raise ValueError('Use a fresh D-drive output with registered counts')
    if arguments.counts == [13] and arguments.modes != ['previous', *MODES]:
        raise ValueError('Initial screening must contain all four registered modes')
    if not {'previous', 'identity'}.issubset(arguments.modes) or len(arguments.modes) != len(set(arguments.modes)):
        raise ValueError('Retain distinct baseline and identity controls')
    if arguments.counts == [10, 16]:
        if arguments.gate_batch is None:
            raise ValueError('Expansion requires the registered screening results')
        gate = json.loads((arguments.gate_batch / 'summary.json').read_text(encoding='utf-8'))
        if any(not gate[mode]['screen_gate_passed'] for mode in arguments.modes if mode not in ('previous', 'identity')):
            raise ValueError('An expansion candidate failed screening')
    frozen = hashes()
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'counts': arguments.counts, 'modes': arguments.modes, 'base_seed': 103100000,
                                              'official_calls': 0, 'formal_calls': 0, 'new_holdout': False,
                                              'metric': 'Q3 case-equal complete mission seconds/source; all failures retained'})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', arguments.counts) if scene['problem'] == 'q3']
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    records = []
    for scene in scenes:
        for mode in arguments.modes:
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 6) if row['success'] else row['failure'],
                  'forecasts', row['policy_stats'].get('task_forecasts', 0),
                  'fallbacks', row['policy_stats'].get('task_route_fallbacks', 0), flush=True)
    summary = summarize(records, arguments.modes)
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == hashes()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

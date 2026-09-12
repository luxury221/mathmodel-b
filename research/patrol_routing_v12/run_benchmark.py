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
sys.path[:0] = [str(HERE), str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/shaped_probe_v11'),
                str(ROOT / 'research/radial_patrol_v7')]
import experiment_v5 as experiment
from final_validation import audit_trace
from policy_v5 import ContinuousPolicy
from policy_adapter import LayoutPolicy, route_context
from radial_policy import RadialPolicy
from shaped_policy import ShapedProbePolicy
from strong_routes import StrongRouter


MODES = {'q3': ('previous', 'multistart', 'guided'),
         'q4': ('previous', 'v11_previous', 'multistart', 'guided', 'layout', 'layout_guided')}


def source_hashes():
    frozen = json.loads((ROOT / 'reports/shaped_probe_validation_v11/validation_20260911_223412/selection.json').read_text(encoding='utf-8'))['source_hashes']
    for relative, digest in frozen.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Previously frozen dependency changed: ' + relative)
    return {**frozen, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('*.py')}}


def evaluate(scene, mode):
    router = StrongRouter('guided' if mode.endswith('guided') else 'multistart') if mode in ('multistart', 'guided', 'layout_guided') else None
    def factory(port, problem, stations, variant, network, mode):
        if problem == 'q3':
            return RadialPolicy(port, problem, stations, variant, network, mode='pilot1600')
        if mode == 'previous':
            return ContinuousPolicy(port, problem, stations, variant, network, mode='certified_fixed')
        policy = LayoutPolicy if mode.startswith('layout') else ShapedProbePolicy
        return policy(port, problem, stations, variant, network, mode='shaped_cost')
    with route_context(router):
        row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['route_events'] = router.events if router is not None else []
    row['route_summary'] = {'calls': len(row['route_events']),
                            'planning_wall_seconds': sum(event['seconds'] for event in row['route_events']),
                            'shortened_plans': sum(event['selected_meters'] < event['original_meters'] - 1e-7 for event in row['route_events']),
                            'timed_solver_fallbacks': sum(event['solver']['status'] == 'deterministic_fallback' for event in row['route_events'])}
    row['timing_audit'] = audit_trace(trace, row['virtual_seconds'])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--counts', nargs='+', type=int, default=[13])
    parser.add_argument('--problems', nargs='+', choices=tuple(MODES), default=list(MODES))
    parser.add_argument('--modes', nargs='+')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    chosen_modes = {problem: tuple(mode for mode in MODES[problem] if args.modes is None or mode in args.modes) for problem in args.problems}
    if args.modes is not None and set(args.modes) - set().union(*(set(modes) for modes in MODES.values())):
        raise ValueError('Unknown benchmark mode')
    frozen = source_hashes()
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] in args.problems]
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                               'modes': chosen_modes, 'official_calls': 0,
                                               'metric': 'case-equal total virtual seconds/source; all failures invalidate the mode mean',
                                               'route_scope': 'reorder only currently available legal action points; no evaluator truth inputs'})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    records = []
    for scene in scenes:
        for mode in chosen_modes[scene['problem']]:
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['problem'], scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'route_s', round(row['route_summary']['planning_wall_seconds'], 2), flush=True)
    summary = {}
    for problem in args.problems:
        summary[problem] = {}
        for mode in chosen_modes[problem]:
            rows = [row for row in records if row['mode'] == mode and row['problem'] == problem]
            summary[problem][mode] = {'runs': len(rows), 'successes': sum(row['success'] for row in rows),
                                     'mean': statistics.mean(row['seconds_per_source'] for row in rows) if rows and all(row['success'] for row in rows) else None,
                                     'timed_solver_fallbacks': sum(row['route_summary']['timed_solver_fallbacks'] for row in rows)}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': source_hashes() == frozen})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

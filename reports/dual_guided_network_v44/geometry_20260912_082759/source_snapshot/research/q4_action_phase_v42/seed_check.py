from __future__ import annotations

import json
import re

import run_phase as runner


def integers(value):
    if isinstance(value, dict):
        return set().union(*(integers(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(integers(item) for item in value)) if value else set()
    return {value} if isinstance(value, int) else set()


def main():
    factory_base = 481942519
    reserved = {471842503, *(scene['seed'] for scene in runner.experiment.make_scenes(factory_base, 'unit', [10]))}
    scene_files = list((runner.ROOT / 'reports').rglob('*scene*.json'))
    prior = [path for path in (runner.ROOT / 'reports').rglob('*seed*audit*.json') if path.parent.name != runner.HERE.name]
    existing = set()
    for path in [*scene_files, *prior]:
        existing |= integers(json.loads(path.read_text(encoding='utf-8-sig')))
    tests = [path for path in (runner.ROOT / 'research').rglob('test*.py') if runner.HERE not in path.parents]
    for path in tests:
        values = {int(value) for value in re.findall(r'\b\d{8,10}\b', path.read_text(encoding='utf-8'))}
        existing |= values
        existing |= {value + problem * 1000000 + profile * 10000 + count * 100
                     for value in values for problem in range(2) for profile in range(6) for count in (10, 13, 16)}
    result = {'purpose': 'correctness_only_not_holdout', 'reserved_unit_factory_base': factory_base,
              'reserved_seeds': sorted(reserved), 'existing_scene_files_checked': len(scene_files),
              'prior_seed_audits_checked': len(prior), 'test_files_checked': len(tests), 'overlap': sorted(reserved & existing)}
    output = runner.ROOT / 'reports/q4_action_phase_v42/unit_seed_audit.json'
    if output.exists():
        raise ValueError('Do not overwrite an existing seed audit')
    runner.experiment.save(output, result)
    print(json.dumps(result, indent=2))
    if result['overlap']:
        raise ValueError('Correctness seed collision')


if __name__ == '__main__':
    main()

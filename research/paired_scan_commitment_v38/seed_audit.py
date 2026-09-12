from __future__ import annotations

import json
import re

import run_paired as runner


def integers(value):
    if isinstance(value, dict):
        return set().union(*(integers(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(integers(item) for item in value)) if value else set()
    return {value} if isinstance(value, int) else set()


def main():
    factory_base = 381738127
    reserved = {361425001, *(scene['seed'] for scene in runner.experiment.make_scenes(factory_base, 'unit', [10]))}
    existing = set()
    scene_files = list((runner.ROOT / 'reports').rglob('*scene*.json'))
    prior_audits = [path for path in (runner.ROOT / 'reports').rglob('*seed*audit*.json')
                    if path.parent.name != 'paired_scan_commitment_v38']
    destination = runner.ROOT / 'reports/paired_scan_commitment_v38/unit_seed_audit_v3.json'
    if destination.exists():
        raise ValueError('Do not overwrite a completed seed audit')
    for path in [*scene_files, *prior_audits]:
        existing |= integers(json.loads(path.read_text(encoding='utf-8-sig')))
    test_files = [path for path in (runner.ROOT / 'research').rglob('test*.py') if runner.HERE not in path.parents]
    for path in test_files:
        values = {int(value) for value in re.findall(r'\b\d{8,10}\b', path.read_text(encoding='utf-8'))}
        existing |= values
        existing |= {value + problem * 1000000 + profile * 10000 + count * 100
                     for value in values for problem in range(2) for profile in range(6) for count in (10, 13, 16)}
    overlap = sorted(reserved & existing)
    result = {'purpose': 'correctness_only_not_new_holdout', 'reserved_unit_factory_base': factory_base,
              'reserved_seeds': sorted(reserved), 'existing_scene_files_checked': len(scene_files),
              'prior_seed_audits_checked': len(prior_audits), 'test_files_checked': len(test_files), 'overlap': overlap}
    runner.experiment.save(destination, result)
    print(json.dumps(result, indent=2))
    if overlap:
        raise ValueError('Unit seed reservation overlaps previous evidence')


if __name__ == '__main__':
    main()

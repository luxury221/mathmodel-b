from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/paired_scan_commitment_v38'))
import numpy as np
import run_paired as runner
from geometry import ring
from offline_benchmark import OfflineRuleWorld, Source
from paired_planner import route_seconds
from paired_policy import PairedScanPolicy
from shapely.geometry import mapping


def invisible_heading(point, receivers, optical):
    point = np.asarray(point)
    if np.linalg.norm(point) > 1800:
        return None
    if any(np.linalg.norm(point - prior) <= 20 for prior in optical):
        return None
    vectors = np.asarray(receivers) - point
    distances = np.linalg.norm(vectors, axis=1)
    if np.any(distances <= 5):
        return None
    close = vectors[distances <= 1000]
    if len(close):
        angles = np.sort(np.mod(np.arctan2(close[:, 1], close[:, 0]), 2 * math.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * math.pi])
        largest = int(np.argmax(gaps))
        gap = float(gaps[largest])
        if gap <= math.pi + 1e-9:
            return None
        heading = float((angles[largest] + gap / 2) % (2 * math.pi))
        direction = np.array([math.cos(heading), math.sin(heading)])
        if np.any(close @ direction >= 0):
            return None
    else:
        gap = 2 * math.pi
        heading = 0.0
    return {'position': point.tolist(), 'heading_deg': math.degrees(heading),
            'radius': 1000.0, 'largest_angular_gap_deg': math.degrees(gap)}


class DiagnosticPolicy(PairedScanPolicy):
    def __init__(self, *args, mode='paired'):
        super().__init__(*args, mode=mode)
        self.rejections = []
        self.decision_stops = None
        original_complete = self.pair_planner.complete
        def recording_complete(stops, kept):
            trial = original_complete(stops, kept)
            if len(stops) == 2 and not trial.empty:
                removed = tuple(index for index in self.remaining_stations if index not in kept)
                unknown_count = len(self.unknown_channels())
                baseline = route_seconds([stop[2] for stop in self.decision_stops], self.position) + 6 * unknown_count * len(self.remaining_stations)
                price = self.pair_planner.proposed_price(stops, removed, self.decision_stops, unknown_count)
                receivers = [*self.coverage.observations, *(stop[2] for stop in stops), *(self.route[index] for index in kept)]
                parts = list(trial.region.geoms) if hasattr(trial.region, 'geoms') else [trial.region]
                witness = None
                for part in sorted(parts, key=lambda part: part.area, reverse=True):
                    representative = np.array(part.representative_point().coords[0])
                    witness = invisible_heading(representative, receivers, self.coverage.clear_observations)
                    if witness is not None:
                        break
                self.rejections.append({'seconds': self.virtual_seconds, 'position': self.position.tolist(),
                                        'unknown_channels': self.unknown_channels(), 'removed': list(removed), 'kept': list(kept),
                                        'planned_points': [stop[2].tolist() for stop in stops],
                                        'planned_actions': [(stop[0], stop[1]) for stop in stops],
                                        'baseline_proxy_seconds': baseline, 'proposed_proxy_seconds': price,
                                        'proxy_saving_seconds': baseline - price, 'remaining_area': trial.area,
                                        'remaining_components': len(parts), 'geometry': mapping(trial.region),
                                        'radio_positions': [point.tolist() for point in receivers],
                                        'optical_positions': [point.tolist() for point in self.coverage.clear_observations],
                                        'physical_witness': witness})
            return trial
        self.pair_planner.complete = recording_complete

    def transit_proposal(self, destination, stops):
        self.decision_stops = stops
        return super().transit_proposal(destination, stops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive diagnostic directory')
    assert invisible_heading([0, 0], [[100, 0], [100, 100]], []) is not None
    assert invisible_heading([0, 0], ring(100, 8), []) is None
    sources = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in [HERE / 'PROTOCOL.md', Path(__file__).resolve()]}
    runner.experiment.save(output / 'protocol.json', {'source_hashes': {**runner.hashes(), **sources},
                                                    'kind': 'post_hoc_rejection_diagnostic', 'official_calls': 0})
    for relative in sources:
        shutil.copy2(ROOT / relative, output / Path(relative).name)
    scenes = json.loads((args.batch / 'scenes_evaluator_only.json').read_text(encoding='utf-8'))
    scene = next(scene for scene in scenes if scene['profile'] == 'max_radius' and scene['count'] == 13)
    holder = []
    def factory(*arguments, mode):
        policy = DiagnosticPolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    row, trace = runner.experiment.evaluate(scene, 'paired', policy_factory=factory)
    reference_path = args.batch / 'records' / (scene['id'] + '__paired.json')
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    if not row['success'] or row['trace_sha256'] != reference['trace_sha256']:
        raise ValueError('Diagnostic recording changed the actual mission')
    rejections = holder[0].rejections
    for index, rejection in enumerate(rejections):
        witness = rejection['physical_witness']
        if witness is None:
            continue
        source = Source(20, tuple(witness['position']), witness['radius'], witness['heading_deg'])
        world = OfflineRuleWorld([source], 361425001)
        for position in rejection['radio_positions']:
            if world.measure(np.asarray(position), 20)['result'] != 'no_signal':
                raise ValueError('Claimed undetected witness produced a radio observation')
        for position in rejection['optical_positions']:
            if world.clear(np.asarray(position), 20)['result'] != 'no_target_in_range':
                raise ValueError('Claimed undetected witness was cleared optically')
        if world.remaining != {20}:
            raise ValueError('Constructed witness did not remain')
        witness['radio_no_signal_count'] = len(rejection['radio_positions'])
        witness['failed_optical_count'] = len(rejection['optical_positions'])
        runner.experiment.save(output / f'constructed_witness_{index}_trace.json', world.trace)
    result = {'reference_scene_id': scene['id'], 'trace_sha256': row['trace_sha256'], 'trace_identical': True,
              'complete_mission_seconds': row['virtual_seconds'], 'timing_audit': runner.run_scan_benchmark.audit_trace(trace, row['virtual_seconds']),
              'rejection_count': len(rejections), 'physical_counterexamples': sum(item['physical_witness'] is not None for item in rejections),
              'rejections': rejections, 'official_calls': 0, 'formal_calls': 0,
              'unit_seed_reused': 361425001, 'new_holdout': False, 'not_a_performance_sample': True}
    runner.experiment.save(output / 'diagnostic.json', result)
    runner.experiment.save(output / 'trace.json', trace)
    print(json.dumps({key: value for key, value in result.items() if key != 'rejections'}, indent=2))
    for rejection in rejections:
        print(json.dumps({key: rejection[key] for key in ('removed', 'planned_actions', 'proxy_saving_seconds', 'remaining_area', 'remaining_components', 'physical_witness')}, indent=2))


if __name__ == '__main__':
    main()

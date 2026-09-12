import math
import unittest

import run_validation as validation


class ValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = validation.build_scenes()

    def test_scene_count_ids_and_seeds(self):
        self.assertEqual(len(self.scenes), 64)
        self.assertEqual(len({scene['id'] for scene in self.scenes}), 64)
        self.assertEqual(len({scene['seed'] for scene in self.scenes}), 64)

    def test_no_q3_or_unknown_variant(self):
        self.assertTrue(all(scene['problem'] == 'q4' for scene in self.scenes))
        self.assertEqual(validation.VARIANTS, (('baseline', 'dual21'), ('joint', 'dual21'), ('baseline', 'grid25')))

    def test_scene_generation_is_deterministic(self):
        self.assertEqual(self.scenes, validation.build_scenes())

    def test_source_bounds_and_channel_uniqueness(self):
        for scene in self.scenes:
            self.assertEqual(len(scene['sources']), scene['count'])
            self.assertEqual(len({source['channel'] for source in scene['sources']}), scene['count'])
            for source in scene['sources']:
                self.assertLessEqual(math.hypot(*source['position']), 1800.0)
                self.assertTrue(1000.0 <= source['radius'] <= 1500.0)
                self.assertIn(source['channel'], range(1, 21))

    def test_profile_balance(self):
        for profile in validation.PROFILES:
            self.assertEqual(sum(scene['profile'] == profile for scene in self.scenes), 8)
        self.assertEqual(sum(scene['profile'] == 'heading_sweep' for scene in self.scenes), 16)

    def test_action_signature_ignores_only_clock(self):
        action = {'action': 'measure', 'channel': 1, 'position': [0.0, 0.0],
                  'response': {'result': 'no_signal'}, 'virtual_seconds': 5.0}
        changed = {**action, 'virtual_seconds': 5.000001}
        self.assertEqual(validation.action_hash([action]), validation.action_hash([changed]))
        self.assertNotEqual(validation.action_hash([action]), validation.action_hash([{**changed, 'channel': 2}]))

    def test_failed_run_is_not_rewarded_as_fast(self):
        rows = [
            {'scene_id': 'first', 'mode': 'baseline', 'network': 'dual21', 'all_cleared': True, 'seconds_per_source': 100},
            {'scene_id': 'first', 'mode': 'joint', 'network': 'dual21', 'all_cleared': False, 'seconds_per_source': None},
        ]
        result = validation.paired_summary(rows, 'baseline_dual21', 'joint_dual21')
        self.assertEqual(result['cases'], 1)
        self.assertEqual(result['candidate_full_clear'], 0)
        self.assertEqual(result['both_full_clear'], 0)
        self.assertNotIn('mean_reduction_pct', result)

    def test_paired_summary_direction_and_ties(self):
        rows = []
        for index, candidate in enumerate((80, 100, 120)):
            for mode, value in (('baseline', 100), ('joint', candidate)):
                rows.append({'scene_id': str(index), 'mode': mode, 'network': 'dual21',
                             'all_cleared': True, 'seconds_per_source': value})
        result = validation.paired_summary(rows, 'baseline_dual21', 'joint_dual21')
        self.assertEqual((result['wins'], result['ties'], result['losses']), (1, 1, 1))
        self.assertAlmostEqual(result['mean_reduction_pct'], 0.0)
        self.assertAlmostEqual(result['worst_slowdown_pct'], 20.0)


if __name__ == '__main__':
    unittest.main()

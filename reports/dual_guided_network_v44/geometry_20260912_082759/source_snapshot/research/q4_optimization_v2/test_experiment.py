import math
import unittest

from experiment import ALL_MODES, build_scenes
from http_selected import policy_factory


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = build_scenes()

    def test_phase_sizes_and_disjoint_seeds(self):
        self.assertEqual(len(self.scenes), 88)
        self.assertEqual(len({scene['seed'] for scene in self.scenes}), 88)
        self.assertEqual(len({scene['id'] for scene in self.scenes}), 88)
        for phase, expected in (('development', 24), ('holdout', 48), ('stress', 16)):
            self.assertEqual(sum(scene['phase'] == phase for scene in self.scenes), expected)

    def test_scene_bounds(self):
        for scene in self.scenes:
            self.assertEqual(scene['problem'], 'q4')
            self.assertEqual(len({source['channel'] for source in scene['sources']}), scene['count'])
            for source in scene['sources']:
                self.assertLessEqual(math.hypot(*source['position']), 1800)
                self.assertTrue(1000 <= source['radius'] <= 1500)

    def test_local_factories_preserve_authoritative_accounting(self):
        for mode in ALL_MODES:
            policy, metadata = policy_factory(mode)(object(), 'q4')
            self.assertIn('InterfacePolicy.account', policy.account.__func__.__qualname__)
            self.assertEqual(metadata['network'], 'dual21')

    def test_local_factory_rejects_wrong_problem_and_network(self):
        for problem, network in (('q3', 'dual21'), ('q4', 'grid25')):
            with self.assertRaises(ValueError):
                policy_factory('scan')(object(), problem, network)


if __name__ == '__main__':
    unittest.main()

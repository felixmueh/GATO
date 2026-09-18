"""CPU checks for repeat aggregation and compatibility with saved experiments."""
import copy
import json
import math
import unittest

from plotting import aggregate, normalize, report


class PlottingTests(unittest.TestCase):
    def manifest(self, **config):
        return {'config': dict(gpu_name='Test GPU', knots=[8], batches=[1], repeats=3,
                               protocol='wall-time', max_sqp_iters=1000, wall_time=10,
                               max_solve_ms=250, **config), 'results': [
            dict(N=8, batch_size=1, repeat=0, status='completed', samples=100,
                 avg_native_ms=2, warmup={'native_ms': 100000}),
            dict(N=8, batch_size=1, repeat=1, status='completed', samples=1,
                 avg_native_ms=8),
            dict(N=8, batch_size=1, repeat=2, status='slow_solve', samples=1,
                 avg_native_ms=300)]}

    def test_equal_repeat_weight_and_retained_slow_outcome(self):
        config, results = normalize(self.manifest())
        row, = aggregate(config, results)
        self.assertEqual(row['mean_native_ms'], 5)
        self.assertEqual(row['solver_frequency_hz'], 200)
        self.assertEqual(row['samples'], 101)
        self.assertEqual(row['all_retained_samples'], 102)
        self.assertTrue(row['partial_repeats'])
        self.assertEqual(row['status'], 'mixed')
        self.assertEqual(results[2]['avg_native_ms'], 300)
        self.assertIn('**250 ms**', report(config, results, [row]))

    def test_reference_compatibility_and_no_mutation(self):
        manifest = {'config': dict(gpu_name='Another GPU', knots=[8], batches=[1], sim_time=10),
                    'results': [dict(N=8, batch_size=1, status='ok', samples=5,
                                     avg_gpu_time_ms=2, avg_goal_distance=.03)]}
        before = copy.deepcopy(manifest)
        config, results = normalize(manifest)
        self.assertEqual(config['max_sqp_iters'], 1)
        self.assertEqual(config['protocol'], 'reference')
        row, = aggregate(config, results)
        self.assertEqual(row['mean_native_ms'], 2)
        self.assertEqual(row['tracking_mean_m'], .03)
        self.assertIn('cap is intentional', report(config, results, [row]))
        self.assertEqual(manifest, before)

    def test_missing_repeats_and_disabled_cutoff(self):
        manifest = self.manifest()
        manifest['results'] = manifest['results'][:1]
        manifest['config']['max_solve_ms'] = 0
        config, results = normalize(manifest)
        row, = aggregate(config, results)
        self.assertTrue(row['partial_repeats'])
        self.assertIn('2 missing', row['outcomes'])
        self.assertEqual(row['recorded_repeats'], 1)
        self.assertIn('cutoff is disabled', report(config, results, [row]))

    def test_duplicate_and_invalid_frequency_rejected(self):
        manifest = self.manifest()
        manifest['results'][1]['repeat'] = 0
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            normalize(manifest)
        manifest = self.manifest()
        manifest['results'][0]['solver_frequency_hz'] = 1
        with self.assertRaisesRegex(ValueError, 'reciprocal'):
            normalize(manifest)

    def test_old_wall_manifest(self):
        manifest = self.manifest()
        del manifest['config']['protocol']
        config, _ = normalize(manifest)
        self.assertEqual(config['protocol'], 'wall-time')

    def test_failed_invalid_timing_is_retained_without_a_frequency(self):
        for value in (0, float('nan'), None):
            with self.subTest(value=value):
                manifest = self.manifest()
                manifest['results'][0].update(status='error', avg_native_ms=value,
                                               solver_frequency_hz=float('nan'))
                config, results = normalize(manifest)
                self.assertIsNone(results[0]['avg_native_ms'])
                self.assertIsNone(results[0]['solver_frequency_hz'])
                self.assertFalse(results[0]['native_time_valid'])
                if value is not None and math.isnan(value):
                    self.assertTrue(math.isnan(results[0]['reported_native_ms']))
                else:
                    self.assertEqual(results[0]['reported_native_ms'], value)
                row, = aggregate(config, results)
                self.assertEqual(row['mean_native_ms'], 8)
                self.assertIn('1 error', report(config, results, [row]))
        # A strict-JSON null is also valid input for a failed repeat.
        manifest = self.manifest()
        manifest['results'][0].update(status='error', avg_native_ms=None)
        normalize(json.loads(json.dumps(manifest, allow_nan=False)))

    def test_completed_invalid_timing_still_rejected(self):
        for value in (0, float('nan'), None):
            with self.subTest(value=value):
                manifest = self.manifest()
                manifest['results'][0]['avg_native_ms'] = value
                with self.assertRaises(ValueError):
                    normalize(manifest)


if __name__ == '__main__':
    unittest.main()

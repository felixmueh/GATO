"""CPU contracts for additive large-N studies and independent build failures."""
import json
import unittest

import run
import task_parameters_large_plan as generation
from adapters.task_parameter_capacity import classify


class LargeHorizonPlans(unittest.TestCase):
    def test_counts_builds_preflight_and_reused_pilots(self):
        plans = generation.plans('a'*40, 'b'*40, 'c'*40)
        total = 0
        for setting, expected in (('rtx5060_internal', 19), ('rtx5060_gazebo', 11), ('jetson', 38)):
            plan = plans[setting]
            self.assertEqual(plan['scientific_episodes'], expected)
            total += expected
            episodes, build_keys, groups = set(), set(), set()
            for e in plan['experiments']:
                self.assertEqual(e['repeats'], 1)
                N = e['cases'][0]['parameters']['N']
                self.assertIn(f'-DKNOTS=32;{N}', e['build']['commands'][0])
                self.assertIn('modules/bsqpN32_tiago_right*.so', e['build']['artifacts'])
                self.assertEqual(len(e['build']['artifacts']), 3)
                self.assertNotIn('{case}', e['preflight']['command'])
                self.assertEqual(e['source_commit'], 'c'*40 if setting.endswith('gazebo') else 'b'*40)
                self.assertNotIn(run.DANGEROUS_SAFETY_FLAG, e['command'])
                for c in e['cases']:
                    self.assertIn(c['parameters']['N'], (256, 512) if setting == 'jetson' else (256,))
                    self.assertEqual(c['parameters']['pcg_max_iter'], 120)
                    if e['id'].startswith('B-'):
                        self.assertNotEqual(c['parameters']['dt_s'], .03)
                    episodes.add(run.digest(run.scientific_identity(e, dict(c, repeat=0), e['machines'][0])))
                build_keys.add(run.build_key(e, {}, '/python', {}))
                groups.add(e['preflight']['group'])
            self.assertEqual(len(build_keys), 2 if setting == 'jetson' else 1)
            self.assertEqual(len(groups), len(build_keys))
            self.assertEqual(len(episodes), expected)
            for e in plans[setting+'_pilots']['experiments']:
                for c in e['cases']:
                    self.assertIn(run.digest(run.scientific_identity(e, dict(c, repeat=0), e['machines'][0])), episodes)
            readiness = plans[setting+'_native_readiness']
            self.assertEqual(readiness['scientific_episodes'], 0)
            for e in readiness['experiments']:
                self.assertNotIn('preflight', e)
                self.assertNotIn('allow_report_only_safety', e)
                self.assertIn('--native-readiness', e['command'])
                self.assertIn(run.build_key(e, {}, '/python', {}), build_keys)
        self.assertEqual(total, 68)

    def test_capacity_sources_and_error_classification(self):
        plans = generation.capacity_plans('b'*40)
        self.assertEqual(len(plans['rtx5060']['experiments']), 1)
        self.assertEqual(len(plans['jetson']['experiments']), 2)
        for plan in plans.values():
            for e in plan['experiments']:
                N = e['cases'][0]['parameters']['N']
                self.assertEqual(e['source_commit'], 'b'*40)
                self.assertIn(f'-DKNOTS={N}', e['build']['commands'][0])
                self.assertEqual(e['build']['artifacts'], [f'resources_N{N}'])
        resource = dict(N=256, plant='tiago_right', kernels=[dict(name='pcg', fits=False)],
            pcg_shared_memory_policy=dict(configuration_error='PCG shared-memory unsupported: capacity exceeded'))
        self.assertIsNone(classify(resource, 0, 256)['execution_error'])
        for error in ('cudaFuncSetAttribute failed', 'cudaDeviceGetAttribute failed', 'attribute verification failed'):
            resource['pcg_shared_memory_policy']['configuration_error'] = error
            self.assertEqual(classify(resource, 0, 256)['outcome'], 'execution_failed')


if __name__ == '__main__':
    unittest.main()

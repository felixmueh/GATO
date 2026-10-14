"""CPU contracts for policy separation and the frozen parameter-study queues."""
import copy
import os
from unittest.mock import patch
import unittest

import run
import task_parameters_plan as generation


class ParameterPlans(unittest.TestCase):
    def setUp(self):
        self.plans = generation.plans('a'*40, 'b'*40)

    def test_counts_reuse_pilots_and_all_N_build(self):
        total = 0
        for setting in ('gtx1070', 'rtx5060_internal', 'jetson', 'rtx5060_gazebo'):
            plan = self.plans[setting]
            total += plan['scientific_episodes']
            self.assertEqual(plan['scientific_episodes'], 47 if setting.endswith('gazebo') else 82)
            episodes = set()
            keys = set()
            for e in plan['experiments']:
                self.assertNotIn(run.DANGEROUS_SAFETY_FLAG, e['command'])
                self.assertNotIn('{case}', e['preflight']['command'])
                self.assertIn('-DKNOTS=16;32;64;128', e['build']['commands'][0])
                self.assertEqual(len(e['build']['artifacts']), 5)
                for c in e['cases']:
                    self.assertEqual(c['parameters']['pcg_max_iter'], 120)
                    if e['id'].startswith('B-'):
                        self.assertNotEqual(c['parameters']['dt_s'], .03)
                    for repeat in range(e['repeats']):
                        episodes.add(run.digest(run.scientific_identity(e, dict(c, repeat=repeat), e['machines'][0])))
                keys.add(run.build_key(e, {'cuda_arch': '120'}, '/python', {'python_abi': 'same'}))
            self.assertEqual(len(keys), 1)
            self.assertEqual(len(episodes), plan['scientific_episodes'])
            for e in self.plans[setting+'_pilots']['experiments']:
                for c in e['cases']:
                    identity = run.digest(run.scientific_identity(e, dict(c, repeat=0), e['machines'][0]))
                    self.assertIn(identity, episodes)
        self.assertEqual(total, 293)

    def test_opt_in_identity_preflight_and_native_cache(self):
        raw = self.plans['rtx5060_gazebo']['experiments'][0]
        safe = run.execution_policy(raw, 'rtx5060', False, True)
        report = run.execution_policy(raw, 'rtx5060', True, True)
        case = dict(raw['cases'][0], repeat=0)
        safe_id = run.scientific_identity(safe, case, 'rtx5060')
        report_id = run.scientific_identity(report, case, 'rtx5060')
        self.assertNotEqual(safe_id, report_id)
        self.assertNotEqual(run.preflight_group(safe), run.preflight_group(report))
        self.assertEqual(run.build_key(safe, {}, '/python', {}), run.build_key(report, {}, '/python', {}))
        receipt = dict(identity=report_id, case=case, machine='rtx5060')
        self.assertEqual(run.receipt_identity(receipt), report_id)
        self.assertEqual(run.execution_command(['worker'], {}, safe), ['worker'])
        self.assertEqual(run.execution_command(['worker'], {}, report), ['worker', run.DANGEROUS_SAFETY_FLAG])
        legacy = self.plans['gtx1070']['experiments'][0]
        self.assertEqual(run.execution_policy(legacy, 'gtx1070', False), legacy)
        self.assertEqual(run.preflight_group(legacy), legacy['preflight']['group'])

    def test_refuse_wrong_context_and_embedded_opt_in(self):
        raw = self.plans['rtx5060_gazebo']['experiments'][0]
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                run.execution_policy(raw, 'rtx5060', True)
        with patch.dict(os.environ, {'GATO_ROS_PROFILE': 'simulation', 'ROS_DOMAIN_ID': '1',
                'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp'}, clear=True):
            self.assertEqual(run.execution_policy(raw, 'rtx5060', True)['execution_safety_policy'], 'report_only_safety_stops')
        for malformed in (dict(raw, execution_safety_policy='report_only_safety_stops'),
                dict(raw, command=raw['command']+[run.DANGEROUS_SAFETY_FLAG]),
                dict(raw, backend='hardware')):
            with self.assertRaises(ValueError):
                run.execution_policy(malformed, 'rtx5060', False, True)
        with self.assertRaises(ValueError):
            run.execution_policy(self.plans['gtx1070']['experiments'][0], 'gtx1070', True, True)


if __name__ == '__main__':
    unittest.main()

"""CPU postprocessing: derived duration representation vs real protocol changes."""
import copy
import importlib.util
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prediction_analysis', ROOT/'examples/randomized_multimodal/analyze_reaching_campaign.py')
analysis = importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


class PredictionDurationProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.campaign = dict(N=32,knot_dt=.03,prediction_seconds=.93,simulation_step=.003,feedback_seconds=.03)
        self.episode = dict(self.campaign,prediction_seconds=(32-1)*.03)

    def test_saved_float_representation_accepted(self):
        self.assertNotEqual(self.campaign['prediction_seconds'],self.episode['prediction_seconds'])
        analysis.validate_campaign_episode(self.episode,self.campaign)
        # Historical horizon uses the same derived-duration definition.
        analysis.validate_campaign_episode(dict(N=64,knot_dt=.03,prediction_seconds=63*.03),dict(N=64,knot_dt=.03,prediction_seconds=1.89))

    def test_real_duration_or_shared_bad_metadata_rejected(self):
        for value in [.930000000001, .94, .90]:
            with self.subTest(value=value):
                bad=dict(self.episode,prediction_seconds=value)
                with self.assertRaises(ValueError):analysis.validate_campaign_episode(bad,self.campaign)
                with self.assertRaises(ValueError):analysis.validate_campaign_episode(bad,dict(self.campaign,prediction_seconds=value))

    def test_exact_knots_dt_and_other_timing_preserved(self):
        for changes in [dict(N=33,prediction_seconds=32*.03),dict(knot_dt=.030000000000000002,prediction_seconds=31*.030000000000000002),dict(simulation_step=.001),dict(feedback_seconds=.06)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                analysis.validate_campaign_episode(dict(self.episode,**changes),self.campaign)

    def test_missing_nonfinite_and_invalid_rejected_in_each_manifest(self):
        for key in ['N','knot_dt','prediction_seconds']:
            for value in [None,float('nan'),float('inf'),0,-1,True]:
                bad=dict(self.episode,**{key:value})
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):analysis.validate_campaign_episode(bad,self.campaign)
            bad=copy.copy(self.episode);del bad[key]
            with self.assertRaises(ValueError):analysis.validate_campaign_episode(bad,self.campaign)
            campaign=dict(self.campaign,**{key:None})
            with self.assertRaises(ValueError):analysis.validate_campaign_episode(self.episode,campaign)


if __name__=='__main__':unittest.main()

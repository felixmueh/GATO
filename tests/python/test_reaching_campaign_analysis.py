"""Campaign accounting tests: censoring, partial runs, failures and zero costs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("reaching_analysis", ROOT / "examples/randomized_multimodal/analyze_reaching_campaign.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class CampaignAccountingTest(unittest.TestCase):
    def test_changed_runtime_or_seed_rejected(self):
        manifest = dict(pair_id=1, binary_sha256="binary-a", args=dict(seed=42),
                        source_sha256={"runner.py": "runner-a"}, source_input_sha256={"pairs.npz": "pairs-a"})
        campaign = dict(binary_sha256="binary-a", episode_seed=42,
                        source_sha256={"runner.py": "runner-a", "manager.py": "manager-a"},
                        input_sha256={"pairs.npz": "pairs-a"})
        analysis.validate_campaign_episode(manifest, campaign)
        for changed in ({**manifest, "binary_sha256": "binary-b"},
                        {**manifest, "args": {"seed": 43}},
                        {**manifest, "source_sha256": {"runner.py": "runner-b"}}):
            with self.assertRaises(ValueError):
                analysis.validate_campaign_episode(changed, campaign)

    def test_partial_failure_and_zero_denominator_accounting(self):
        episodes, records = {}, []
        for pair in range(8):
            for batch in (1, 16):
                key = pair, batch
                completed = pair != 6 and key != (5, 16)
                state = "completed" if completed else "technical_failure" if key == (5, 16) else "pending"
                attempts = ([{"state": "completed"}] if completed else
                            [{"state": "technical_failure"}, {"state": "technical_failure"}] if key == (5, 16) else [])
                records.append(dict(pair_id=pair, batch=batch, state=state, attempts=attempts))
                if not completed:
                    continue
                valid = not (pair == 4 or key in ((2, 1), (3, 16)))
                cost = 0 if pair == 7 else 100 if batch == 1 else 120 if pair == 1 else 80
                outcome = "timeout" if not valid else "reached"
                e = dict(pair_id=pair, batch=batch, episode_path=f"/saved/pair{pair}/B{batch}",
                         outcome=outcome, reached=outcome == "reached", valid_reach=valid, physical_valid=valid,
                         executed_quadratic_cost=cost, reached_seconds=0 if pair == 7 else 2 if batch == 1 else 1,
                         max_position_excess_deg=0 if valid else 10,
                         max_speed_ratio=float("inf") if pair == 4 else .5, max_torque_ratio=.5)
                episodes[key] = e, (np.zeros(1),), {}
        records[0]["attempts"].insert(0, {"state": "interrupted"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"manifest.json").write_text(json.dumps(dict(pair_ids=list(range(8)), batches=[1, 16])))
            (root/"status.json").write_text(json.dumps(dict(episodes=records, manifest_sha256=analysis.digest(root/"manifest.json"))))
            with patch.object(analysis, "episode", side_effect=lambda _, r, __, ___: episodes.get((r["pair_id"], r["batch"]))), \
                    patch.object(analysis, "figure_outputs"):
                result = analysis.analyze(root, root/"analysis")
            d = result["denominators"]
            self.assertEqual((d["completed_episodes"], d["paired_completed"], d["partial_pairs"], d["pairs_without_completed_episode"]), (13, 6, 1, 1))
            self.assertEqual((d["current_technical_failure_episodes"], d["historical_technical_failure_attempts"]), (1, 2))
            self.assertEqual(result["outcome_contingency"], dict(both_valid_reach=3, B16_only_valid_reach=1, B1_only_valid_reach=1, neither_valid_reach=1))
            stats = result["eligible_performance"]["cost_gain_percent"]
            self.assertEqual((stats["total"], stats["finite"], stats["minimum"], stats["maximum"]), (3, 2, -20., 20.))
            self.assertEqual(result["eligible_gain_omissions"]["cost_gain_omission_reason"], {"zero_B1_denominator": 1})
            self.assertEqual(result["limit_distributions"]["B1"]["max_speed_ratio"]["positive_infinity"], 1)
            candidates = json.loads((root/"analysis/candidate_cases.json").read_text())
            self.assertEqual(candidates["wins"][0]["pair_id"], 2)
            self.assertEqual(candidates["wins"][0]["cost_gain_percent"], None)
            self.assertEqual({x["pair_id"] for x in candidates["regressions"]}, {1, 3})
            # Strict JSON serialization preserves Infinity as an explicit string,
            # instead of dropping the completed numerical-failure observation.
            saved = json.loads((root/"analysis/summary.json").read_text())
            self.assertIn("Infinity", [e["max_speed_ratio"] for e in saved["episodes"]])


if __name__ == "__main__":
    unittest.main()

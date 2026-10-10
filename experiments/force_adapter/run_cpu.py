"""Run frozen CPU correspondence once and preserve a source/raw receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .orchestration import AppliedSegment, TickInputs, WrenchConvention, run_tick
from .records import write_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new raw directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    directory = Path(__file__).resolve().parent
    root = directory.parents[1]
    protocol = json.loads((directory / "cpu_protocol.json").read_text())
    batch = protocol["batch_size"]
    forces = np.zeros((batch, 6))
    forces[:, 0] = np.arange(batch)
    inputs = TickInputs(protocol["previous_time_s"], protocol["observed_time_s"],
                        np.array(protocol["previous_state"]), np.array(protocol["observed_state"]),
                        tuple(AppliedSegment(a, b, np.array([u])) for a, b, u in protocol["applied_history"]),
                        np.column_stack([100 + np.arange(batch), np.zeros(batch)]),
                        np.arange(batch * 3).reshape(batch, 3),
                        np.arange(batch * 4).reshape(batch, 4))
    convention = WrenchConvention("world", "tool origin", "stub world", "tool origin")
    outputs = {}
    for mode in protocol["modes"]:
        calls = []
        class StubEstimator:
            def generate_batch(self):
                calls.append("generate")
                return forces

            def update_frozen(self, selected, errors, candidates):
                calls.append("update")
                np.testing.assert_array_equal(candidates, forces)

        def transform(state, force, convention):
            return force

        def predict(request):
            calls.append("predict")
            state = np.tile(request.inputs.previous_state, (batch, 1))
            for segment in request.inputs.applied_history:
                state[:, 0] += (segment.control[0] + request.hypothesis_wrenches[:, 0]) * (
                    segment.end_s - segment.start_s)
            return state

        def solve(request):
            calls.append("solve")
            return request.warm_seeds + 10

        outputs[mode] = run_tick(mode, inputs, estimator=StubEstimator(), predictor=predict,
                                 optimizer=solve, convention=convention, transform_wrench=transform,
                                 observer=lambda record: write_record(args.output / mode, record))
        expected = (["generate", "solve", "predict", "update"] if mode == "original"
                    else ["generate", "predict", "update", "solve"])
        assert calls == expected
        assert outputs[mode]["selected_index"] == protocol["expected_selected_index"]
    for field in ("hypotheses", "predictions", "errors", "selected_force", "selected_plan"):
        np.testing.assert_array_equal(outputs["original"][field], outputs["adapted"][field])
    assert outputs["original"]["plans"].shape == (16, 4)
    assert outputs["adapted"]["plans"].shape == (1, 4)
    source_paths = sorted(directory.glob("*.py")) + [directory / "cpu_protocol.json",
        root / "examples/force_estimator.py", root / "tests/python/test_force_adapter.py"]
    receipt = {
        "protocol_id": protocol["protocol_id"], "status": "CPU correspondence PASS",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        "raw_sha256": {str(path.relative_to(args.output)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(args.output.rglob("*")) if path.is_file()},
        "limitation": "CPU stub assertions only; no native/TIAGo qualification, performance or memory benefit"
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

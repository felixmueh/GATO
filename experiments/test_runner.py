"""Public CLI integration tests using real temporary Git repositories/processes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        shutil.copytree(HERE, self.root / "experiments", ignore=shutil.ignore_patterns("local.json", "__pycache__"))
        self.output = Path(self.temporary.name) / "results"
        self.cache = Path(self.temporary.name) / "cache"
        self.env = dict(os.environ, RUNNER_TEST_FAIL="0", RUNNER_TEST_SLEEP="0")
        (self.root / ".gitignore").write_text(".worktrees/\nexperiments/local.json\n")
        (self.root / "worker.py").write_text('''import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--case'); p.add_argument('--output')
a=p.parse_args(); out=Path(a.output); c=json.loads(Path(a.case).read_text())
(out/'worker.pid').write_text(str(os.getpid()))
time.sleep(float(c['parameters'].get('sleep',os.environ.get('RUNNER_TEST_SLEEP','0'))))
if c['parameters'].get('crash') and os.environ.get('RUNNER_TEST_FAIL')=='1':
    raise SystemExit(7)
(out/'metrics.json').write_text(json.dumps({'outcome':'rejected','value':c['parameters'].get('value',0),'python_prefix':sys.prefix}))
''')
        (self.root / "builder.py").write_text('''import os,sys
from pathlib import Path
target=Path(sys.argv[1]); (target/'artifact.bin').write_bytes(b'compiled')
counter=Path(os.environ['BUILD_COUNTER'])
counter.write_text(str(int(counter.read_text())+1 if counter.exists() else 1))
''')
        self.git("init", "-q")
        self.git("-c", "user.name=Runner Test", "-c", "user.email=runner@example.invalid", "add", ".")
        self.git("-c", "user.name=Runner Test", "-c", "user.email=runner@example.invalid", "commit", "-qm", "fixture")
        commit = self.git("rev-parse", "HEAD").strip()
        self.experiment = {"id": "reach", "source_commit": commit, "machines": ["local"],
                           "command": ["{python}", "{source}/worker.py", "--case", "{case}", "--output", "{output}"],
                           "cases": [{"id": "near", "parameters": {"value": 1}}]}
        self.save_plan()
        (self.root / "experiments/local.json").write_text(json.dumps({
            "output": str(self.output), "cache": str(self.cache), "python": sys.executable,
            "machine": "local"}))

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], text=True)

    def save_plan(self):
        (self.root / "experiments/plan.json").write_text(json.dumps({"version": 1, "experiments": [self.experiment]}))

    def cli(self, *args, expected=0):
        result = subprocess.run([sys.executable, str(self.root / "experiments/run.py"),
                                 "--plan", str(self.root / "experiments/plan.json"), *args],
                                cwd=self.root, env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def receipts(self):
        return sorted([json.loads(p.read_text()) for p in self.output.glob("*/*/*/attempt-*/receipt.json")
                       if "_preflight" not in p.parts], key=lambda r: r["started"])

    def test_resume_new_case_and_changed_parameters_preserve_results(self):
        self.cli()
        self.cli()
        self.assertEqual(len(self.receipts()), 1)
        self.assertEqual(self.receipts()[0]["status"], "completed")  # Scientific rejection is complete.
        self.experiment["cases"].append({"id": "far", "parameters": {"value": 2}})
        self.save_plan()
        self.cli()
        self.assertEqual(len(self.receipts()), 2)
        self.experiment["cases"][0]["parameters"]["value"] = 3
        self.save_plan()
        self.cli()
        self.assertEqual(len(self.receipts()), 3)
        index = json.loads((self.output / "index.json").read_text())
        self.assertEqual({r["metrics"]["value"] for r in index}, {1, 2, 3})

    def test_failed_execution_requires_explicit_retry(self):
        self.experiment["cases"][0]["parameters"]["crash"] = True
        self.save_plan()
        self.env["RUNNER_TEST_FAIL"] = "1"
        self.cli(expected=1)
        self.env["RUNNER_TEST_FAIL"] = "0"
        self.cli(expected=1)
        self.assertEqual(len(self.receipts()), 1)
        self.cli("--retry-failed")
        self.assertEqual([r["status"] for r in self.receipts()], ["failed", "completed"])

    def test_virtual_environment_preserved_for_configured_and_default_python(self):
        venv = Path(self.temporary.name) / "venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--symlinks", str(venv)], check=True)
        python = venv / "bin/python"
        self.assertTrue(python.is_symlink())
        config_path = self.root / "experiments/local.json"
        config = json.loads(config_path.read_text())
        for configured in (True, False):
            with self.subTest(configured=configured):
                if configured:
                    config["python"] = str(python)
                else:
                    config.pop("python")
                config_path.write_text(json.dumps(config))
                self.experiment["cases"][0]["parameters"]["value"] = int(configured)
                self.save_plan()
                result = subprocess.run([sys.executable if configured else str(python),
                                         str(self.root / "experiments/run.py"),
                                         "--plan", str(self.root / "experiments/plan.json")],
                                        cwd=self.root, env=self.env, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for path in self.output.glob("*/*/*/attempt-*/metrics.json"):
                    self.assertEqual(json.loads(path.read_text())["python_prefix"], str(venv))
                for receipt in self.receipts():
                    self.assertEqual(receipt["command"][0], str(python))

    def test_frozen_inputs_are_checked_and_collected(self):
        (self.root / "inputs").mkdir()
        path = self.root / "inputs/task.bin"
        path.write_bytes(b"frozen")
        self.experiment["inputs"] = {"inputs/task.bin": hashlib.sha256(b"frozen").hexdigest()}
        self.save_plan()
        self.cli()
        copied = list(self.output.glob("*/*/_inputs/*/inputs/task.bin"))
        self.assertEqual(copied[0].read_bytes(), b"frozen")
        path.write_bytes(b"changed")
        self.cli()  # Already retained evidence needs neither inputs nor builds.
        self.experiment["cases"].append({"id": "pending", "parameters": {}})
        self.save_plan()
        self.cli(expected=1)
        self.assertEqual(len(self.receipts()), 2)
        self.assertEqual(self.receipts()[-1]["status"], "blocked")

    def test_build_shared_across_cases_and_rebuilt_if_artifact_changes(self):
        self.env["BUILD_COUNTER"] = str(self.root / "build-count")
        self.experiment["build"] = {
            "commands": [["{python}", "{source}/builder.py", "{build}"]], "artifacts": ["artifact.bin"]}
        self.experiment["cases"].append({"id": "far", "parameters": {}})
        self.save_plan()
        self.cli()
        self.assertEqual((self.root / "build-count").read_text(), "1")
        next(self.cache.glob("builds/*/artifact.bin")).write_bytes(b"stale")
        self.experiment["cases"].append({"id": "other", "parameters": {}})
        self.save_plan()
        self.cli()
        self.assertEqual((self.root / "build-count").read_text(), "2")

    def test_input_changed_during_build_is_not_executed(self):
        path = self.root / "task.bin"
        path.write_bytes(b"frozen")
        self.experiment["inputs"] = {"task.bin": hashlib.sha256(b"frozen").hexdigest()}
        self.experiment["build"] = {
            "commands": [["{python}", "-c", f"from pathlib import Path; "
                          f"Path({str(path)!r}).write_bytes(b'changed'); "
                          "Path('{build}/ready').write_text('built')"]],
            "artifacts": ["ready"]}
        self.save_plan()
        self.cli(expected=1)
        self.assertEqual(self.receipts()[0]["status"], "blocked")
        self.assertIn("input changed", self.receipts()[0]["error"])
        self.assertFalse(list(self.output.glob("*/*/*/attempt-*/worker.pid")))

    def test_interrupt_stops_worker_and_new_invocation_retries(self):
        self.env["RUNNER_TEST_SLEEP"] = "20"
        process = subprocess.Popen([sys.executable, str(self.root / "experiments/run.py"),
                                         "--plan", str(self.root / "experiments/plan.json")],
                                   cwd=self.root, env=self.env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: process.kill() if process.poll() is None else None)
        deadline = time.monotonic() + 10
        pid_files = []
        while time.monotonic() < deadline:
            pid_files = list(self.output.glob("*/*/*/attempt-*/worker.pid"))
            if pid_files:
                break
            time.sleep(.05)
        self.assertTrue(pid_files, "Worker did not start")
        worker_pid = int(pid_files[0].read_text())
        # Different output/cache folders still share the repository execution lock.
        config_path = self.root / "experiments/local.json"
        config = json.loads(config_path.read_text())
        config["cache"] = str(self.cache.parent / "other-cache")
        config_path.write_text(json.dumps(config))
        self.cli("--output", str(self.output.parent / "other-results"), expected=1)
        config["cache"] = str(self.cache)
        config_path.write_text(json.dumps(config))
        process.send_signal(signal.SIGINT)
        self.assertEqual(process.wait(timeout=10), 130)
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)
        self.assertEqual(self.receipts()[0]["status"], "interrupted")
        self.env["RUNNER_TEST_SLEEP"] = "0"
        self.cli()
        self.assertEqual([r["status"] for r in self.receipts()], ["interrupted", "completed"])

    def test_upload_retry_does_not_rerun_completed_cases(self):
        bin_dir = self.root / "fake-bin"
        bin_dir.mkdir()
        rsync = bin_dir / "rsync"
        rsync.write_text(f'#!{sys.executable}\nimport os,sys\nsys.exit(int(os.environ.get("UPLOAD_FAIL", "0")))\n')
        rsync.chmod(0o755)
        self.env["PATH"] = str(bin_dir) + os.pathsep + self.env["PATH"]
        config_path = self.root / "experiments/local.json"
        config = json.loads(config_path.read_text())
        config["sync_to"] = "desktop:/data/gato-results"
        config_path.write_text(json.dumps(config))
        self.env["UPLOAD_FAIL"] = "1"
        self.cli(expected=1)
        self.env["UPLOAD_FAIL"] = "0"
        self.cli()
        self.assertEqual(len(self.receipts()), 1)


    def test_case_error_continues_independent_cases(self):
        self.env["RUNNER_TEST_FAIL"] = "1"
        self.experiment["cases"][0]["parameters"]["crash"] = True
        self.experiment["cases"].append({"id": "independent", "parameters": {}})
        self.save_plan()
        self.cli(expected=1)
        self.assertEqual([r["status"] for r in self.receipts()], ["failed", "completed"])

    def test_failed_build_blocks_only_shared_build_dependents(self):
        self.experiment["build"] = {"commands": [["{python}", "-c", "raise SystemExit(9)"]],
                                    "artifacts": ["missing"]}
        self.experiment["cases"].append({"id": "same-build", "parameters": {}})
        independent = dict(self.experiment, id="other", build={})
        plan = {"version": 1, "experiments": [self.experiment, independent]}
        (self.root / "experiments/plan.json").write_text(json.dumps(plan))
        self.cli(expected=1)
        self.assertEqual([r["status"] for r in self.receipts()], ["blocked", "blocked", "completed", "completed"])
        self.assertEqual(len(list(self.cache.glob("builds/*/stderr.log"))), 1)

    def test_preflight_runs_once_and_failure_holds_only_its_family(self):
        counter = self.root / "preflight-count"
        self.experiment["preflight"] = {"command": ["{python}", "-c",
            f"from pathlib import Path; p=Path({str(counter)!r}); "
            "p.write_text(str(int(p.read_text())+1 if p.exists() else 1)); raise SystemExit(4)"]}
        self.experiment["cases"].append({"id": "far", "parameters": {}})
        independent = {key: value for key, value in self.experiment.items() if key != "preflight"}
        independent["id"] = "independent"
        (self.root / "experiments/plan.json").write_text(json.dumps({
            "version": 1, "experiments": [self.experiment, independent]}))
        self.cli(expected=1)
        self.assertEqual(counter.read_text(), "1")
        self.assertEqual([r["status"] for r in self.receipts()], ["blocked", "blocked", "completed", "completed"])
        pf = list(self.output.glob("local/_preflight/*/attempt-*/receipt.json"))
        self.assertEqual(json.loads(pf[0].read_text())["status"], "failed")
        self.assertTrue((pf[0].parent / "stderr.log").exists())
        independent["cases"] = list(independent["cases"])
        self.experiment["cases"].append({"id": "new", "parameters": {}})
        (self.root / "experiments/plan.json").write_text(json.dumps({
            "version": 1, "experiments": [self.experiment, independent]}))
        self.cli(expected=1)  # Newly pending case does not implicitly retry failed preflight.
        self.assertEqual(counter.read_text(), "1")

    def test_successful_preflight_is_shared_by_explicit_group(self):
        counter = self.root / "preflight-count"
        self.experiment["preflight"] = {"group": "common", "command": ["{python}", "-c",
            f"from pathlib import Path; p=Path({str(counter)!r}); "
            "p.write_text(str(int(p.read_text())+1 if p.exists() else 1))"]}
        independent = dict(self.experiment, id="other")
        (self.root / "experiments/plan.json").write_text(json.dumps({
            "version": 1, "experiments": [self.experiment, independent]}))
        self.cli()
        self.assertEqual(counter.read_text(), "1")
        self.assertEqual(len(self.receipts()), 2)

    def test_preflight_resume_reuses_success_and_changed_configuration_runs_anew(self):
        counter = self.root / "preflight-count"
        self.experiment["preflight"] = {"command": ["{python}", "-c",
            f"from pathlib import Path; p=Path({str(counter)!r}); "
            "p.write_text(str(int(p.read_text())+1 if p.exists() else 1))"]}
        self.save_plan()
        self.cli()
        self.experiment["cases"].append({"id": "new", "parameters": {}})
        self.save_plan()
        self.cli()
        self.assertEqual(counter.read_text(), "1")
        self.experiment["timing_mode"] = "B"
        self.save_plan()
        self.cli()
        self.assertEqual(counter.read_text(), "2")

    def test_bookkeeping_and_environment_changes_do_not_rerun_evidence(self):
        self.cli()
        runner = self.root / "experiments/run.py"
        runner.write_text(runner.read_text() + "\n# Bookkeeping-only edit.\n")
        (self.root / "experiments/adapters/unrelated.py").write_text("# Unrelated adapter.\n")
        self.experiment.update(description="New prose", research_question="RQ-timing")
        self.save_plan()
        config_path = self.root / "experiments/local.json"
        config = json.loads(config_path.read_text())
        config["python"] = "/nonexistent/unrelated-python"
        config_path.write_text(json.dumps(config))
        self.cli()
        self.assertEqual(len(self.receipts()), 1)
        self.assertIn("packages", self.receipts()[0]["environment"])
        self.experiment["protocol"] = "changed-scientific-protocol"
        self.save_plan()
        self.cli("--dry-run")
        self.assertIn("pending: reach/near/0", self.cli("--dry-run").stdout)

    def test_registered_import_suppresses_build_preflight_and_execution(self):
        retained = self.output / "retained-jetson"
        retained.mkdir(parents=True)
        (retained / "raw.json").write_text('{"timing":1}')
        entry = {"machine": "local", "experiment": "reach", "status": "completed",
                 "source_commit": self.experiment["source_commit"], "protocol": None, "inputs": {},
                 "case": dict(self.experiment["cases"][0], repeat=0), "path": "retained-jetson"}
        (self.output / "imports.json").write_text(json.dumps([entry]))
        self.experiment["build"] = {"commands": [["missing-build-tool"]], "artifacts": ["missing"]}
        self.experiment["preflight"] = {"command": ["missing-preflight-tool"]}
        self.save_plan()
        self.cli()
        self.assertEqual(self.receipts(), [])
        self.assertFalse((self.cache / "builds").exists())
        self.assertFalse((self.output / "local/_preflight").exists())
        shutil.rmtree(retained)
        result = self.cli()
        self.assertIn("keeping its completed status", result.stderr)
        self.assertEqual(self.receipts(), [])

    def test_corrupt_import_registration_holds_to_avoid_accidental_rerun(self):
        self.output.mkdir()
        (self.output / "imports.json").write_text('{"not": "a completion registry"}')
        result = self.cli(expected=1)
        self.assertIn("Cannot determine saved progress", result.stderr)
        self.assertEqual(self.receipts(), [])
        self.assertFalse(self.cache.exists())

    def test_dry_run_does_not_fetch_or_create_cache(self):
        self.experiment["source_commit"] = "a" * 40
        self.save_plan()
        self.cli("--dry-run")
        self.assertFalse(self.cache.exists())
        self.assertFalse(self.output.exists())

    def test_legacy_receipt_identity_is_reused(self):
        self.cli()
        receipt_path = next(self.output.glob("*/*/*/attempt-*/receipt.json"))
        receipt = json.loads(receipt_path.read_text())
        receipt["identity"] = {"experiment": {key: value for key, value in self.experiment.items()
                                               if key != "cases"}, "case": receipt["case"],
                               "runner": {"run.py": "historical"}, "environment": {"old": "packages"}}
        receipt_path.write_text(json.dumps(receipt))
        self.cli()
        self.assertEqual(len(self.receipts()), 1)

    def test_declared_build_helper_is_validated_and_part_of_build_key(self):
        self.env["BUILD_COUNTER"] = str(self.root / "build-count")
        helper = self.root / "experiments/build-helper.txt"
        helper.write_bytes(b"original recipe")
        self.experiment["build"] = {
            "commands": [["{python}", "{source}/builder.py", "{build}"]],
            "artifacts": ["artifact.bin"],
            "inputs": {"build-helper.txt": hashlib.sha256(helper.read_bytes()).hexdigest()}}
        self.save_plan()
        self.cli()
        helper.write_bytes(b"changed recipe")
        self.experiment["cases"].append({"id": "new", "parameters": {}})
        self.save_plan()
        self.cli(expected=1)
        self.assertEqual(self.receipts()[-1]["status"], "blocked")
        self.assertIn("build input hash mismatch", self.receipts()[-1]["error"])
        self.experiment["build"]["inputs"]["build-helper.txt"] = hashlib.sha256(helper.read_bytes()).hexdigest()
        self.save_plan()
        self.cli("--retry-failed")
        self.assertEqual((self.root / "build-count").read_text(), "2")
        self.assertEqual(self.receipts()[-1]["status"], "completed")

    def test_timing_mode_and_backend_change_scientific_identity(self):
        self.experiment.update(timing_mode="A", backend="internal")
        self.save_plan()
        self.cli()
        self.experiment["timing_mode"] = "B"
        self.save_plan()
        self.cli()
        self.experiment["backend"] = "gazebo"
        self.save_plan()
        self.cli()
        self.assertEqual(len(self.receipts()), 3)
        self.assertEqual({(r["identity"]["timing_mode"], r["identity"]["backend"]) for r in self.receipts()},
                         {("A", "internal"), ("B", "internal"), ("B", "gazebo")})

    def test_wall_timeout_stops_only_current_case_then_continues(self):
        self.experiment["timeout_seconds"] = .2
        self.experiment["cases"][0]["parameters"]["sleep"] = 20
        self.experiment["cases"].append({"id": "independent", "parameters": {}})
        self.save_plan()
        self.cli(expected=1)
        self.assertEqual([r["status"] for r in self.receipts()], ["failed", "completed"])
        self.assertIn("Execution wall-clock timeout", self.receipts()[0]["error"])
        pid_files = list(self.output.glob("*/*/*/attempt-*/worker.pid"))
        for pid in pid_files:
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    def test_explicit_plan_is_required(self):
        result = subprocess.run([sys.executable, str(self.root / "experiments/run.py")],
                                cwd=self.root, env=self.env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--plan is required", result.stderr)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()

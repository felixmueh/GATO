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
time.sleep(float(os.environ.get('RUNNER_TEST_SLEEP','0')))
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
        result = subprocess.run([sys.executable, str(self.root / "experiments/run.py"), *args],
                                cwd=self.root, env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def receipts(self):
        return [json.loads(p.read_text()) for p in sorted(self.output.glob("*/*/*/attempt-*/receipt.json"))]

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
                                         str(self.root / "experiments/run.py")],
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
        copied = list(self.output.glob("*/*/*/attempt-*/inputs/inputs/task.bin"))
        self.assertEqual(copied[0].read_bytes(), b"frozen")
        path.write_bytes(b"changed")
        self.cli(expected=1)
        self.assertEqual(len(self.receipts()), 1)

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
        self.assertEqual(self.receipts()[0]["status"], "failed")
        self.assertIn("input changed", self.receipts()[0]["error"])
        self.assertFalse(list(self.output.glob("*/*/*/attempt-*/worker.pid")))

    def test_interrupt_stops_worker_and_new_invocation_retries(self):
        self.env["RUNNER_TEST_SLEEP"] = "20"
        process = subprocess.Popen([sys.executable, str(self.root / "experiments/run.py")],
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
        # A different output directory still shares the same execution lock.
        self.cli("--output", str(self.output.parent / "other-results"), expected=1)
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


if __name__ == "__main__":
    unittest.main()

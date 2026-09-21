"""Exercise shell profile switching and Docker reuse without ROS or a Docker daemon."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tiago_tools"
ENV_KEYS = (
    "GATO_ROS_PROFILE", "ROS_DOMAIN_ID", "RMW_IMPLEMENTATION",
    "CYCLONEDDS_URI", "ROS_LOCALHOST_ONLY", "USER_RC_LOADED",
)


def shell_environment(tmp_path, commands, *, interactive=False, profile="tiago"):
    env = os.environ.copy()
    env.update(
        HOME=str(tmp_path), GATO_ROS_PROFILE=profile, ROS_DOMAIN_ID="99",
        RMW_IMPLEMENTATION="rmw_fastrtps_cpp", CYCLONEDDS_URI="/stale.xml",
        ROS_LOCALHOST_ONLY="1",
    )
    dump = "import json, os; print('PROFILE_RESULT=' + json.dumps({k: os.environ.get(k) for k in " + repr(ENV_KEYS) + "}))"
    commands += f"\n{shlex.quote(sys.executable)} -c {shlex.quote(dump)}"
    args = ["bash", "--noprofile"]
    if interactive:
        args += ["--rcfile", str(TOOLS / "lib/ros_shell.bash"), "-i"]
    else:
        args += ["--norc"]
    result = subprocess.run(
        args + ["-c", commands], env=env, cwd=tmp_path,
        text=True, capture_output=True, check=True,
    )
    line = next(line for line in result.stdout.splitlines() if line.startswith("PROFILE_RESULT="))
    return json.loads(line.removeprefix("PROFILE_RESULT="))


@pytest.mark.parametrize("profile,domain,localhost,xml", [
    ("simulation", "1", "0", "cyclone_pal_loopback.xml"),
    ("tiago", "2", None, "cyclone_tiago_ethernet.xml"),
])
def test_switch_profiles_in_existing_shell(tmp_path, profile, domain, localhost, xml):
    other = "tiago" if profile == "simulation" else "simulation"
    commands = "set -eu\n" + "\n".join(
        f"source {shlex.quote(str(TOOLS / ('ros_' + selected + '.sh')))}"
        for selected in [other, profile, profile]
    )
    env = shell_environment(tmp_path, commands)
    assert env["GATO_ROS_PROFILE"] == profile
    assert env["ROS_DOMAIN_ID"] == domain
    assert env["ROS_LOCALHOST_ONLY"] == localhost
    assert env["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert env["CYCLONEDDS_URI"] == str(TOOLS / xml)


def test_launcher_shell_preserves_bashrc_but_profile_wins(tmp_path):
    (tmp_path / ".bashrc").write_text(
        "export USER_RC_LOADED=yes\n"
        "export GATO_ROS_PROFILE=simulation ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1\n"
        "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp CYCLONEDDS_URI=/stale.xml\n"
    )
    env = shell_environment(tmp_path, ":", interactive=True)
    assert env["USER_RC_LOADED"] == "yes"
    assert env["GATO_ROS_PROFILE"] == "tiago"
    assert env["ROS_DOMAIN_ID"] == "2"
    assert env["ROS_LOCALHOST_ONLY"] is None
    assert env["RMW_IMPLEMENTATION"] == "rmw_cyclonedds_cpp"
    assert env["CYCLONEDDS_URI"] == str(TOOLS / "cyclone_tiago_ethernet.xml")


@pytest.mark.parametrize("profile", ["simulation", "tiago"])
def test_profiles_require_source(profile):
    result = subprocess.run(
        ["bash", str(TOOLS / f"ros_{profile}.sh")], text=True, capture_output=True,
    )
    assert result.returncode == 1
    assert "Use: source" in result.stderr


@pytest.fixture
def docker_mock(tmp_path):
    # Record Docker's real argv, rather than interpreting a shell command string.
    # Any unexpected lifecycle operation fails instead of touching a real daemon.
    binary = tmp_path / "docker"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
        "if args[:2] == ['image', 'inspect']: pass\n"
        "elif args[:1] == ['ps']: print('existing' if os.environ['CONTAINER_EXISTS'] == '1' else '')\n"
        "elif args[0] in ['create', 'start', 'exec']: pass\n"
        "else: sys.exit('Unexpected Docker operation: ' + repr(args))\n"
    )
    binary.chmod(0o755)
    xhost = tmp_path / "xhost"
    xhost.write_text("#!/bin/sh\nexit 0\n")
    xhost.chmod(0o755)
    log = tmp_path / "docker_calls.jsonl"
    env = os.environ.copy()
    env.update(PATH=f"{tmp_path}:{env['PATH']}", DOCKER_CALLS=str(log))

    def run(*args, exists=False, overlay=False):
        env["CONTAINER_EXISTS"] = "1" if exists else "0"
        if log.exists():
            log.unlink()
        if overlay:
            command = ["bash", "-c", 'set -Eeuo pipefail; REPO_ROOT="$1"; source "$1/tiago_tools/lib/docker.sh"; CONTAINER_REPO_ROOT=/workspace/GATO; docker_main "${@:2}"', "test-launcher", str(ROOT)]
        else:
            command = ["bash", str(TOOLS / "docker.sh")]
        result = subprocess.run(command + list(args), env=env, cwd=tmp_path, text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize("exists", [False, True])
def test_tiago_profile_applies_on_creation_and_reuse(docker_mock, exists):
    result, calls = docker_mock("--target", "jetson", "--ros-profile", "tiago", exists=exists)
    assert result.returncode == 0, result.stderr
    create = [call for call in calls if call[0] == "create"]
    assert bool(create) is not exists
    shells = create + [call for call in calls if call[0] == "exec"]
    assert len(shells) == (1 if exists else 2)
    for call in shells:
        assert "GATO_ROS_PROFILE=tiago" in call
        assert call[-3:] == ["/bin/bash", "--rcfile", "/workspace/tiago_tools/lib/ros_shell.bash"]
    assert not any(call[0] in ["rm", "stop", "build"] for call in calls)


def test_default_simulation_and_overlay_path(docker_mock):
    result, calls = docker_mock(exists=True, overlay=True)
    assert result.returncode == 0, result.stderr
    shell = next(call for call in calls if call[0] == "exec")
    assert "GATO_ROS_PROFILE=simulation" in shell
    assert shell[-1] == "/workspace/GATO/tiago_tools/lib/ros_shell.bash"


@pytest.mark.parametrize("args", [("--ros-profile",), ("--ros-profile", "unknown")])
def test_invalid_profile_fails_before_docker(docker_mock, args):
    result, calls = docker_mock(*args)
    assert result.returncode != 0
    assert "profile" in result.stderr
    assert calls == []


def test_no_attach_does_not_claim_to_reconfigure_existing_shells(docker_mock):
    result, calls = docker_mock("--ros-profile", "tiago", "--no-attach", exists=True)
    assert result.returncode == 0, result.stderr
    assert not any(call[0] in ["create", "exec", "stop", "rm"] for call in calls)
    assert "Existing shells keep their settings" in result.stdout

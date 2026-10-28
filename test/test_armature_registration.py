"""Host-only API/cache checks, independent of the experimental GRiD emitter.

The constructor stand-in isolates GATO's forwarding; numerical and generated
dynamics qualification belongs to GRiD's own armature tests.
"""
import importlib
import json
from pathlib import Path

import pytest

import gato
from gato import _cache, builder


@pytest.fixture
def arm_urdf(tmp_path):
    links, joints = [], []
    for i in (1, 2):
        links.append(f'''<link name="l{i}"><inertial><mass value="1"/>
          <inertia ixx=".02" iyy=".03" izz=".04" ixy="0" ixz="0" iyz="0"/>
          </inertial></link>''')
        parent = "base" if i == 1 else "l1"
        joints.append(f'''<joint name="j{i}" type="revolute">
          <parent link="{parent}"/><child link="l{i}"/>
          <origin xyz=".2 0 0"/><axis xyz="0 0 1"/>
          <limit lower="-2" upper="2" effort="20" velocity="3"/>
          <dynamics damping=".2" friction=".1"/></joint>''')
    urdf = tmp_path / "arm.urdf"
    urdf.write_text('<robot name="arm"><link name="base"/>' + ''.join(links + joints)
                    + '<link name="tool"/><joint name="EE" type="fixed">'
                      '<parent link="l2"/><child link="tool"/></joint></robot>')
    return urdf


@pytest.fixture
def emitter(monkeypatch):
    calls = []

    class Generator:
        def __init__(self, robot, **kwargs):
            calls.append(kwargs)
            self.kwargs = kwargs
            self._cost_frames = [("EE",)]

        def gen_all_code(self, **kwargs):
            Path(kwargs["output_path"]).write_text(json.dumps(self.kwargs, sort_keys=True))

    module = importlib.import_module("grid_codegen.GRiDCodeGenerator")
    monkeypatch.setattr(module, "GRiDCodeGenerator", Generator)
    return calls


def test_registration_records_effective_values_and_preserves_cache(arm_urdf, tmp_path, emitter, monkeypatch):
    cache = tmp_path / "cache"
    kwargs = dict(cache_dir_override=cache, collision_res=None, contact_frames=[])
    values = [.4, .2]
    meta = gato.register_robot("armaturebot", arm_urdf, joint_armature=values, **kwargs)
    entry = _cache.robot_entry("armaturebot", cache)
    key = entry["codegen_key"]
    assert emitter[-1]["JOINT_ARMATURE"] == (.4, .2)
    assert meta["joint_armature"] == entry["options"]["joint_armature"] == [.4, .2]
    assert meta["grid_codegen_source_sha256"] == builder._grid_codegen_source_id()
    values[0] = 9
    assert _cache.robot_entry("armaturebot", cache)["meta"]["joint_armature"] == [.4, .2]
    assert emitter[-1]["JOINT_ARMATURE"] == (.4, .2)

    # A one-shot iterable is normalized before both keying and generation.
    gato.register_robot("armaturebot", arm_urdf, joint_armature=iter([.4, .2]), **kwargs)
    assert len(emitter) == 1
    assert _cache.robot_entry("armaturebot", cache)["codegen_key"] == key

    gato.register_robot("armaturebot", arm_urdf, joint_armature=[.41, .2], **kwargs)
    assert _cache.robot_entry("armaturebot", cache)["codegen_key"] != key
    assert (cache / "robots" / key / "armaturebot" / "grid.cuh").exists()
    changed_key = _cache.robot_entry("armaturebot", cache)["codegen_key"]
    monkeypatch.setattr(builder, "_grid_codegen_source_id", lambda: "changed-emitter")
    fresh = gato.register_robot("armaturebot", arm_urdf, joint_armature=[.41, .2], **kwargs)
    assert _cache.robot_entry("armaturebot", cache)["codegen_key"] != changed_key
    assert fresh["grid_codegen_source_sha256"] == "changed-emitter"


def test_none_and_zero_are_neutral(arm_urdf, tmp_path, emitter, monkeypatch):
    cache = tmp_path / "cache"
    kwargs = dict(cache_dir_override=cache, collision_res=None, contact_frames=[])
    default = gato.register_robot("neutralbot", arm_urdf, **kwargs)
    key = _cache.robot_entry("neutralbot", cache)["codegen_key"]

    def should_not_hash():
        pytest.fail("neutral armature must not change the legacy emitter identity")

    monkeypatch.setattr(builder, "_grid_codegen_source_id", should_not_hash)
    zero = gato.register_robot("neutralbot", arm_urdf, joint_armature=[0, -0.0], **kwargs)
    assert zero == default
    entry = _cache.robot_entry("neutralbot", cache)
    assert entry["codegen_key"] == key
    assert "joint_armature" not in entry["options"]
    assert "joint_armature" not in zero and "grid_codegen_source_sha256" not in zero
    assert len(emitter) == 1 and "JOINT_ARMATURE" not in emitter[0]
    # The direct codegen/key APIs also normalize an explicit zero vector.
    assert builder._codegen_key(arm_urdf, "EE", None, [], False, joint_armature=[0, 0]) == key
    direct = builder.codegen(arm_urdf, "neutralbot", out_dir=tmp_path / "direct", register=False,
                             collision_res=None, contact_frames=[], joint_armature=[0, 0])
    assert "joint_armature" not in direct and "JOINT_ARMATURE" not in emitter[-1]


@pytest.mark.parametrize("bad", ([0], [0, 0, 0], [-.1, .2], [float("nan"), 0],
                                 [0, float("inf")], [0, "bad"], .1, "00"))
def test_invalid_armature_rejected_before_generation(arm_urdf, tmp_path, emitter, bad):
    for invoke in (
        lambda: gato.register_robot("badarm", arm_urdf, joint_armature=bad,
                                    cache_dir_override=tmp_path / "cache"),
        lambda: builder.codegen(arm_urdf, "badarm", out_dir=tmp_path / "out",
                                 register=False, joint_armature=bad),
        lambda: builder._codegen_key(arm_urdf, "EE", None, [], False, joint_armature=bad),
    ):
        with pytest.raises(ValueError, match="joint_armature"):
            invoke()
    assert not emitter


def test_armature_and_friction_are_forwarded_together(arm_urdf, tmp_path, emitter):
    meta = builder.codegen(arm_urdf, "frictionarm", out_dir=tmp_path / "out", register=False,
                           collision_res=None, contact_frames=[], joint_armature=[.4, .2],
                           joint_dynamics=True, friction_width=.1)
    assert emitter[-1]["JOINT_ARMATURE"] == (.4, .2)
    assert emitter[-1]["USE_JOINT_DYNAMICS"] is True
    assert emitter[-1]["FRICTION_MODEL"] == "tanh" and emitter[-1]["FRICTION_WIDTH"] == .1
    assert meta["joint_armature"] == [.4, .2] and meta["friction_width"] == .1


def test_topology_validation_is_delegated(arm_urdf, tmp_path, monkeypatch):
    def unsupported(robot, **kwargs):
        assert kwargs["JOINT_ARMATURE"] == (.4, .2)
        raise NotImplementedError("JOINT_ARMATURE unsupported topology")

    module = importlib.import_module("grid_codegen.GRiDCodeGenerator")
    monkeypatch.setattr(module, "GRiDCodeGenerator", unsupported)
    with pytest.raises(NotImplementedError, match="unsupported topology"):
        builder.codegen(arm_urdf, "unsupported", out_dir=tmp_path / "out", register=False,
                        collision_res=None, joint_armature=[.4, .2])


def test_source_identity_tracks_contents_and_relative_paths(tmp_path, monkeypatch):
    import grid_codegen

    def package(path):
        path.mkdir()
        (path / "__init__.py").write_text("# emitter\n")
        (path / "algorithm.py").write_text("value = 1\n")
        return path / "__init__.py"

    first, second = package(tmp_path / "first"), package(tmp_path / "second")
    monkeypatch.setattr(grid_codegen, "__file__", str(first))
    digest = builder._grid_codegen_source_id()
    monkeypatch.setattr(grid_codegen, "__file__", str(second))
    assert builder._grid_codegen_source_id() == digest
    (second.parent / "algorithm.py").write_text("value = 2\n")
    assert builder._grid_codegen_source_id() != digest


def test_checkout_build_forwards_armature_before_compilation(arm_urdf, tmp_path, emitter, monkeypatch):
    monkeypatch.setattr(builder, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(builder, "load_registry", lambda: {})
    recorded = []
    monkeypatch.setattr(builder, "_update_registry", lambda name, meta: recorded.append(meta))

    class BeforeCompile(Exception):
        pass

    def stop_at_cmake(*args, **kwargs):
        raise BeforeCompile

    monkeypatch.setattr(builder.subprocess, "run", stop_at_cmake)
    with pytest.raises(BeforeCompile):
        builder.build(arm_urdf, name="checkoutarm", joint_armature=iter([.4, .2]),
                      collision_res=None, contact_frames=[])
    assert emitter[-1]["JOINT_ARMATURE"] == (.4, .2)
    assert recorded[-1]["joint_armature"] == [.4, .2]
    assert recorded[-1]["codegen_key"] == builder._codegen_key(
        arm_urdf, "EE", None, [], False, joint_armature=[.4, .2])

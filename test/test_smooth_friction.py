"""Host checks for opt-in model identity, tables and unsupported configurations."""
from types import SimpleNamespace
import pytest
from gato import builder, register_robot


def robot():
    joints = [SimpleNamespace(jtype="revolute", is_mimic=False, friction=.3, damping=.15)]
    return SimpleNamespace(get_joints_ordered_by_id=lambda: joints)


def test_smoothing_explicit_and_validated():
    assert builder._joint_friction_config(robot(), None) is None
    assert builder._joint_friction_config(robot(), 1.) == [(.3, .15, 1.)]
    assert builder._joint_friction_config(robot(), [.5]) == [(.3, .15, .5)]
    for width in (0, -1, float("nan"), float("inf"), [], [1, 2]):
        with pytest.raises(ValueError, match="positive finite"):
            builder._joint_friction_config(robot(), width)
    for width in (True, "1", ["1"]):
        with pytest.raises(ValueError, match="numeric scalar"):
            builder._joint_friction_config(robot(), width)
    r = robot(); r.get_joints_ordered_by_id()[0].damping = -1
    with pytest.raises(ValueError, match="nonnegative"):
        builder._joint_friction_config(r, 1.)
    r = robot(); r.get_joints_ordered_by_id()[0].jtype = "floating"
    with pytest.raises(ValueError, match="scalar joints"):
        builder._joint_friction_config(r, 1.)


def test_smoothing_changes_codegen_identity_only_when_enabled(urdfs):
    args = (urdfs["indy7"], "EE", .15, None, False)
    original = builder._codegen_key(*args)
    assert builder._codegen_key(*args, None) == original
    assert builder._codegen_key(*args, 1.) != original
    assert builder._codegen_key(*args, .5) == builder._codegen_key(*args, 1.)


@pytest.mark.parametrize("option", [{"floating_base": True}, {"variant": "eh"}, {"variant": "fc"}])
def test_unsupported_variants_fail_before_codegen(option):
    with pytest.raises(ValueError, match="first-order"):
        register_robot("unused", "does-not-exist.urdf", joint_friction_smoothing=1., **option)


def test_width_validation_and_setter_copy():
    import numpy as np
    from gato import BSQP
    from gato._friction import smoothing_widths
    for value in (0., -1., np.nan, np.inf, 1e-100, 1e100, [], [1., 2., 3.]):
        with pytest.raises(ValueError, match="positive finite"):
            smoothing_widths(value, 2)
    for value in (True, "1", ["1"], [[1., 2.]], 1j):
        with pytest.raises(ValueError, match="numeric scalar"):
            smoothing_widths(value, 2)
    uploads = []
    solver = BSQP.__new__(BSQP)
    solver.lib = SimpleNamespace(SMOOTH_JOINT_FRICTION=True)
    solver.nq = 2
    solver.solver = SimpleNamespace(set_joint_friction_smoothing=lambda values: uploads.append(values.copy()))
    values = np.array([.03, .06], dtype=np.float32)
    solver.set_joint_friction_smoothing(values)
    values[:] = 4
    np.testing.assert_array_equal(uploads[-1], np.array([.03, .06], dtype=np.float32))
    previous = solver.joint_friction_smoothing.copy()
    with pytest.raises(ValueError):
        solver.set_joint_friction_smoothing([.03, np.nan])
    np.testing.assert_array_equal(solver.joint_friction_smoothing, previous)
    assert len(uploads) == 1
    solver.set_joint_friction_smoothing(.25)
    np.testing.assert_array_equal(uploads[-1], [.25, .25])
    solver.lib.SMOOTH_JOINT_FRICTION = False
    with pytest.raises(ValueError, match="rebuild"):
        solver.set_joint_friction_smoothing(.1)


def test_reregister_changes_default_without_codegen_or_module_identity(tmp_path, monkeypatch, urdfs):
    import numpy as np
    from gato import _cache, robot_info
    cache = tmp_path / "cache"
    monkeypatch.setenv("GATO_CACHE_DIR", str(cache))
    name = "runtime_friction_indy"
    register_robot(name, urdfs["indy7"], N=[8], collision_res=None, joint_friction_smoothing=.03)
    first = _cache.robot_entry(name)
    directory = cache / "robots" / first["codegen_key"] / name
    headers = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.glob("*.cuh")}
    module_key = _cache.module_key(first, 8, "default", directory.parent, "120")[0]
    def forbid_codegen(*args, **kwargs):
        pytest.fail("Changing runtime widths must not regenerate headers")
    monkeypatch.setattr(builder, "codegen", forbid_codegen)
    widths = np.linspace(.04, .09, 6)
    register_robot(name, urdfs["indy7"], N=[8], collision_res=None, joint_friction_smoothing=widths)
    second = _cache.robot_entry(name)
    assert first["codegen_key"] == second["codegen_key"]
    assert _cache.module_key(second, 8, "default", directory.parent, "120")[0] == module_key
    np.testing.assert_allclose(robot_info(name)["joint_friction_smoothing"], widths)
    assert headers == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in directory.glob("*.cuh")}
    with pytest.raises(ValueError, match="positive finite"):
        register_robot(name, urdfs["indy7"], N=[8], collision_res=None, joint_friction_smoothing=-1.)
    assert _cache.robot_entry(name) == second  # failed update leaves the default intact


def test_constructor_uses_registration_default_and_instance_override(monkeypatch):
    import numpy as np
    from gato import interface
    class Native:
        def __init__(self, *args):
            self.widths = None
        def set_joint_friction_smoothing(self, values):
            self.widths = values.copy()
        def set_rho_adaptation(self, value):
            pass
    module = SimpleNamespace(BSQP_float=Native, NQ=2, NV=2, SMOOTH_JOINT_FRICTION=True)
    monkeypatch.setattr(interface.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(interface, "_pin_available", lambda: False)
    monkeypatch.setattr(interface, "robot_info", lambda name: {"joint_friction_smoothing": [.03, .04]})
    monkeypatch.setattr(interface.BSQP, "set_f_ext_B", lambda *args: None)
    first = interface.BSQP("unused", batch_size=1, N=8, dt=.01, plant_type="mock", linsys="pcg")
    second = interface.BSQP("unused", batch_size=1, N=8, dt=.01, plant_type="mock", linsys="pcg",
                            joint_friction_smoothing=.2)
    np.testing.assert_allclose(first.solver.widths, [.03, .04])
    np.testing.assert_allclose(second.solver.widths, [.2, .2])
    first.set_joint_friction_smoothing(.1)
    np.testing.assert_allclose(second.solver.widths, [.2, .2])
    module.SMOOTH_JOINT_FRICTION = False
    disabled = interface.BSQP("unused", batch_size=1, N=8, dt=.01, plant_type="mock", linsys="pcg")
    assert disabled.solver.widths is None
    with pytest.raises(ValueError, match="rebuild"):
        interface.BSQP("unused", batch_size=1, N=8, dt=.01, plant_type="mock", joint_friction_smoothing=.1)


def test_cached_module_defaults_follow_its_registration_on_name_collision(tmp_path, monkeypatch):
    import numpy as np
    from gato import _cache, interface
    class Native:
        def __init__(self, *args):
            self.widths = None
        def set_joint_friction_smoothing(self, values):
            self.widths = values.copy()
        def set_rho_adaptation(self, value):
            pass
    cache = tmp_path / "cache"
    entry = {"codegen_key": "registered-key", "meta": {"joint_friction_smoothing": [.03, .04]}}
    module = SimpleNamespace(BSQP_float=Native, NQ=2, NV=2, SMOOTH_JOINT_FRICTION=True,
                             _GATO_CACHE_DIR=str(cache), _GATO_CODEGEN_KEY="registered-key")
    loaded = False
    def import_module(name):
        if not loaded:
            raise ImportError("No bundled module at this horizon")
        return module
    def load_module(*args):
        nonlocal loaded
        loaded = True
        return module
    def resolve(name, cache_path):
        assert cache_path == cache  # follows module origin, not a later environment change
        return entry, cache / "robots" / entry["codegen_key"]
    monkeypatch.setattr(interface.importlib, "import_module", import_module)
    monkeypatch.setattr(_cache, "load_module", load_module)
    monkeypatch.setattr(_cache, "resolve", resolve)
    monkeypatch.setattr(interface, "_pin_available", lambda: False)
    monkeypatch.setattr(interface, "robot_info", lambda name: {"joint_friction_smoothing": [1., 1.]})
    monkeypatch.setattr(interface.BSQP, "set_f_ext_B", lambda *args: None)
    first = interface.BSQP("unused", batch_size=1, N=8, dt=.01, plant_type="indy7", linsys="pcg")
    np.testing.assert_allclose(first.solver.widths, [.03, .04])
    entry["meta"]["joint_friction_smoothing"] = [.05, .06]
    # Module now comes from import_module (sys.modules); registration remains authoritative.
    second = interface.BSQP("unused", batch_size=1, N=8, dt=.01, plant_type="indy7", linsys="pcg")
    np.testing.assert_allclose(second.solver.widths, [.05, .06])
    np.testing.assert_allclose(first.solver.widths, [.03, .04])
    entry["codegen_key"] = "different-robot"
    with pytest.raises(ValueError, match="restart the process"):
        interface.BSQP("unused", batch_size=1, N=8, dt=.01, plant_type="indy7", linsys="pcg")


def test_cache_loader_records_module_registration_origin(tmp_path, monkeypatch):
    import sys
    from gato import _cache
    cache = tmp_path / "cache"
    entry = {"codegen_key": "test-key"}
    module = SimpleNamespace()
    monkeypatch.setattr(_cache, "resolve", lambda name, path: (entry, cache / "robots/test-key"))
    monkeypatch.setattr(_cache, "ensure_module", lambda *args, **kwargs: cache / "fake.so")
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda value: None))
    monkeypatch.setattr(_cache.importlib.util, "spec_from_file_location", lambda *args: spec)
    monkeypatch.setattr(_cache.importlib.util, "module_from_spec", lambda value: module)
    fullname = "gato.bsqpN8_test_registration_origin"
    assert fullname not in sys.modules
    try:
        result = _cache.load_module("test_registration_origin", 8, cache=cache)
        assert result._GATO_CACHE_DIR == str(cache.resolve())
        assert result._GATO_CODEGEN_KEY == "test-key"
        assert _cache.load_module("test_registration_origin", 8, cache=cache) is result
    finally:
        sys.modules.pop(fullname, None)

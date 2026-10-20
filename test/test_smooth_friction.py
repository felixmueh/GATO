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
        with pytest.raises(ValueError, match="JSON numeric"):
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
    assert builder._codegen_key(*args, .5) != builder._codegen_key(*args, 1.)


@pytest.mark.parametrize("option", [{"floating_base": True}, {"variant": "eh"}, {"variant": "fc"}])
def test_unsupported_variants_fail_before_codegen(option):
    with pytest.raises(ValueError, match="first-order"):
        register_robot("unused", "does-not-exist.urdf", joint_friction_smoothing=1., **option)

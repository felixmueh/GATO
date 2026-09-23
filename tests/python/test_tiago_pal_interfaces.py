"""Reject incompatible controller APIs before any controller service is called."""

from types import SimpleNamespace

import pytest

from gato_tiago import ros_tiago


class PalSwitchRequest:
    # A ROS-generated request rejects assignments to removed fields.
    __slots__ = ("activate_controllers", "deactivate_controllers", "strictness",
                 "activate_asap", "timeout")


def test_switch_uses_pal_request_and_reports_server_reason():
    arm = ros_tiago.TiagoRightArmClient.__new__(ros_tiago.TiagoRightArmClient)
    arm.controller_manager = "/controller_manager"
    arm.ros = {"SwitchController": SimpleNamespace(Request=PalSwitchRequest),
               "Duration": SimpleNamespace}
    requests = []

    def service_call(name, kind, request, timeout):
        requests.append(request)
        return SimpleNamespace(ok=False, message="hardware interfaces unavailable")

    arm._service_call = service_call
    with pytest.raises(RuntimeError, match="hardware interfaces unavailable"):
        arm._switch_controllers(activate=["arm_right_controller"], deactivate=[],
                                strictness=2, timeout_sec=5.0)
    assert requests[0].activate_asap is True
    assert requests[0].activate_controllers == ["arm_right_controller"]
    assert requests[0].strictness == 2


def test_humble_schema_is_rejected_before_creating_ros_node(monkeypatch):
    # The old package has no data_type; model the actual incompatible import.
    import sys
    from types import ModuleType
    msg = ModuleType("controller_manager_msgs.msg")
    msg.HardwareInterface = SimpleNamespace(
        get_fields_and_field_types=lambda: dict.fromkeys(
            ("name", "is_available", "is_claimed")))
    msg.ControllerState = SimpleNamespace(get_fields_and_field_types=lambda: {})
    srv = ModuleType("controller_manager_msgs.srv")
    srv.SwitchController = SimpleNamespace(Request=PalSwitchRequest, Response=SimpleNamespace)
    monkeypatch.setitem(sys.modules, "controller_manager_msgs.msg", msg)
    monkeypatch.setitem(sys.modules, "controller_manager_msgs.srv", srv)
    with pytest.raises(RuntimeError, match="PAL.*rebuild"):
        ros_tiago.validate_controller_manager_interfaces()

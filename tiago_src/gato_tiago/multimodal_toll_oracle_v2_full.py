"""Frozen full-campaign contract for Tiago tool-center Oracle V2.

All scientific constants and pure mathematics are reused from the accepted
V1 static implementation.  V2 changes only the prerequisite authentication
boundary and artifact namespace after V1's launch-format rejection.
"""

from pathlib import Path

from gato_tiago.multimodal_toll_oracle_v1 import *  # noqa: F401,F403
from gato_tiago.multimodal_toll_oracle_v2 import (
    V1_REJECTED_REPORT_ONLY as V1_REJECTED_REPORT_ONLY,
)


ORACLE_EXECUTION_AUTHORIZATION = None
ORACLE_PROTOCOL_VERSION = "tiago_tool_center_toll_oracle_v2_1"
ORACLE_OUTPUT_PATH = Path(
    "/tmp/tiago-tool-center-toll-oracle-v2-authorized-once/oracle.json"
)
AUTHORIZED_ORIG_ARGV = (
    "python", "-B", "-m", "gato_tiago.multimodal_toll_oracle_v2_runner",
    "--execute", "--output", str(ORACLE_OUTPUT_PATH),
)

PREREQUISITE_ROOT = Path(
    "/tmp/tiago-tool-center-toll-oracle-v2-prerequisite-authorized-once"
)
PREREQUISITE_ARTIFACT_PINS = {
    "final_json": (
        PREREQUISITE_ROOT / "prerequisite.json",
        "b1805a61517805e3315c857d4192feef851e54cdd97338177c83e04c28a1b49b",
    ),
    "final_npz": (
        PREREQUISITE_ROOT / "prerequisite.npz",
        "d6705e1d741a192dd5c897ffa879b2c2c58d273b4c5fb88a1990c608015264ab",
    ),
    "final_manifest": (
        PREREQUISITE_ROOT / "prerequisite.manifest.json",
        "1b1ca64e5eef51c4c16d8798b8314b44023ba4498b42a24406e6c703dfd0ddbe",
    ),
    "latest_pointer": (
        PREREQUISITE_ROOT / "prerequisite.partial.latest.json",
        "3d9ba295d252ab7bcd0f48d9c585cfc99238ab9068b732377b6cacf4f6e68d36",
    ),
}
TASK_ARRAY_COUNT = 603
MODEL_ARRAY_COUNT = 150

def frozen_v2_metadata() -> dict:
    """Return V2 isolation metadata without opening any retained artifact."""

    return {
        "protocol": ORACLE_PROTOCOL_VERSION,
        "output": str(ORACLE_OUTPUT_PATH),
        "prerequisite_pins": {
            name: {"path": str(path), "sha256": digest}
            for name, (path, digest) in PREREQUISITE_ARTIFACT_PINS.items()
        },
        "prerequisite_artifact_loads": 1,
        "rejected_v1_artifact_loads": 0,
        "v1_rejected_report_only": dict(V1_REJECTED_REPORT_ONLY),
        "scientific_protocol_source": "multimodal_toll_oracle_v1",
        "scientific_changes_from_v1": (),
    }

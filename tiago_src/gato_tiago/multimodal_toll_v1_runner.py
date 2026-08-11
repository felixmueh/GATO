"""Hard-blocked Stage V1 runner namespace.

The V1 runner is deliberately not implemented or authorized in this static
checkpoint.  It owns a fresh artifact path and may never consume V0 arrays.
"""

from pathlib import Path

from gato_tiago.multimodal_toll_v1 import V1_OUTPUT_PATH, frozen_v1_metadata


RUNNER_EXECUTION_AUTHORIZATION = None
AUTHORIZED_OUTPUT_PATH = Path(V1_OUTPUT_PATH)
FROZEN_EXTENSION_HASHES = {
    "bsqp.bsqpN64_tiago_right_multimodal_toll": "3d3c1e0808a57b6189a0267e701a64d0b11ed13cda37b65873781954ffaa8551",
    "bsqp.bsqpN64_tiago_right": "f0944080523a212edce0af72d323f5eeffd587adde2f0bea6063a31fabff2f41",
    "bsqp.bsqpN64_tiago_right_multimodal": "75650c37b3cd0e729922cdd82e65dd81214bb29f9591d0a7466aac197908b1fe",
}


def execute_v1_runner(output, *, authorization=None):
    del output, authorization
    raise RuntimeError("Stage V1 runner execution is blocked pending audit")


def describe_v1_runner():
    return {
        "schema": frozen_v1_metadata(),
        "authorization": None,
        "v0_artifact_inputs": [],
        "v0_array_inputs": [],
        "fresh_artifact_namespace": str(AUTHORIZED_OUTPUT_PATH),
        "artifact_policy": {
            "immutable_generation_zero_before_model_or_rng": True,
            "resume_allowed": False,
            "rerun_allowed": False,
        },
        "extension_binaries_reused_without_build": dict(FROZEN_EXTENSION_HASHES),
    }

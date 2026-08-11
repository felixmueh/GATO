"""Hard-blocked isolated worker namespace for Stage V1."""


WORKER_EXECUTION_AUTHORIZATION = None
SUPPORTED_MODULES = (
    "bsqp.bsqpN64_tiago_right_multimodal_toll",
    "bsqp.bsqpN64_tiago_right",
    "bsqp.bsqpN64_tiago_right_multimodal",
)


def execute_worker(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("Stage V1 worker execution is blocked pending audit")

"""Shared experiment settings, also usable by the CLI and CPU-only plotting."""

REFERENCE_DT = .01
CONTROL_DT = .01
INTEGRATION_DT = .001
COST_ANCHOR_KNOTS = 32
DEFAULT_INIT_SOLVES = 5


def prediction_step(n, horizon_time=None):
    """Spacing of N state knots over a fixed duration or the historical grid."""
    return REFERENCE_DT if horizon_time is None else horizon_time / (n - 1)


def cost_rule(horizon_time, plant="indy7"):
    if plant == "tiago_right":
        return ("TIAGo tracking weights; running and limit weights scaled by prediction_dt/0.008; "
                "terminal position weight unchanged; velocity and state-limit terms include terminal knot" if horizon_time is not None
                else "TIAGo tracking weights unchanged at 10 ms prediction spacing")
    if horizon_time is None:
        return 'Historical weights including u_cost=1e-8*N'
    return ('Running weights scaled by prediction_dt/0.01; control weight anchored '
            'to N32 (3.2e-7); terminal position weight unchanged; velocity includes terminal knot')

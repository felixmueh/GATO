"""Reuse the existing estimator algorithm with this tick's frozen hypotheses."""
import numpy as np

from examples.force_estimator import ForceEstimator


class FrozenBatchForceEstimator(ForceEstimator):
    def generate_batch(self):
        cached = getattr(self, "_tick_candidates", None)
        return cached.copy() if cached is not None else super().generate_batch()

    def update_frozen(self, index, errors, candidates):
        if getattr(self, "_tick_candidates", None) is not None:
            raise RuntimeError("estimator updates cannot overlap")
        self._tick_candidates = np.array(candidates, copy=True)
        try:
            # Parent algorithm, including alpha/beta from the existing controller.
            return super().update(index, errors, alpha=0.6, beta=0.5)
        finally:
            self._tick_candidates = None

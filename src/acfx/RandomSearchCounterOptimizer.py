from typing import Dict, Tuple

import numpy as np
import pandas as pd
from .abstract import ModelBasedCounterOptimizer
from overrides import overrides


class RandomSearchCounterOptimizer(ModelBasedCounterOptimizer):
    def __init__(self, model, X: pd.DataFrame, feature_bounds: Dict[str, Tuple[float, float]], n_iter: int = 100):
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("Model must implement predict_proba()")
        self.model = model
        self.X = X
        self.feature_bounds = feature_bounds
        self.n_iter = n_iter

    @overrides
    def optimize_proba(self, target_class: int, feature_masked: list[str]) -> Dict[str, float]:
        base_instance = self.X.mean().copy()
        best_instance = base_instance.copy()
        best_score = self.model.predict_proba([base_instance])[0][target_class]

        for _ in range(self.n_iter):
            candidate = base_instance.copy()
            for feature_name in self.X.columns:
                if feature_name in feature_masked and feature_name in self.feature_bounds:
                    min_val, max_val = self.feature_bounds[feature_name]
                    candidate[feature_name] = np.random.uniform(min_val, max_val)

            score = self.model.predict_proba([candidate])[0][target_class]
            if score > best_score:
                best_score = score
                best_instance = candidate.copy()

        return best_instance.to_dict()
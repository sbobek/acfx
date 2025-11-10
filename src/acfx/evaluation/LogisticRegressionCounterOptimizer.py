from typing import List, Dict

import pandas as pd
import numpy as np
from overrides import overrides
from sklearn.linear_model._base import LinearClassifierMixin
from ..abstract import ModelBasedCounterOptimizer
import numbers

class LogisticRegressionCounterOptimizer(ModelBasedCounterOptimizer):
    def __init__(self, model:LinearClassifierMixin, X: pd.DataFrame, feature_bounds:dict):
        if not hasattr(model, 'coef_'):
            raise AttributeError('model.coef_ must be set')
        if feature_bounds is None:
            raise ValueError("feature_bounds must be set")

        if not isinstance(feature_bounds, dict) or not all(
                isinstance(k, str) and
                isinstance(v, tuple) and
                len(v) == 2 and
                all(isinstance(i, numbers.Number) for i in v)
                for k, v in feature_bounds.items()
        ):
            raise AttributeError("feature_bounds must be a dict with string keys and tuple of two floats as values")

        self.__feature_bounds = feature_bounds
        self.model = model
        self.X = X

    @overrides
    def optimize_proba(self, target_class: int, feature_masked: List[str]) -> Dict[str, float]:
        optimized_instances = []
        # target_class_name = self.model.feature_names_in_[target_class]

        for index, instance in self.X.iterrows():
            coefficients = self.model.coef_[target_class]  # Extract model coefficients for the target class
            direction = np.sign(coefficients)
            optimized_instance = instance.copy()

            for i, feature_name in enumerate(self.X.columns):
                if feature_name not in feature_masked:
                    continue
                if feature_name in self.__feature_bounds:
                    min_val, max_val = self.__feature_bounds[feature_name]
                    if direction[i] > 0:
                        optimized_instance[feature_name] = max_val  # Increase feature value if positive impact
                    else:
                        optimized_instance[feature_name] = min_val  # Decrease feature value if negative impact

            optimized_instances.append(optimized_instance)

        df_optimized = pd.DataFrame(optimized_instances)
        avg_optimized = df_optimized.mean().to_dict()
        return avg_optimized

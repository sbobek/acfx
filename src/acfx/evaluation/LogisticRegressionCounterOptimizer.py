import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from ..abstract import ModelBasedCounterOptimizer

class LogisticRegressionCounterOptimizer(ModelBasedCounterOptimizer):
    def __init__(self, model:ClassifierMixin, X: pd.DataFrame, feature_bounds:dict):
        if not hasattr(model, 'coef_'):
            raise AttributeError('optimizer requires model.coef_ to be set')
        if not isinstance(getattr(model, 'coef_', None), list):
            raise AttributeError('coef_ must be a list')
        if feature_bounds is None:
            raise ValueError("feature_bounds cannot be None")
        if not isinstance(feature_bounds, list):
            raise AttributeError('feature_bounds must be a list')

        super().__init__(model, X)
        self.__feature_bounds = feature_bounds
        self.model = model
        self.X = X

    def optimize_proba(self, target_class, feature_masked):
        """
        Modifies the instance to increase the probability of the target class by adjusting feature values.

        Parameters:
            model: A logistic regression model with accessible coefficients.
            target_class: The desired class to optimize towards.
            feature_masked: A boolean mask array (same shape as instance) indicating which features can be modified.

        Returns:
            Optimized instance as a numpy array.
        """

        for index, instance in self.X.iterrows():
            coefficients = self.model.coef_[target_class]  # Extract model coefficients for the target class

            # Identify the direction of optimization
            direction = np.sign(coefficients)

            optimized_instance = instance.copy()

            if len(direction) < 0:
                return optimized_instance

            for i, modifiable in enumerate(feature_masked):
                if modifiable:
                    if i in self.__feature_bounds:
                        min_val, max_val = self.__feature_bounds[i]
                        if direction[i] > 0:
                            optimized_instance[i] = max_val  # Increase feature value if positive impact
                        else:
                            optimized_instance[i] = min_val  # Decrease feature value if negative impact

            return optimized_instance
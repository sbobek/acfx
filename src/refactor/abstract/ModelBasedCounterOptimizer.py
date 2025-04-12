from abc import ABC, abstractmethod

import pandas as pd


class ModelBasedCounterOptimizer(ABC):
    @abstractmethod
    def optimize_proba(self, target_class, feature_masked):
        """
        The abstract method calculates probabilities taking into account the optimization of given parameters towards the target class.
        """
        pass

    def __init__(self, model, X: pd.DataFrame):
        self.model = model
        self.X = X
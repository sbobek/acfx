from abc import ABC, abstractmethod

class ModelBasedCounterOptimizer(ABC):
    @abstractmethod
    def optimize_proba(self, target_class, feature_masked):
        """
        Returns modified instance to increase the probability of the target class by adjusting feature values.
        """
        pass

from abc import ABC, abstractmethod
from typing import List


class ModelBasedCounterOptimizer(ABC):
    @abstractmethod
    def optimize_proba(self, target_class: int, feature_masked: List[str]):
        """
        Modifies the instance to increase the probability of the target class by adjusting feature values.

        Parameters:
        -----------
        target_class:
            The desired class to optimize towards.
        feature_masked:
            List of interchangeable features

        Returns:
        -------
        Optimized instance as a numpy array.
        """
        pass

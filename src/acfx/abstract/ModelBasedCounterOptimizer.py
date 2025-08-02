from abc import ABC, abstractmethod
from typing import List, Dict


class ModelBasedCounterOptimizer(ABC):
    @abstractmethod
    def optimize_proba(self, target_class: int, feature_masked: List[str]) -> Dict[str, float]:
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
        Dictionary of feature names and their optimized values
        """
        pass

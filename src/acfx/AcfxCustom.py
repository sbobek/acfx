import numpy as np
import pandas as pd
from overrides import overrides
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.base import ClassifierMixin
from typing import Sequence, Tuple, Dict, Optional, List, Self
from .ACFX import ACFX
from .abstract import OptimizerType, ModelBasedCounterOptimizer


class AcfxCustom(ACFX):
    """
        AcfxCustom: A Counterfactual Explanation Model (using custom blackbox)
    """
    def __init__(self, blackbox: ClassifierMixin):
        """

        Parameters
        ----------
        blackbox:
            Custom blackbox explainer
        """
        super().__init__(blackbox)

    @overrides
    def counterfactual(self, query_instance: np.ndarray, desired_class: int, num_counterfactuals: int = 1, proximity_weight: float = 1,
                       sparsity_weight: float = 1, plausibility_weight: float = 0, diversity_weight: float = 1,
                       init_points: int = 10,
                       n_iter: int = 1000, sampling_from_model: bool = True) -> np.ndarray:

        if self.optimizer is None:
            raise ValueError("Optimizer must be initialized in fit() before calling counterfactual().")
        return super().counterfactual(query_instance, desired_class, num_counterfactuals, proximity_weight, sparsity_weight,
                                                plausibility_weight, diversity_weight, init_points,
                                      n_iter, sampling_from_model)

    def fit(self, X: pd.DataFrame, pbounds: Dict[str, Tuple[float, float]], optimizer : ModelBasedCounterOptimizer=None,
            causal_order: Optional[Sequence[int]]=None,
            adjacency_matrix: Optional[np.ndarray]=None, y=None, masked_features: Optional[List[str]] = None,
            categorical_indicator: Optional[List[bool]] = None, features_order: Optional[List[str]] = None,
            bayesian_causality:bool = False,  bayesian_model : Optional[DiscreteBayesianNetwork]=None, num_bins:Optional[int] = None) -> Self:
        """
        Fits explainer to the sampled data and blackbox model provided in the constructor

        :return:
        self
            Fitted estimator.

        Parameters
        ----------
        X : {sparse matrix} of shape (n_samples, n_features)
            Used for counterfactuals generation

        pbounds:
            The bounds for each feature to search over (dict with feature names as keys and tuple (min, max) as values).

        optimizer:
            Custom optimizer compliant with blackbox predictor

        causal_order:
            The order of variables in the causal graph.

        adjacency_matrix:
            The adjacency matrix representing the causal structure.

        y : array-like of shape (n_samples,)
            Target values used for blackbox model fitting only. You can provide fitted blackbox to constructor or fit it in this method by providing this parameter

        masked_features:
            List of interchangeable features

        categorical_indicator:
            True at the index where the variable should be treated as categorical

        features_order:
            order of features in query instance

        bayesian_causality:
            skip adjacency and calculate discrete bayesian network for causal loss.
            A discrete bayesian network will be fitted to calculate causal loss component.

        bayesian_model:
            optionally, provide pre-fitted discrete bayesian model, if bayesian_causality is True.
            If it is not provided and=bayesian_causality, it will be fitted with the num_bins param

        num_bins:
            Number of bins to use for discretizing continuous features

        """
        self.optimizer_type = OptimizerType.Custom
        if optimizer is None:
            raise ValueError("Optimizer must be given for AcfxCustom")
        self.optimizer = optimizer
        return super().fit(X, pbounds, causal_order, adjacency_matrix,
                    y, masked_features,categorical_indicator, features_order, bayesian_causality, bayesian_model, num_bins)
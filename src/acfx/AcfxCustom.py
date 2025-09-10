import numpy as np
import pandas as pd
from overrides import overrides
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

    def fit(self, X:pd.DataFrame, adjacency_matrix:Optional[np.ndarray], casual_order:Optional[Sequence[int]],
            pbounds:Dict[str, Tuple[float, float]],
            optimizer : ModelBasedCounterOptimizer=None, y=None, masked_features:Optional[List[str]]=None,
            categorical_indicator:Optional[List[bool]]=None, features_order:Optional[List[str]] =None) -> Self:
        """
        Fits explainer to the sampled data and blackbox model provided in the constructor

        :return:
        self
            Fitted estimator.

        Parameters
        ----------
        X : {sparse matrix} of shape (n_samples, n_features)
            Used for counterfactuals generation

        adjacency_matrix:
            The adjacency matrix representing the causal structure.

        casual_order:
            The order of variables in the causal graph.

        pbounds:
            The bounds for each feature to search over (dict with feature names as keys and tuple (min, max) as values).

        optimizer:
            Custom optimizer compliant with blackbox predictor

        y : array-like of shape (n_samples,)
            Target values used for blackbox model fitting only. You can provide fitted blackbox to constructor or fit it in this method by providing this parameter

        masked_features:
            List of interchangeable features

        categorical_indicator:
            True at the index where the variable should be treated as categorical

        features_order:
            order of features in query instance
        """
        self.optimizer_type = OptimizerType.Custom
        if optimizer is None:
            raise ValueError("Optimizer must be given for AcfxCustom")
        self.optimizer = optimizer
        return super().fit(X, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)
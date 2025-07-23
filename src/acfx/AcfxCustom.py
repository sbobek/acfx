from sklearn.base import ClassifierMixin

from .ACFX import ACFX
from .abstract import OptimizerType


class AcfxCustom(ACFX):
    """
        AcfxCustom: A Counterfactual Explanation Model (using custom blackbox)
    """
    def __init__(self, blackbox: ClassifierMixin):
        super().__init__(blackbox)

    def counterfactual(self, desired_class, num_counterfactuals=1, proximity_weight=1,
                       sparsity_weight=1, plausibility_weight=0, diversity_weight=1, init_points=10,
                       n_iter=1000, sampling_from_model=True, optimizer=None):
        if self.optimizer is None:
            raise ValueError("Optimizer must be initialized in fit() before calling counterfactual().")
        return super().counterfactual(desired_class, num_counterfactuals, proximity_weight, sparsity_weight,
                                                plausibility_weight, diversity_weight, init_points,
                                      n_iter, sampling_from_model)

    def fit(self, X, query_instance,adjacency_matrix, casual_order, pbounds, optimizer=None, y=None, masked_features=None,
            categorical_indicator=None, features_order=None):
        self.optimizer_type = OptimizerType.Custom
        if optimizer is None:
            raise ValueError("Optimizer must be given for AcfxCustom")
        self.optimizer = optimizer
        return super().fit(X, query_instance, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)
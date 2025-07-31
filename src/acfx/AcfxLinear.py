from sklearn.linear_model._base import LinearClassifierMixin

from .ACFX import ACFX
from .abstract import OptimizerType


class AcfxLinear(ACFX):
    """
        AcfxCustom: A Counterfactual Explanation Model (using linear additive model as blackbox)
    """
    def __init__(self, blackbox: LinearClassifierMixin):
        super().__init__(blackbox)

    def fit(self, X, query_instance, adjacency_matrix, casual_order, pbounds, y=None, masked_features=None,
            categorical_indicator=None, features_order=None):
        self.optimizer_type = OptimizerType.LinearAdditive
        return super().fit(X, query_instance, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)

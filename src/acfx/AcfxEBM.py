from interpret.glassbox import ExplainableBoostingClassifier

from .ACFX import ACFX
from .abstract import OptimizerType

class AcfxEBM(ACFX):
    """
        AcfxCustom: A Counterfactual Explanation Model (using EBM as blackbox)
    """
    def __init__(self, blackbox: ExplainableBoostingClassifier):
        super().__init__(blackbox)

    def fit(self, X, query_instance,adjacency_matrix, casual_order, pbounds, y=None, masked_features=None,
            categorical_indicator=None, features_order=None):
        self.optimizer_type = OptimizerType.EBM
        return super().fit(X, query_instance, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)


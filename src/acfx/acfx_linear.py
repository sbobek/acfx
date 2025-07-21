from sklearn.base import ClassifierMixin

from src.acfx.acfx import ACFX
from src.refactor.abstract.OptimizerType import OptimizerType


class AcfxLinear(ACFX):
    def __init__(self, blackbox: ClassifierMixin):
        super().__init__(blackbox)

    def fit(self, X, query_instance, adjacency_matrix, casual_order, pbounds, y=None, masked_features=None,
            categorical_indicator=None, features_order=None):
        self.optimizer_type = OptimizerType.LogisticRegression
        return super().fit(X, query_instance, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)

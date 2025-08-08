from typing import Sequence, Tuple, Dict, Optional, List

import numpy as np
from overrides import overrides
from sklearn.linear_model._base import LinearClassifierMixin

from .ACFX import ACFX
from .abstract import OptimizerType


class AcfxLinear(ACFX):
    """
        AcfxCustom: A Counterfactual Explanation Model (using linear additive model as blackbox)
    """
    def __init__(self, blackbox: LinearClassifierMixin):
        """

        Parameters
        ----------
        blackbox:
            Linear blackbox explainer
        """
        super().__init__(blackbox)

    @overrides
    def fit(self, X, query_instance: np.ndarray, adjacency_matrix:Optional[np.ndarray], casual_order:Optional[Sequence[int]],
            pbounds:Dict[str, Tuple[float, float]],y=None, masked_features:Optional[List[str]] = None,
            categorical_indicator:Optional[List[bool]] =None, features_order:Optional[List[str]] =None):
        self.optimizer_type = OptimizerType.LinearAdditive
        return super().fit(X, query_instance, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)

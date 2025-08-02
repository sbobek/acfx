from typing import Sequence, Tuple, Dict, Optional, List

import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from overrides import overrides

from .ACFX import ACFX
from .abstract import OptimizerType

class AcfxEBM(ACFX):
    """
        AcfxCustom: A Counterfactual Explanation Model (using EBM as blackbox)
    """
    def __init__(self, blackbox: ExplainableBoostingClassifier):
        """

        Parameters
        ----------
        blackbox:
            EBM blackbox explainer
        """
        super().__init__(blackbox)

    @overrides
    def fit(self, X, query_instance: np.ndarray, adjacency_matrix:np.ndarray, casual_order:Sequence[int],
            pbounds:Dict[str, Tuple[float, float]],y=None, masked_features:Optional[List[str]] = None,
            categorical_indicator:Optional[List[bool]] =None, features_order:Optional[List[str]] =None):
        self.optimizer_type = OptimizerType.EBM
        return super().fit(X, query_instance, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)


from typing import Sequence, Tuple, Dict, Optional, List, Self

import numpy as np
import pandas as pd
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
    def fit(self, X:pd.DataFrame, adjacency_matrix:Optional[np.ndarray], casual_order:Optional[Sequence[int]],
            pbounds:Dict[str, Tuple[float, float]],y=None, masked_features:Optional[List[str]] = None,
            categorical_indicator:Optional[List[bool]] =None, features_order:Optional[List[str]] =None) -> Self:
        self.optimizer_type = OptimizerType.EBM
        return super().fit(X, adjacency_matrix, casual_order, pbounds,
                    y, masked_features,categorical_indicator, features_order)


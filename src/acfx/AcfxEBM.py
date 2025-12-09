from typing import Sequence, Tuple, Dict, Optional, List, Self

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from overrides import overrides
from pgmpy.models import DiscreteBayesianNetwork

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
    def fit(self, X:pd.DataFrame, pbounds:Dict[str, Tuple[float, float]], causal_order:Optional[Sequence[int]]=None,
            adjacency_matrix:Optional[np.ndarray]=None
            ,y=None, masked_features:Optional[List[str]] = None,
            categorical_indicator:Optional[List[bool]] =None, features_order:Optional[List[str]] =None,
            bayesian_causality:bool = False,  bayesian_model : Optional[DiscreteBayesianNetwork]=None,
            num_bins:Optional[int] = None) -> Self:
        self.optimizer_type = OptimizerType.EBM
        return super().fit(X, pbounds, causal_order,adjacency_matrix,
                    y, masked_features,categorical_indicator, features_order, bayesian_causality, bayesian_model, num_bins)


from typing import Sequence, Tuple, Dict, Optional, List, Self

import numpy as np
import pandas as pd
from overrides import overrides
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.linear_model._base import LinearClassifierMixin

from .ACFX import ACFX
from .abstract import OptimizerType


class AcfxLinear(ACFX):
    """
    Counterfactual explanation model using a linear additive classifier.

    This class configures the ACFX counterfactual explanation model to use
    the optimizer intended for linear additive classifiers. The black-box
    model is provided as a ``LinearClassifierMixin``.

    Parameters
    ----------
    blackbox : LinearClassifierMixin
        Linear classifier for which counterfactual explanations are
        generated.

    Notes
    -----
    The linear additive optimizer type is selected when :meth:`fit` is
    called. Model fitting and storage of the counterfactual-generation
    configuration are delegated to the base :class:`ACFX` implementation.

    The counterfactual-generation method inherited from :class:`ACFX`
    requires the fitted linear classifier to expose a ``coef_`` attribute
    when the linear additive optimizer is used.
    """
    def __init__(self, blackbox: LinearClassifierMixin):
        """
        Initialize the linear-model-based counterfactual explainer.

        Parameters
        ----------
        blackbox : LinearClassifierMixin
            Linear classifier for which counterfactual explanations are
            generated. The classifier is passed to the base ``ACFX``
            constructor.
        """
        super().__init__(blackbox)

    @overrides
    def fit(self, X:pd.DataFrame, pbounds:Dict[str, Tuple[float, float]], causal_order:Optional[Sequence[int]]=None,
            adjacency_matrix:Optional[np.ndarray]=None,
            y=None, masked_features:Optional[List[str]] = None,
            categorical_indicator:Optional[List[bool]] =None, features_order:Optional[List[str]] =None,
            bayesian_causality:bool = False,  bayesian_model : Optional[DiscreteBayesianNetwork]=None,
            num_bins:Optional[int] = None) -> Self:
        """
               Fit the explainer configuration to the provided data.

               This method selects the linear additive optimizer type and delegates
               the remaining fitting procedure to the base :class:`ACFX`
               implementation. If ``y`` is provided, the base implementation fits
               the black-box model using ``X`` and ``y``.

               Parameters
               ----------
               X : pandas.DataFrame of shape (n_samples, n_features)
                   Dataset used for counterfactual generation. If ``y`` is provided,
                   ``X`` is also used to fit the black-box model.

               pbounds : dict[str, tuple[float, float]]
                   Bounds used when searching for counterfactual values. Each key is
                   a feature name and each value is a ``(min, max)`` tuple specifying
                   the search bounds for that feature.

               causal_order : Sequence[int] or None
                   Order of variables in the causal graph. This parameter is used
                   with ``adjacency_matrix`` when Bayesian causality is disabled.

               adjacency_matrix : numpy.ndarray or None
                   Adjacency matrix representing the causal structure. This
                   parameter is used with ``causal_order`` when Bayesian causality
                   is disabled.

               y : array-like of shape (n_samples,) or None
                   Target values used for fitting the black-box model. If ``y`` is
                   not provided, the black-box model is not fitted by this method
                   and is expected to have been fitted beforehand.

               masked_features : list[str] or None
                   List of interchangeable features used during counterfactual
                   generation.

               categorical_indicator : list[bool] or None
                   Boolean indicators specifying which features should be treated
                   as categorical. Each position corresponds to a feature.

               features_order : list[str] or None
                   Order of features in a query instance.

               bayesian_causality : bool
                   Whether to use a discrete Bayesian network for the causal loss.
                   If True, ``adjacency_matrix`` and ``causal_order`` are not used.
                   Instead, a discrete Bayesian network is used to calculate the
                   causal loss component.

               bayesian_model : DiscreteBayesianNetwork or None
                   Pre-fitted discrete Bayesian network used when
                   ``bayesian_causality`` is True. The model must be trained on the
                   same ``X`` and with the same ``num_bins``, because counterfactual
                   candidates are discretized using the bins derived from ``X``.
                   This compatibility is not validated.

                   If no model is provided while ``bayesian_causality`` is True,
                   the base implementation initializes and fits a Bayesian model
                   using ``X``, ``categorical_indicator``, and ``num_bins``.

               num_bins : int or None
                   Number of bins used to discretize continuous features when
                   Bayesian causality is enabled.

               Returns
               -------
               Self
                   Fitted explainer instance configured to use the linear additive
                   optimizer.

               Notes
               -----
               Calling this method sets ``optimizer_type`` to
               ``OptimizerType.LinearAdditive`` before delegating to ``ACFX.fit``.

               The black-box model must expose a ``coef_`` attribute before
               counterfactual generation. If ``y`` is provided, the base
               implementation fits the model before counterfactuals are generated.
               If ``y`` is not provided, the supplied model is expected to have
               already been fitted.
               """
        self.optimizer_type = OptimizerType.LinearAdditive
        return super().fit(X, pbounds, causal_order, adjacency_matrix,
                    y, masked_features,categorical_indicator, features_order, bayesian_causality, bayesian_model, num_bins)

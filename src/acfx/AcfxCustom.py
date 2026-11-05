import numpy as np
import pandas as pd
from overrides import overrides
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.base import ClassifierMixin
from typing import Sequence, Tuple, Dict, Optional, List, Self
from .ACFX import ACFX
from .abstract import OptimizerType, ModelBasedCounterOptimizer


class AcfxCustom(ACFX):
    """
    Counterfactual explanation model using a custom optimizer.

    ``AcfxCustom`` extends :class:`ACFX` with support for a user-provided
    optimizer implementing the :class:`ModelBasedCounterOptimizer` interface.
    The optimizer is supplied through :meth:`fit` and is subsequently passed
    to the counterfactual-generation procedure implemented by the base class.

    The supplied black-box classifier is used to generate predictions and
    evaluate counterfactual candidates. If target values are passed to
    :meth:`fit`, the base ``ACFX`` implementation fits the classifier using
    the supplied feature data and target values. Otherwise, the classifier is
    expected to have been fitted before counterfactual generation.

    Parameters
    ----------
    blackbox : ClassifierMixin
        Black-box classifier used to generate predictions and evaluate
        counterfactual candidates.

    Attributes
    ----------
    blackbox : ClassifierMixin
        Black-box classifier supplied to the constructor.

    optimizer_type : OptimizerType or None
        Type of optimizer used for counterfactual generation. It is
        initialized to ``None`` by :class:`ACFX` and set to
        ``OptimizerType.Custom`` when :meth:`fit` is called.

    optimizer : ModelBasedCounterOptimizer or None
        Custom optimizer supplied to :meth:`fit`. It is initialized to
        ``None`` by :class:`ACFX` and must be configured before
        :meth:`counterfactual` is called.

    X : pandas.DataFrame or None
        Reference dataset used for counterfactual generation. It is set by
        :meth:`fit`.

    pbounds : dict[str, tuple[float, float]] or None
        Feature bounds used during counterfactual generation. It is set by
        :meth:`fit`.

    categorical_indicator : list[bool] or None
        Boolean indicators specifying which features should be treated as
        categorical. It is set by :meth:`fit`.

    features_order : list[str] or None
        Order of features in query instances. It is set by :meth:`fit`.

    masked_features : list[str] or None
        List of interchangeable features used during counterfactual
        generation. It is set by :meth:`fit`.

    bayesian_causality : bool or None
        Indicates whether a discrete Bayesian network is used for the causal
        loss. It is initialized to ``None`` and set by :meth:`fit`.

    bayesian_model : DiscreteBayesianNetwork or None
        Discrete Bayesian network used when Bayesian causality is enabled.
        It may be supplied to :meth:`fit` or trained internally by the base
        implementation.

    num_bins : int or None
        Number of bins used for discretizing continuous features when
        Bayesian causality is enabled.

    adjacency_matrix : numpy.ndarray or None
        Adjacency matrix representing the causal structure when Bayesian
        causality is disabled.

    causal_order : Sequence[int] or None
        Order of variables in the causal graph when Bayesian causality is
        disabled.

    Notes
    -----
    The constructor only passes the black-box classifier to the base
    :class:`ACFX` constructor. It does not configure the custom optimizer.

    The optimizer must be supplied when :meth:`fit` is called. Although the
    current method signature assigns ``None`` as the default value of
    ``optimizer``, the method raises a ``ValueError`` if no optimizer is
    provided.

    Calling :meth:`fit` sets ``optimizer_type`` to
    ``OptimizerType.Custom`` and stores the supplied optimizer in
    ``self.optimizer`` before delegating the remaining configuration to
    :meth:`ACFX.fit`.

    Calling :meth:`counterfactual` before a custom optimizer has been
    configured raises a ``ValueError``. After checking the optimizer, the
    method delegates counterfactual generation to
    :meth:`ACFX.counterfactual`.

    Depending on the causality configuration established by :meth:`fit`, the
    base implementation generates counterfactuals using either the standard
    causal representation or a discrete Bayesian network.
    """
    def __init__(self, blackbox: ClassifierMixin):
        """

        Parameters
        ----------
        blackbox: ClassifierMixin
            Custom blackbox explainer
        """
        super().__init__(blackbox)

    @overrides
    def counterfactual(self, query_instance: np.ndarray, desired_class: int, num_counterfactuals: int = 1, proximity_weight: float = 1,
                       sparsity_weight: float = 1, plausibility_weight: float = 0, diversity_weight: float = 1,
                       init_points: int = 10,
                       n_iter: int = 1000, sampling_from_model: bool = True) -> np.ndarray:
        """
            Generates counterfactuals using the custom optimizer.

            Parameters
            ----------
            query_instance : numpy.ndarray
                The instance to generate counterfactuals for.

            desired_class : int
                The target class for the counterfactuals.

            num_counterfactuals : int
                The number of counterfactual instances to generate.

            proximity_weight : float
                Weight for proximity loss component.

            sparsity_weight : float
                Weight for sparsity loss component.

            plausibility_weight : float
                Weight for plausibility loss component.

            diversity_weight : float
                Weight for diversity loss component.

            init_points : int
                Number of initial points for Bayesian Optimization.

            n_iter : int
                Number of iterations for Bayesian Optimization.

            sampling_from_model : bool
                True if you want to generate samples from model after sampling
                from data.

            Returns
            -------
            numpy.ndarray
                The generated counterfactuals that minimize the loss function.
            """

        if self.optimizer is None:
            raise ValueError("Optimizer must be initialized in fit() before calling counterfactual().")
        return super().counterfactual(query_instance, desired_class, num_counterfactuals, proximity_weight, sparsity_weight,
                                                plausibility_weight, diversity_weight, init_points,
                                      n_iter, sampling_from_model)

    def fit(self, X: pd.DataFrame, pbounds: Dict[str, Tuple[float, float]], optimizer : ModelBasedCounterOptimizer=None,
            causal_order: Optional[Sequence[int]]=None,
            adjacency_matrix: Optional[np.ndarray]=None, y=None, masked_features: Optional[List[str]] = None,
            categorical_indicator: Optional[List[bool]] = None, features_order: Optional[List[str]] = None,
            bayesian_causality:bool = False,  bayesian_model : Optional[DiscreteBayesianNetwork]=None, num_bins:Optional[int] = None) -> Self:
        """
            Fits explainer to the sampled data and blackbox model provided in the constructor.

            Parameters
            ----------
            X : pandas.DataFrame of shape (n_samples, n_features)
                Dataset used for counterfactuals generation.

            pbounds : dict[str, tuple[float, float]]
                The bounds for each feature to search over (dict with feature names
                as keys and tuple (min, max) as values).

            optimizer : ModelBasedCounterOptimizer or None
                Custom optimizer compliant with blackbox predictor.

            causal_order : Sequence[int] or None
                The order of variables in the causal graph.

            adjacency_matrix : numpy.ndarray or None
                The adjacency matrix representing the causal structure.

            y : array-like of shape (n_samples,) or None
                Dataset target values used for blackbox model fitting only. You can
                provide fitted blackbox to constructor or fit it in this method by
                providing this parameter.

            masked_features : list[str] or None
                List of interchangeable features.

            categorical_indicator : list[bool] or None
                True at the index where the variable should be treated as categorical.

            features_order : list[str] or None
                Order of features in query instance.

            bayesian_causality : bool
                Calculate discrete Bayesian network for causal loss. If
                bayesian_causality is True, then adjacency_matrix and causal_order
                are not used. Instead, a discrete Bayesian network will be used to
                calculate causal loss component.

            bayesian_model : DiscreteBayesianNetwork or None
                Optionally, provide pre-fitted discrete Bayesian model, if
                bayesian_causality is True. It must be trained on the same X with
                the same num_bins, because counterfactual candidates are discretized
                with the bins of X (this is not validated). If it is not provided
                and bayesian_causality is True, then it will be initialized internally
                and fitted with the num_bins param.

            num_bins : int or None
                Number of bins to use for discretizing continuous features.

            Returns
            -------
            Self
                Fitted estimator.

        """
        self.optimizer_type = OptimizerType.Custom
        if optimizer is None:
            raise ValueError("Optimizer must be given for AcfxCustom")
        self.optimizer = optimizer
        return super().fit(X, pbounds, causal_order, adjacency_matrix,
                    y, masked_features,categorical_indicator, features_order, bayesian_causality, bayesian_model, num_bins)
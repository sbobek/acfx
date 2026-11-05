from typing import Sequence, Tuple, Dict, Optional, List, Self

import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from abc import ABC, abstractmethod

from .evaluation import generate_cfs_bayesian
from .abstract import OptimizerType
from .evaluation import generate_cfs
from .evaluation import train_bayesian_model

class ACFX(ABC, BaseEstimator, TransformerMixin):
    """
    Abstract base class for ACFX counterfactual explanation models.

    ``ACFX`` defines the shared interface and implementation used to configure
    and generate counterfactual explanations for a supplied black-box
    classifier. Concrete subclasses configure the optimizer type and may
    provide additional optimizer-specific behavior.

    The class stores the reference data, feature bounds, feature metadata,
    causal configuration, and optimizer configuration required by
    :meth:`counterfactual`. Counterfactual generation is delegated to either
    ``generate_cfs`` or ``generate_cfs_bayesian``, depending on whether a
    Bayesian model is available.

    Parameters
    ----------
    blackbox : ClassifierMixin
        Black-box classifier used to generate predictions and evaluate
        counterfactual candidates. If target values are later supplied to
        :meth:`fit`, the classifier is fitted using the provided ``X`` and
        ``y``. Otherwise, the supplied classifier is expected to have been
        fitted beforehand.

    Attributes
    ----------
    blackbox : ClassifierMixin
        Black-box classifier supplied to the constructor.

    X : pandas.DataFrame or None
        Reference dataset used for counterfactual generation. It is set by
        :meth:`fit`.

    pbounds : dict[str, tuple[float, float]] or None
        Feature bounds used during counterfactual generation. It is set by
        :meth:`fit`.

    optimizer_type : OptimizerType or None
        Type of optimizer used for counterfactual generation. It is initially
        ``None`` and must be set by a concrete subclass before
        :meth:`counterfactual` is called.

    optimizer : object or None
        Custom optimizer used when ``optimizer_type`` is
        ``OptimizerType.Custom``. It is initially ``None``.

    categorical_indicator : list[bool] or None
        Boolean indicators specifying which features are categorical. It is
        set by :meth:`fit`.

    features_order : list[str] or None
        Order of features in query instances. It is set by :meth:`fit`.

    masked_features : list[str] or None
        List of interchangeable features used during counterfactual
        generation. It is set by :meth:`fit`.

    bayesian_causality : bool or None
        Indicates whether a discrete Bayesian network is used for the causal
        loss. It is initialized to ``None`` and set by :meth:`fit`.

    bayesian_model : DiscreteBayesianNetwork or None
        Discrete Bayesian network used for Bayesian causal-loss calculation.
        It may be supplied to :meth:`fit` or trained internally when Bayesian
        causality is enabled.

    num_bins : int or None
        Number of bins used to discretize continuous features for Bayesian
        causal-loss calculation. It is initialized to ``None`` and set by
        :meth:`fit` when Bayesian causality is enabled.

    adjacency_matrix : numpy.ndarray or None
        Adjacency matrix representing the causal structure used when Bayesian
        causality is disabled. It is initialized to ``None`` and may be set
        by :meth:`fit`.

    causal_order : Sequence[int] or None
        Order of variables in the causal graph used when Bayesian causality
        is disabled. It is initialized to ``None`` and may be set by
        :meth:`fit`.

    Notes
    -----
    This is an abstract class because :meth:`fit` is declared as an abstract
    method. Concrete subclasses are responsible for configuring the optimizer
    type before delegating the remaining fitting procedure to this class.

    Calling the constructor only stores the black-box classifier and
    initializes the remaining configuration attributes. It does not fit the
    classifier, select an optimizer, train a Bayesian network, or prepare the
    explainer for counterfactual generation.

    A concrete subclass must call :meth:`fit` before
    :meth:`counterfactual`. The counterfactual method raises a ``ValueError``
    if ``optimizer_type`` has not been set.

    When Bayesian causality is disabled, the class stores
    ``adjacency_matrix`` and ``causal_order``. When Bayesian causality is
    enabled, these two parameters are not stored by :meth:`fit`; instead,
    ``num_bins`` and ``bayesian_model`` are configured.

    The class inherits estimator behavior from ``BaseEstimator`` and
    ``TransformerMixin``. The methods explicitly implemented here are
    ``fit``, ``predict``, and ``counterfactual``.
    """
    def __init__(self, blackbox:ClassifierMixin):
        """

        Parameters
        ----------
        blackbox: ClassifierMixin
            Black-box classifier used to generate predictions.
        """
        self.bayesian_causality = None
        self.num_bins = None
        self.blackbox = blackbox
        self.optimizer = None
        self.optimizer_type = None
        self.X = None
        self.categorical_indicator = None
        self.features_order = None
        self.pbounds = None
        self.adjacency_matrix = None
        self.causal_order = None
        self.masked_features = None
        self.bayesian_model = None


    @abstractmethod
    def fit(self, X: pd.DataFrame, pbounds: Dict[str, Tuple[float, float]], causal_order: Optional[Sequence[int]]=None,
            adjacency_matrix: Optional[np.ndarray]=None, y=None, masked_features: Optional[List[str]] = None,
            categorical_indicator: Optional[List[bool]] = None, features_order: Optional[List[str]] = None,
            bayesian_causality:bool = False, bayesian_model : Optional[DiscreteBayesianNetwork]=None, num_bins:Optional[int] = None) -> Self:
        """
        Fits explainer to the sampled data and blackbox model provided in the constructor.

        Returns
        -------
        Self
            Fitted estimator.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            Dataset used for counterfactuals generation.

        pbounds : dict[str, tuple[float, float]]
            The bounds for each feature to search over (dict with feature names
            as keys and tuple (min, max) as values).

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
            and bayesian_causality is True, then it will be initialized
            internally and fitted with the num_bins param.

        num_bins : int or None
            Number of bins to use for discretizing continuous features
            (Bayesian causality only).

        """
        if y is not None:
            self.blackbox.fit(X, y)
        self.X = X
        self.categorical_indicator = categorical_indicator
        self.features_order = features_order
        self.bayesian_causality = bayesian_causality

        if not self.bayesian_causality:
            self.adjacency_matrix = adjacency_matrix
            self.causal_order = causal_order
        else:
            self.num_bins = num_bins
            if bayesian_model is None:
                self.bayesian_model = train_bayesian_model(self.X, self.categorical_indicator, self.num_bins)
            else:
                self.bayesian_model = bayesian_model
        self.pbounds = pbounds
        self.masked_features = masked_features
        return self

    def predict(self, X):
        """
        Predicts using blackbox model.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Dataset to predict.

        Returns
        -------
        array-like of shape (n_samples,)
            Prediction class labels for samples in X by blackbox model.
        """
        return self.blackbox.predict(X)

    def counterfactual(self, query_instance: np.ndarray, desired_class:int, num_counterfactuals: int =1, proximity_weight : float =1,
                       sparsity_weight : float =1, plausibility_weight : float =0, diversity_weight : float =1, init_points : int =10,
                       n_iter : int =1000, sampling_from_model : bool=True) -> np.ndarray:
        """
        Generates counterfactuals.

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
        if plausibility_weight > 0:
            if not self.bayesian_causality:
                if self.causal_order is None:
                    raise ValueError("Causal order must be provided if plausibility loss is on")
                if self.adjacency_matrix is None:
                    raise ValueError("adjacency_matrix must be provided")
                if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
                    raise ValueError("adjacency matrix must have same number of rows and columns")
                if self.adjacency_matrix.shape[0] != len(self.causal_order):
                    raise ValueError("adjacency matrix must be of same length as causal order")
            else:
                if self.bayesian_model is None:
                    raise ValueError("Bayesian model must be provided")
                if self.num_bins is None:
                    raise ValueError("num_bins must be provided")

        if query_instance is None:
            raise ValueError("query_instance must not be None")
        if self.optimizer_type is None:
            raise ValueError("optimizer_type must be set via fit() before calling counterfactual()")
        if self.optimizer is None and self.optimizer_type is OptimizerType.Custom:
            raise ValueError("optimizer must be set before calling counterfactual()")
        if self.optimizer_type is OptimizerType.LinearAdditive:
            if not hasattr(self.blackbox, 'coef_'):
                raise AttributeError('optimizer requires model.coef_ as linear coefficients to be set')
        if self.bayesian_model is None:
            return generate_cfs(query_instance=query_instance,
                                desired_class=desired_class,
                                adjacency_matrix=self.adjacency_matrix,
                                causal_order=self.causal_order,
                                proximity_weight=proximity_weight,
                                sparsity_weight=sparsity_weight,
                                plausibility_weight=plausibility_weight,
                                diversity_weight=diversity_weight,
                                bounds=self.pbounds,
                                model=self.blackbox,
                                features_order=self.features_order,
                                masked_features= self.masked_features,
                                categorical_indicator= self.categorical_indicator,
                                X=self.X,
                                num_cfs=num_counterfactuals,
                                init_points=init_points,
                                n_iter=n_iter,
                                sampling_from_model=sampling_from_model,
                                optimizer_type=self.optimizer_type,
                                optimizer=self.optimizer)
        else:
            return generate_cfs_bayesian(query_instance=query_instance,
                                desired_class=desired_class,
                                bayesian_model=self.bayesian_model,
                                categorical_bins_num=self.num_bins,
                                proximity_weight=proximity_weight,
                                sparsity_weight=sparsity_weight,
                                plausibility_weight=plausibility_weight,
                                diversity_weight=diversity_weight,
                                bounds=self.pbounds,
                                model=self.blackbox,
                                features_order=self.features_order,
                                masked_features= self.masked_features,
                                categorical_indicator= self.categorical_indicator,
                                X=self.X,
                                num_cfs=num_counterfactuals,
                                init_points=init_points,
                                n_iter=n_iter,
                                sampling_from_model=sampling_from_model,
                                optimizer_type=self.optimizer_type,
                                optimizer=self.optimizer)




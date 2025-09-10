from typing import Sequence, Tuple, Dict, Optional, List, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from abc import ABC, abstractmethod
from .abstract import OptimizerType
from .evaluation import generate_cfs


class ACFX(ABC, BaseEstimator, TransformerMixin):
    """
        ACFX: A Counterfactual Explanation Model
    """
    def __init__(self, blackbox:ClassifierMixin):
        """

        Parameters
        ----------
        blackbox:
            Blackbox explainer
        """
        self.blackbox = blackbox
        self.optimizer = None
        self.optimizer_type = None
        self.X = None
        self.categorical_indicator = None
        self.features_order = None
        self.pbounds = None
        self.adjacency_matrix = None
        self.casual_order = None
        self.masked_features = None


    @abstractmethod
    def fit(self, X:pd.DataFrame, adjacency_matrix:Optional[np.ndarray], casual_order:Optional[Sequence[int]],
            pbounds:Dict[str, Tuple[float, float]],y=None, masked_features:Optional[List[str]] = None,
            categorical_indicator:Optional[List[bool]] =None, features_order:Optional[List[str]] =None) -> Self:
        """
        Fits explainer to the sampled data and blackbox model provided in the constructor

        :return:
        self
            Fitted estimator.

        Parameters
        ----------
        X : {sparse matrix} of shape (n_samples, n_features)
            Used for counterfactuals generation

        adjacency_matrix:
            The adjacency matrix representing the causal structure.

        casual_order:
            The order of variables in the causal graph.

        pbounds:
            The bounds for each feature to search over (dict with feature names as keys and tuple (min, max) as values).

        y : array-like of shape (n_samples,).
            Target values used for blackbox model fitting only. You can provide fitted blackbox to constructor or fit it in this method by providing this parameter

        masked_features:
            List of interchangeable features

        categorical_indicator:
            True at the index where the variable should be treated as categorical

        features_order:
            order of features in query instance
        """
        if y is not None:
            self.blackbox.fit(X, y)
        self.X = X
        self.categorical_indicator = categorical_indicator
        self.features_order = features_order
        self.adjacency_matrix = adjacency_matrix
        self.casual_order = casual_order
        self.pbounds = pbounds
        self.masked_features = masked_features
        return self

    def predict(self, X):
        """
        Predicts using blackbox model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Used for counterfactuals generation

        Returns
        -------
        Prediction class labels for samples in X by blackbox model
        """
        return self.blackbox.predict(X)

    def counterfactual(self, query_instance: np.ndarray, desired_class:int, num_counterfactuals: int =1, proximity_weight : float =1,
                       sparsity_weight : float =1, plausibility_weight : float =0, diversity_weight : float =1, init_points : int =10,
                       n_iter : int =1000, sampling_from_model : bool=True) -> np.ndarray:
        """
        Generates counterfactuals

        Parameters
        ----------
        query_instance:
            The instance to generate counterfactuals for.
        desired_class:
            The target class for the counterfactuals.
        num_counterfactuals:
            The number of counterfactual instances to generate.
        proximity_weight:
            Weight for proximity loss component
        sparsity_weight:
            Weight for sparsity loss component
        plausibility_weight:
            Weight for plausibility loss component
        diversity_weight:
            Weight for diversity loss component
        init_points:
            Number of initial points for Bayesian Optimization.
        n_iter:
            Number of iterations for Bayesian Optimization.
        sampling_from_model:
            true if you want to generate samples from model after sampling from data and generating with relationship graph

        Returns
        -------
        np.ndarray:
            The generated counterfactuals that minimize the loss function.
        """
        if plausibility_weight > 0:
            if self.casual_order is None:
                raise ValueError("Casual order must be provided if plausibility loss is on")
            if self.adjacency_matrix is None:
                raise ValueError("adjacency_matrix must be provided")
            if self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
                raise ValueError("adjacency matrix must have same number of rows and columns")
            if self.adjacency_matrix.shape[0] != len(self.casual_order):
                raise ValueError("adjacency matrix must be of same length as casual order")

        if query_instance is None:
            raise ValueError("query_instance must not be None")
        if self.optimizer_type is None:
            raise ValueError("optimizer_type must be set via fit() before calling counterfactual()")
        if self.optimizer is None and self.optimizer_type is OptimizerType.Custom:
            raise ValueError("optimizer must be set before calling counterfactual()")
        if self.optimizer_type is OptimizerType.LinearAdditive:
            if not hasattr(self.blackbox, 'coef_'):
                raise AttributeError('optimizer requires model.coef_ as linear coefficients to be set')
        return generate_cfs(query_instance=query_instance,
                            desired_class=desired_class,
                            adjacency_matrix=self.adjacency_matrix,
                            casual_order=self.casual_order,
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




from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from abc import ABC, abstractmethod

from src.refactor.abstract.OptimizerType import OptimizerType
from src.refactor.model.ccfs import generate_cfs


class ACFX(ABC, BaseEstimator, TransformerMixin):
    """
        ACFX: A Counterfactual Explanation Model
    """
    def __init__(self, blackbox:ClassifierMixin):
        self.blackbox = blackbox
        self.query_instance = None
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
    def fit(self, X, query_instance,adjacency_matrix, casual_order, pbounds,y=None, masked_features = None,
            categorical_indicator=None, features_order=None):
        """
        Fits explainer to the sampled data and blackbox model provided in the constructor

        :param X:
        :type X: pd.DataFrame
        :param y:
        :return:

        Parameters
        ----------
        categorical_indicator
        """
        if y is not None:
            self.blackbox.fit(X, y)
        self.X = X
        self.categorical_indicator = categorical_indicator
        self.features_order = features_order
        self.query_instance = query_instance
        self.adjacency_matrix = adjacency_matrix
        self.casual_order = casual_order
        self.pbounds = pbounds
        self.masked_features = masked_features
        return self

    def predict(self, X):
        return self.blackbox.predict(X)

    def counterfactual(self, desired_class, num_counterfactuals=1, proximity_weight=1,
                       sparsity_weight=1, plausibility_weight=0, diversity_weight=1, init_points=10,
                       n_iter=1000, sampling_from_model=True):
        if self.query_instance is None:
            raise ValueError("query_instance must be set via fit() before calling counterfactual()")
        if self.optimizer_type is None:
            raise ValueError("optimizer_type must be set via fit() before calling counterfactual()")
        if self.optimizer is None and self.optimizer_type is OptimizerType.Custom:
            raise ValueError("optimizer must be set before calling counterfactual()")
        return generate_cfs(query_instance=self.query_instance,
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




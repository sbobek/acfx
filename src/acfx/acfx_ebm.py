from interpret.glassbox import ExplainableBoostingClassifier

from src.acfx.acfx import ACFX
from src.refactor.evaluation.EBMCounterOptimizer import EBMCounterOptimizer
from src.refactor.model.ccfs import generate_cfs


class AcfxEBM(ACFX):
    def __init__(self, blackbox: ExplainableBoostingClassifier):
        super().__init__(blackbox)
        self.optimizer = None
        self.X = None
        self.categorical_indicator = None
        self.features_order = None
        self.pbounds = None
        self.query_instance = None
        self.adjacency_matrix = None
        self.casual_order = None
        self.masked_features = None

    def predict(self, X):
        self.blackbox.predict(X)

    def counterfactual(self, desired_class, num_counterfactuals=1, proximity_weight=1,
                       sparsity_weight=1, plausibility_weight=0,diversity_weight=1, init_points=10, n_iter=1000, sample_from_model=False):
        return generate_cfs(query_instance=self.query_instance,
                            desired_class=desired_class,
                            adjacency_matrix=self.adjacency_matrix,
                            causal_order=self.casual_order,
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
                            sample_from_model=sample_from_model)

    def fit(self, X, query_instance,adjacency_matrix, casual_order, pbounds,y=None, masked_features = None,
            categorical_indicator=None, features_order=None):
        if y is not None:
            self.blackbox.fit(X,y)
        self.optimizer = EBMCounterOptimizer(self.blackbox, X)
        self.X = X
        self.categorical_indicator = categorical_indicator
        self.features_order = features_order
        self.query_instance = query_instance
        self.adjacency_matrix = adjacency_matrix
        self.casual_order = casual_order
        self.pbounds = pbounds
        self.masked_features = masked_features
        return self


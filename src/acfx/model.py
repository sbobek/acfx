from sklearn.base import BaseEstimator, TransformerMixin


class ACFX(BaseEstimator, TransformerMixin):
    """
        ACFX: A Counterfactual Explanation Model
    """
    def __init__(self, blackbox):
        pass

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        pass

    def counterfactual(self, instance):
        pass
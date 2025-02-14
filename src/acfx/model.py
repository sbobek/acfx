from sklearn.base import BaseEstimator, TransformerMixin


class ACFX(BaseEstimator, TransformerMixin):
    """
        ACFX: A Counterfactual Explanation Model
    """
    def __init__(self, blackbox):
        pass

    def fit(self, X=None, y=None):
        """
        Fits explainer to the sampled data and blackbox model provided in the constructor

        :param X:
        :type X: pd.DataFrame
        :param y:
        :return:
        """
        return self

    def predict(self, X):
        pass

    def counterfactual(self, instance):
        pass
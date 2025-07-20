from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from abc import ABC, abstractmethod

class ACFX(ABC, BaseEstimator, TransformerMixin):
    """
        ACFX: A Counterfactual Explanation Model
    """
    def __init__(self, blackbox:ClassifierMixin):
        self.blackbox = blackbox

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fits explainer to the sampled data and blackbox model provided in the constructor

        :param X:
        :type X: pd.DataFrame
        :param y:
        :return:
        """
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def counterfactual(self, instance):
        pass



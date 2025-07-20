from sklearn.base import ClassifierMixin

from src.acfx.acfx import ACFX


class AcfxLinear(ACFX):
    def __init__(self, blackbox: ClassifierMixin):
        super().__init__(blackbox)

    def predict(self, X):
        pass

    def counterfactual(self, instance):
        pass

    def fit(self, X=None, y=None):
        pass

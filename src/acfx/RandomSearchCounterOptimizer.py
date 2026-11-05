from typing import Dict, Tuple

import numpy as np
import pandas as pd
from .abstract import ModelBasedCounterOptimizer
from overrides import overrides


class RandomSearchCounterOptimizer(ModelBasedCounterOptimizer):
    """
        Random-search optimizer for model-based counterfactual generation.

        The optimizer starts from an instance constructed from the column-wise
        means of the reference dataset. It then generates candidate instances by
        sampling selected features uniformly within their configured bounds.

        Each candidate is evaluated using the model's predicted probability for
        the requested target class. The candidate with the highest observed
        target-class probability is returned.

        Parameters
        ----------
        model
            Predictive model used to evaluate candidate instances. The model must
            implement a ``predict_proba`` method.

        X : pandas.DataFrame of shape (n_samples, n_features)
            Reference dataset used to construct the starting instance. The
            starting instance is obtained by calculating the mean of each column
            in ``X``. The column names are also used to identify features during
            optimization.

        feature_bounds : dict[str, tuple[float, float]]
            Mapping from feature names to their sampling bounds. Each tuple
            contains the minimum and maximum value used for uniform random
            sampling of the corresponding feature.

        n_iter : int
            Number of randomly generated candidate instances to evaluate.
            The default is 100.

        Raises
        ------
        AttributeError
            If ``model`` does not implement a ``predict_proba`` method.

        Notes
        -----
        Only features that occur in both ``feature_masked`` and
        ``feature_bounds`` are modified during optimization. All other features
        retain their column-wise mean values from ``X``.

        The optimizer evaluates the initial mean-based instance before generating
        random candidates. Consequently, the returned instance may be the
        initial mean-based instance if none of the generated candidates obtains
        a higher predicted probability for the target class.

        No random seed is set by this class. Therefore, repeated calls may return
        different results unless the NumPy random generator is controlled
        externally.
        """
    def __init__(self, model, X: pd.DataFrame, feature_bounds: Dict[str, Tuple[float, float]], n_iter: int = 100):
        """
            Initialize the random-search counterfactual optimizer.

            Parameters
            ----------
            model
                Predictive model used to evaluate candidate instances. The model
                must implement a ``predict_proba`` method whose output can be
                indexed by sample and target class.

            X : pandas.DataFrame of shape (n_samples, n_features)
                Reference dataset used to construct the starting instance through
                column-wise means. Its column names identify the features
                considered during optimization.

            feature_bounds : dict[str, tuple[float, float]]
                Mapping from feature names to ``(min, max)`` bounds used for
                uniform random sampling.

            n_iter : int
                Number of randomly generated candidate instances to evaluate.
                The default is 100.

            Raises
            ------
            AttributeError
                If ``model`` does not implement a ``predict_proba`` method.
            """
        if not hasattr(model, 'predict_proba'):
            raise AttributeError("Model must implement predict_proba()")
        self.model = model
        self.X = X
        self.feature_bounds = feature_bounds
        self.n_iter = n_iter

    @overrides
    def optimize_proba(self, target_class: int, feature_masked: list[str]) -> Dict[str, float]:
        """
        Search for an instance with increased target-class probability.

        The search starts from an instance containing the column-wise means
        of ``X``. During each iteration, a new candidate is created from this
        same mean-based instance. Features listed in ``feature_masked`` are
        sampled independently from uniform distributions when corresponding
        bounds are available in ``feature_bounds``.

        A candidate replaces the current best instance only when its
        predicted probability for ``target_class`` is strictly greater than
        the current best score.

        Parameters
        ----------
        target_class : int
            Index of the target class in the probability array returned by
            ``model.predict_proba``.

        feature_masked : list[str]
            Feature names that may be modified during random search. A feature
            is modified only if it is also present in ``feature_bounds``.
            Features not included in this list retain their column-wise mean
            values.

        Returns
        -------
        dict[str, float]
            Best instance found during optimization, represented as a mapping
            from feature names to feature values. The result includes all
            columns present in ``X``, including features that were not
            modified during the search.

        Notes
        -----
        The initial score is calculated as::

            model.predict_proba([base_instance])[0][target_class]

        Candidate scores are calculated in the same way. This method assumes
        that the model accepts the supplied single-instance input and that
        the returned probability array can be indexed using
        ``[0][target_class]``.

        Sampling is performed with ``numpy.random.uniform``. Bounds are used
        directly as the lower and upper sampling limits.

        Each candidate is initialized independently from the mean-based
        instance. Changes from one iteration are therefore not carried into
        the next iteration.
        """
        base_instance = self.X.mean().copy()
        best_instance = base_instance.copy()
        best_score = self.model.predict_proba([base_instance])[0][target_class]

        for _ in range(self.n_iter):
            candidate = base_instance.copy()
            for feature_name in self.X.columns:
                if feature_name in feature_masked and feature_name in self.feature_bounds:
                    min_val, max_val = self.feature_bounds[feature_name]
                    candidate[feature_name] = np.random.uniform(min_val, max_val)

            score = self.model.predict_proba([candidate])[0][target_class]
            if score > best_score:
                best_score = score
                best_instance = candidate.copy()

        return best_instance.to_dict()
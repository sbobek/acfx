from typing import List, Dict

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from overrides import overrides
from sklearn.utils.extmath import softmax
import numpy as np

from ..abstract import ModelBasedCounterOptimizer


class EBMCounterOptimizer(ModelBasedCounterOptimizer):
    # Y_TEST = '_eco_y_test'
    # Y_PRED = '_eco_y_pred'
    # IS_MODIFIABLE = '_eco_is_modifiable'

    def __init__(self, model: ExplainableBoostingClassifier, X: pd.DataFrame):
        self.model = model
        self.X = X
        self.updated_features = {}

    def _get_optimized_feature_value(self, feature_name, feature_idx, feature_val, features, feature_masked, term_idx,
                                     class_idx) -> Dict[str, float]:
        """
        Returns
        -------
        feature value with maximum score for given target class.

        @Todo Needs changes to return optimized value due to given strategy.
        """
        # if feature is modifiable and not yet optimized
        if feature_name in feature_masked and feature_name not in self.updated_features:
            # if multiclass classification take bins for term and class
            if len(self.model.term_scores_[term_idx].shape) > 1:
                class_term_scores = self.model.term_scores_[term_idx].T[class_idx]
            else:
                # else take score for class 1 or 1 - score for class 1
                class_term_scores = self.model.term_scores_[term_idx] if class_idx == 1 else 1 - self.model.term_scores_[
                    term_idx]
            # take term that gives best score for target class
            class_max = np.max(class_term_scores)
            try:
                feature_score_idx = np.where(class_term_scores[1:-1] == class_max)[0][0]  ##this is score, not value imho
            except:
                print(np.where(class_term_scores[1:-1] == class_max))
            # we bin differently for main effects and pairs, so first
            # get the list containing the bins for different resolutions
            bin_levels = self.model.bins_[feature_idx]
            # print(f'Feature score index for feature {feature_name} is {feature_score_idx} which represents score equal: {class_max} test: {class_term_scores[feature_score_idx+1]}')
            # what resolution do we need for this term (main resolution, pair
            # resolution, etc.), but limit to the last resolution available
            bins = bin_levels[min(len(bin_levels), len(features)) - 1]

            if len(bins) == 0:
                feature_val = self.X[feature_name].sample(1).values[0]
            else:
                if isinstance(bins, dict):
                    # categorical feature
                    # 'unknown' category strings are in the last bin (-1)
                    feature_val = list(bins.values())[
                        feature_score_idx - 1]  # if maxscore was 0, or -1 just assign random value
                else:
                    # continuous feature
                    # Get the lower and upper bounds of the specified bin
                    lower_idx = feature_score_idx - 1
                    upper_idx = feature_score_idx

                    if lower_idx == -1:
                        lower = self.model.feature_bounds_[feature_idx][0]
                    else:
                        lower = bins[lower_idx]

                    if upper_idx == len(bins):
                        upper = self.model.feature_bounds_[feature_idx][1]
                    else:
                        upper = bins[upper_idx]
                    # print(f'Drawing randomly from :{lower} to {upper}')

                    # Draw a random number from the range defined by the bin
                    feature_val = np.random.uniform(lower, upper)

            # print(f'This translates into feature value: {feature_val}')

            self.updated_features.update({feature_name: feature_val})
        elif feature_name in self.updated_features:
            feature_val = self.updated_features.get(feature_name)
        else:
            self.updated_features.update({feature_name: feature_val})

        return feature_val

    @overrides
    def optimize_proba(self, target_class : int, feature_masked: List[str]) -> Dict[str, float]:
        """
        The method calculates probabilities taking into account the optimization of given parameters towards the target class.
        Method is based on a default ebm's predict_proba

        Parameters:
        ebm:
            Trained EBM model
        X:
            Dataset
        target_class:
            Target class from which we take the features
        featured_masked:
            List of interchangeable features
        """
        if target_class not in self.model.classes_:
            raise KeyError(f'Class "{target_class}" does not exists in given EBM model')

        class_idx = np.where(self.model.classes_ == target_class)[0][0]
        self.updated_features = {}
        sample_scores = []
        cf = {}
        for index, sample in self.X.iterrows():
            # start from the intercept for each sample
            score = self.model.intercept_.copy()
            if isinstance(score, float) or len(score) == 1:
                # regression or binary classification
                score = float(score)

            # we have 2 terms, so add their score contributions
            for term_idx, features in enumerate(self.model.term_features_):
                # indexing into a tensor requires a multi-dimensional index
                tensor_index = []
                # main effects will have 1 feature, and pairs will have 2 features
                for feature_idx in features:
                    feature_name = self.model.feature_names_in_[feature_idx]  # Get the feature name by index
                    feature_val = sample[feature_name]  # Use the feature name to get the correct value from the sample
                    bin_idx = 0  # if missing value, use bin index 0

                    if feature_val is not None and feature_val is not np.nan:
                        # we bin differently for main effects and pairs, so first
                        # get the list containing the bins for different resolutions
                        bin_levels = self.model.bins_[feature_idx]

                        # what resolution do we need for this term (main resolution, pair
                        # resolution, etc.), but limit to the last resolution available
                        bins = bin_levels[min(len(bin_levels), len(features)) - 1]

                        # here is where the magic is located
                        feature_val = self._get_optimized_feature_value(feature_name, feature_idx, feature_val,
                                                                        features, feature_masked, term_idx, class_idx)

                        if isinstance(bins, dict):
                            # categorical feature
                            # 'unknown' category strings are in the last bin (-1)
                            bin_idx = bins.get(feature_val, -1)
                            if bin_idx == -1:
                                # check value as string
                                bin_idx = bins.get(str(feature_val), -1)
                        else:
                            # continuous feature
                            try:
                                # try converting to a float, if that fails it's 'unknown'
                                feature_val = float(feature_val)
                                # add 1 because the 0th bin is reserved for 'missing'
                                bin_idx = np.digitize(feature_val, bins) + 1
                            except ValueError:
                                # non-floats are 'unknown', which is in the last bin (-1)
                                bin_idx = -1

                        if len(self.model.term_scores_[term_idx].shape) > 1:
                            sc = self.model.term_scores_[term_idx].T[class_idx][bin_idx]
                        else:
                            sc = self.model.term_scores_[term_idx][bin_idx]
                        # print(f'And feature value {feature_val} translates back to bin index: {bin_idx} which represents score: {sc}')

                        tensor_index.append(bin_idx)

                # local_score is also the local feature importance
                local_score = self.model.term_scores_[term_idx][tuple(tensor_index)]

                score += local_score
            sample_scores.append(score)

        predictions = np.array(sample_scores)

        if hasattr(self.model, 'classes_'):
            # classification
            if len(self.model.classes_) == 2:
                # binary classification

                # softmax expects two logits for binary classification
                # the first logit is always equivalent to 0 for binary classification
                predictions = [[0, x] for x in predictions]
            predictions = softmax(predictions)

        #return predictions, self.updated_features
        return self.updated_features

    # def check_samples(self, target_class, y_test_key, feature_masked):
    #     X = self.X.copy()
    #     predictions = self.optimize_proba(target_class, feature_masked)
    #     X.loc[:, self.Y_TEST] = np.argmax(predictions, axis=1)
    #     X.loc[:, self.Y_PRED] = X[self.Y_TEST].map({key: val for key, val in enumerate(self.model.classes_)})
    #     X.loc[:, self.IS_MODIFIABLE] = np.where(X[y_test_key] != X[self.Y_PRED], 1, 0)
    #
    #     return X
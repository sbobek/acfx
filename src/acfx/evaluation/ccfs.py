import random
from typing import Sequence, Dict, Tuple, Optional, List

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import ClassifierMixin
from sklearn.linear_model._base import LinearClassifierMixin
from ..abstract import ModelBasedCounterOptimizer
from ..abstract import OptimizerType
from .LogisticRegressionCounterOptimizer import LogisticRegressionCounterOptimizer
from .loss import compute_loss
from .EBMCounterOptimizer import EBMCounterOptimizer

def __generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
                         plausibility_weight,
                         diversity_weight, bounds, model, optimizer:ModelBasedCounterOptimizer, sampling_from_model:bool,
                         categorical_indicator=None, features_order=None, masked_features=None, cfs=[], X=None,
                         init_points=5, n_iter=1000):
    """
    Generate a single counterfactual that minimizes the loss function using Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def update_masked_features_dict(features_order, masked_features, **params):
        for feature, value in zip(features_order, query_instance):
            if feature not in masked_features:
                params[feature] = value
        return params

    def black_box_function(trial):
        # Suggest values for each feature based on bounds
        kwargs = {}
        for i, key in enumerate(features_order):
            if categorical_indicator[i]:
                kwargs[key] = trial.suggest_categorical(key, list(range(int(bounds[key][0]), int(bounds[key][1]) + 1)))
            else:
                kwargs[key] = trial.suggest_float(key, bounds[key][0], bounds[key][1], log=False)

        kwargs = update_masked_features_dict(features_order, masked_features, **kwargs)
        # todo rounding
        cf = np.array([[int(round(kwargs[key])) if categorical_indicator[i] else kwargs[key]
                        for i, key in enumerate(features_order)]]).reshape(1, -1)

        if len(cfs) > 0:
            cfst = np.vstack(cfs + [cf])
        else:
            cfst = cf

        loss = compute_loss(model, cf, query_instance, desired_class, adjacency_matrix,
                            causal_order, proximity_weight, sparsity_weight, plausibility_weight,
                            diversity_weight, pbounds=bounds, features_order=features_order,
                            masked_features=masked_features,
                            categorical=categorical_indicator, allcfs=cfst)
        # Return the first value in the loss array, Optuna needs a single scalar value to minimize
        return -loss[0, 1]

    # Initialize Optuna study
    study = optuna.create_study(direction='maximize')

    # Define seen_points set to track uniqueness of points
    seen_points = set()

    def is_unique_point(point_dict):
        point_tuple = tuple(point_dict[key] for key in features_order)
        if point_tuple in seen_points:
            return False
        seen_points.add(point_tuple)
        return True

    def clip_values(cf, pbounds):
        # Create a new dictionary with clipped values based on the bounds in pbounds
        return {feature: np.clip(value, pbounds[feature][0], pbounds[feature][1])
                for feature, value in cf.items()}


    def sample_from_data(Xdesired, features_order, init_points, sampled_trials, study):
        sample_size = min(int(init_points * 0.5), len(Xdesired))
        Xsample = Xdesired.sample(sample_size)
        for i, r in Xsample.iterrows():
            candidate_point = dict(r[features_order])
            if is_unique_point(candidate_point):
                trial_params = {key: candidate_point[key] for key in features_order}
                study.enqueue_trial(trial_params)
                sampled_trials += 1
        init_points = max(0, init_points - sampled_trials)
        return init_points, sample_size, sampled_trials

    def sample_from_model(Xdesired, bounds, desired_class, features_order, init_points,
                          masked_features, sample_size, sampled_trials, study):
        def optimizer_iteration(masked_features, total_lists: list, optimizer: ModelBasedCounterOptimizer,
                                desired_class):
            num_elements = random.randint(1, len(masked_features))
            selected_features = random.sample(masked_features, num_elements)
            set_to_check = set(selected_features)
            # Check if the set is in the list of sets
            found = any(set(item) == set_to_check for item in total_lists)
            if found:
                return None
            total_lists.append(selected_features)
            try:
                cf = optimizer.optimize_proba(desired_class, feature_masked=selected_features)
                return cf
            except Exception as ex:
                print(f'optimize_proba error occured: {ex}')
                return None
        print(
            f'Sampling from model... Already sampled {sample_size} from {Xdesired.shape[0]} possible in data sampling...')
        # optimizer = get_ebm_optimizer(model, pd.DataFrame(query_instance.reshape(1, -1), columns=features_order))
        total_lists = []
        for i in range(min(init_points, 2 ** len(masked_features))):
            cf = optimizer_iteration(masked_features, total_lists, optimizer, desired_class)
            if cf is None:
                continue

            # check if values of cf are aligned with the ranges, and if not, clip it to the range
            cf = clip_values(cf, bounds)

            if isinstance(cf, dict):
                cf_dict = cf
            elif isinstance(cf, np.ndarray) or isinstance(cf, list):
                cf_dict = {key: cf[i] for i, key in enumerate(features_order)}
            else:
                raise ValueError("Unexpected format for cf")

            if is_unique_point(cf_dict):
                sampled_trials += 1
                study.enqueue_trial(cf_dict)
        return sampled_trials

    sampled_trials = 0
    if X is not None:
        Xdesired = X[model.predict(X) == desired_class].drop_duplicates()
        init_points, sample_size, sampled_trials = sample_from_data(X, features_order,
                                                                              init_points,
                                                                              sampled_trials, study)
        if init_points > 0 and sampling_from_model:
            sampled_trials = sample_from_model(Xdesired, bounds, desired_class, features_order,
                                               init_points, masked_features, sample_size, sampled_trials,
                                               study)
    else:
        raise ValueError('X is None')

    # Optimize the study with the defined number of iterations
    study.optimize(black_box_function, n_trials=n_iter + sampled_trials)

    # Extract the best parameters (counterfactual)
    best_params = study.best_params
    best_params = update_masked_features_dict(features_order, masked_features, **best_params)
    # todo rounding should be done to the featoure boundaries, not to nearest integer
    best_cf = np.array([[int(round(best_params[key])) if categorical_indicator[i] else best_params[key]
                         for i, key in enumerate(features_order)]]).reshape(1, -1)

    return best_cf

def _generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
                        plausibility_weight,
                        diversity_weight, bounds, model, categorical_indicator=None, features_order=None,
                        masked_features=None, cfs=[], X=None, init_points=5, n_iter=1000,
                        optimizer_type: OptimizerType = OptimizerType.EBM, optimizer=None, sampling_from_model=True):

    def is_null_or_empty(var):
        if var is None:
            return True
        if isinstance(var, (list, tuple, dict, set)) and len(var) == 0:
            return True
        if isinstance(var, np.ndarray) and var.size == 0:
            return True
        return False

    def get_optimizer():
        def get_ebm_optimizer(model_classifier: ExplainableBoostingClassifier, query_instance: pd.DataFrame):
            return EBMCounterOptimizer(model_classifier, query_instance)
        def get_logistic_regression_optimizer(model_classifier: LinearClassifierMixin, query_instance: pd.DataFrame):
            return LogisticRegressionCounterOptimizer(model_classifier, query_instance, bounds)

        if optimizer_type == OptimizerType.EBM:
            return get_ebm_optimizer(model, pd.DataFrame(query_instance.reshape(1, -1), columns=features_order))
        if optimizer_type == OptimizerType.LinearAdditive:
            lro = get_logistic_regression_optimizer(model,
                                                    pd.DataFrame(query_instance.reshape(1, -1), columns=features_order))
            return lro
        if optimizer_type == OptimizerType.Custom:
            return optimizer
        else:
            raise NotImplementedError()

    if is_null_or_empty(categorical_indicator):
        categorical_indicator = [False] * len(bounds)
    if is_null_or_empty(features_order):
        features_order = list(bounds.keys())
    if is_null_or_empty(masked_features):
        masked_features = features_order

    return __generate_single_cf(query_instance=query_instance, desired_class=desired_class, adjacency_matrix=adjacency_matrix,
                                causal_order=causal_order, proximity_weight=proximity_weight, sparsity_weight=sparsity_weight,
                                plausibility_weight=plausibility_weight, diversity_weight=diversity_weight, bounds=bounds,
                                model=model, optimizer=get_optimizer(), sampling_from_model= sampling_from_model,
                                categorical_indicator=categorical_indicator, features_order= features_order,
                                masked_features=masked_features, cfs=cfs, X= X, init_points= init_points, n_iter=n_iter)

def generate_cfs(query_instance:np.ndarray, desired_class:int, adjacency_matrix:np.ndarray, casual_order : Sequence[int], proximity_weight : float,
                 sparsity_weight: float, plausibility_weight: float, diversity_weight: float, bounds:Dict[str, Tuple[float, float]],
                 model:ClassifierMixin, features_order:Optional[List[str]] =None,
                 masked_features:Optional[List[str]] =None, categorical_indicator:Optional[List[bool]] =None, X:Optional[pd.DataFrame] =None,
                 num_cfs:int=1, init_points:int=10, n_iter:int=1000, sampling_from_model:bool=False,
                 optimizer_type : OptimizerType = OptimizerType.EBM, optimizer : ModelBasedCounterOptimizer=None) -> np.ndarray:
    """
    Generate multiple counterfactuals that minimize the loss function using Bayesian Optimization.

    Parameters:
        query_instance:
            The instance to generate counterfactuals for.
        desired_class:
            The target class for the counterfactuals.
        adjacency_matrix:
            The adjacency matrix representing the causal structure.
        casual_order:
            The order of variables in the causal graph.
        proximity_weight:
            Weight for proximity loss component
        sparsity_weight:
            Weight for sparsity loss component
        plausibility_weight:
            Weight for plausibility loss component
        diversity_weight:
            Weight for diversity loss component
        bounds:
            The bounds for each feature to search over (dict with feature names as keys and tuple (min, max) as values).
        model:
            The predictive model used to predict class labels.
        features_order:
            order of features in query instance
        masked_features:
            masked features vector (features to skip)
        categorical_indicator:
            True at the index where the variable should be treated as categorical
        num_cfs:
            The number of counterfactual instances to generate.
        init_points:
            Number of initial points for Bayesian Optimization.
        n_iter:
            Number of iterations for Bayesian Optimization.
        X:
            training dataset to sample from.
        sampling_from_model:
            true if you want to generate samples from model after sampling from data and generating with relationship graph
        optimizer_type:
            type of optimizer used on model to generate counterfactuals. If you use OptimizerType.Custom, you need to provide optimizer instance
        optimizer:
            instance of optimizer (use for optimizer_type = OptimizerType.Custom)

    Returns
    -------
    np.ndarray:
        The generated counterfactuals that minimize the loss function.
    """
    cfs = []
    for _ in range(num_cfs):
        cf = _generate_single_cf(query_instance, desired_class, adjacency_matrix,
                                 casual_order, proximity_weight, sparsity_weight,
                                 plausibility_weight, diversity_weight,
                                 bounds, model, categorical_indicator, features_order, masked_features=masked_features,
                                 cfs=cfs, X=X, init_points=init_points, n_iter=n_iter,
                                 sampling_from_model=sampling_from_model, optimizer_type=optimizer_type, optimizer=optimizer)
        cfs.append(cf)

    return np.vstack(cfs)

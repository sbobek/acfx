import random
import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from src.refactor.abstract.ModelBasedCounterOptimizer import ModelBasedCounterOptimizer
from src.refactor.abstract.OptimizerType import OptimizerType
from src.refactor.evaluation.LogisticRegressionCounterOptimizer import LogisticRegressionCounterOptimizer
from src.refactor.evaluation.casual_counterfactuals import compute_loss
from src.refactor.evaluation.EBMCounterOptimizer import EBMCounterOptimizer
from src.refactor.evaluation.multi_dataset_evaluation import log2file


def __generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
                         plausibility_weight,
                         diversity_weight, bounds, model, optimizer:ModelBasedCounterOptimizer, sampling_from_model:bool,
                         categorical_indicator=None, features_order=None, masked_features=None, cfs=[], X=None,
                         init_points=5, n_iter=1000):
    """
    Generate a single counterfactual that minimizes the loss function using Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if categorical_indicator is None:
        categorical_indicator = [False] * len(bounds)
    if features_order is None:
        features_order = list(bounds.keys())
    if masked_features is None:
        masked_features = features_order

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
                kwargs[key] = trial.suggest_uniform(key, bounds[key][0], bounds[key][1])

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


    def sample_from_data(Xdesired, features_order, init_points, is_unique_point, sampled_trials, study):
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

    def sample_from_model(Xdesired, bounds, clip_values, desired_class, features_order, init_points,
                          is_unique_point, masked_features, sample_size, sampled_trials, study):
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
                _, cf = optimizer.optimize_proba(desired_class, feature_masked=selected_features)
                return cf
            except Exception as ex:
                log2file(f'optimize_proba error occured: {ex}')
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
                                                                              init_points, is_unique_point,
                                                                              sampled_trials, study)
        if init_points > 0 and sampling_from_model:
            sampled_trials = sample_from_model(Xdesired, bounds, clip_values, desired_class, features_order,
                                               init_points, is_unique_point, masked_features, sample_size, sampled_trials,
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
                        diversity_weight, bounds, model,categorical_indicator=None, features_order=None,
                        masked_features=None, cfs=[], X=None,init_points=5, n_iter=1000,
                        optimizer_type: OptimizerType = OptimizerType.EBM, optimizer=None, sample_from_model=True):
    def get_optimizer():
        def get_ebm_optimizer(model: ExplainableBoostingClassifier, query_instance: pd.DataFrame):
            return EBMCounterOptimizer(model, query_instance)
        def get_logistic_regression_optimizer(model, query_instance: pd.DataFrame):
            return LogisticRegressionCounterOptimizer(model, query_instance)

        if optimizer_type == OptimizerType.EBM:
            return get_ebm_optimizer(model, pd.DataFrame(query_instance.reshape(1, -1), columns=features_order))
        if optimizer_type == OptimizerType.LogisticRegression:
            lro = get_logistic_regression_optimizer(model,
                                                    pd.DataFrame(query_instance.reshape(1, -1), columns=features_order))
            lro.set_feature_bounds(bounds)
            return lro
        if optimizer_type == OptimizerType.Custom:
            return optimizer
        else:
            raise NotImplementedError()
    return __generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
                                plausibility_weight,
                                diversity_weight, bounds, model, get_optimizer(), categorical_indicator, features_order,
                                masked_features, cfs, X, init_points, n_iter, sample_from_model)



def generate_cfs(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight,
                 sparsity_weight, plausibility_weight, diversity_weight, bounds, model, features_order,
                 masked_features=None, categorical_indicator=None, X=None,
                 num_cfs=1, init_points=10, n_iter=1000, sample_from_model=False):
    """
    Generate multiple counterfactuals that minimize the loss function using Bayesian Optimization.

    Parameters:
        query_instance: The instance to generate counterfactuals for.
        desired_class: The target class for the counterfactuals.
        adjacency_matrix: The adjacency matrix representing the causal structure.
        causal_order: The order of variables in the causal graph.
        proximity_weight: Weight for proximity loss component
        sparsity_weight: Weight for sparsity loss component
        plausibility_weight: Weight for plausibility loss component
        diversity_weight: Weight for diversity loss component
        bounds: The bounds for each feature to search over (dict with feature names as keys and tuple (min, max) as values).
        model: The predictive model used to predict class labels.
        features_order: order of features in query instance (zweryfikowac)
        masked_features: masked features vector (features to skip)
        categorical_indicator: True at the index where the variable should be treated as categorical
        num_cfs: The number of counterfactual instances to generate.
        init_points: Number of initial points for Bayesian Optimization.
        n_iter: Number of iterations for Bayesian Optimization.
        X: training dataset to sample from.

    Returns:
        The generated counterfactuals that minimize the loss function.
    """
    cfs = []
    for _ in range(num_cfs):
        cf = _generate_single_cf(query_instance, desired_class, adjacency_matrix,
                                 causal_order, proximity_weight, sparsity_weight,
                                 plausibility_weight, diversity_weight,
                                 bounds, model, categorical_indicator, features_order, masked_features=masked_features,
                                 cfs=cfs, X=X, init_points=init_points, n_iter=n_iter)
        cfs.append(cf)

    return np.vstack(cfs)

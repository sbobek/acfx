import random
import uuid
from typing import Sequence, Dict, Tuple, Optional, List
import traceback
import numpy as np
import optuna
import pandas as pd

from interpret.glassbox import ExplainableBoostingClassifier
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.base import ClassifierMixin
from sklearn.linear_model._base import LinearClassifierMixin
from ..abstract import ModelBasedCounterOptimizer
from ..abstract import OptimizerType
from .LogisticRegressionCounterOptimizer import LogisticRegressionCounterOptimizer
from .loss import compute_loss, compute_loss_bayesian
from .EBMCounterOptimizer import EBMCounterOptimizer
from .bayesian_model import discretize_ndarray, compute_bin_edges

def __generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
                         plausibility_weight,
                         diversity_weight, bounds, model, optimizer:ModelBasedCounterOptimizer, sampling_from_model:bool,
                         categorical_indicator=None, features_order=None, masked_features=None, cfs=[], X=None,
                         init_points=5, n_iter=1000, bayesian_causality : bool = False, bins_num: Optional[int] = None,
                         bayesian_model: Optional[DiscreteBayesianNetwork] = None):
    """
       Generate a single counterfactual that minimizes the loss function using Optuna.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def update_masked_features_dict(features_order, masked_features, **params):
        qi = query_instance
        if (1,len(features_order)) == query_instance.shape:
            qi = query_instance.flatten()
        for feature, value in zip(features_order, qi):
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

        if not bayesian_causality:
            loss = compute_loss(model,
                                cf,
                                query_instance,
                                desired_class,
                                adjacency_matrix,
                                causal_order,
                                proximity_weight,
                                sparsity_weight,
                                plausibility_weight,
                                diversity_weight,
                                pbounds=bounds,
                                features_order=features_order,
                                masked_features=masked_features,
                                categorical=categorical_indicator,
                                allcfs=cfst)
        else:
            categorized = discretize_ndarray(cf, categorical_indicator, bin_edges)
            loss = compute_loss_bayesian(model, cf, categorized, query_instance, desired_class, bayesian_model,
                                         proximity_weight, sparsity_weight, plausibility_weight,
                                diversity_weight, pbounds=bounds, features_order=features_order,
                                masked_features=masked_features,
                                categorical=categorical_indicator, allcfs=cfst)
        # Return the first value in the loss array, Optuna needs a single scalar value to minimize
        return -loss[0, 1]

    # Initialize Optuna study
    study_name = f"study_{uuid.uuid4().hex}"
    study = optuna.create_study(direction='maximize', storage=None, study_name=study_name)

    # Define seen_points set to track uniqueness of points
    seen_points = set()

    def is_unique_point(point_dict, pbounds):
        for key, (min_val, max_val) in pbounds.items():
            val = point_dict.get(key)
            if val is None or not (min_val <= val <= max_val):
                return False

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
            if is_unique_point(candidate_point, bounds):
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
                print(traceback.format_exc())
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

            if is_unique_point(cf_dict, bounds):
                sampled_trials += 1
                study.enqueue_trial(cf_dict)
        return sampled_trials

    sampled_trials = 0
    if X is not None:
        Xdesired = X[model.predict(X) == desired_class].drop_duplicates()
        init_points, sample_size, sampled_trials = sample_from_data(Xdesired, features_order,
                                                                              init_points,
                                                                              sampled_trials, study)
        if init_points > 0 and sampling_from_model:
            sampled_trials = sample_from_model(Xdesired, bounds, desired_class, features_order,
                                               init_points, masked_features, sample_size, sampled_trials,
                                               study)
    else:
        raise ValueError('X is None')

    # candidates must use the bins of the data the network was trained on; ACFX.fit trains it on this X,
    # a pre-fitted bayesian_model must come from the same X and number of bins (this is not validated)
    bin_edges = compute_bin_edges(X[features_order].to_numpy(), categorical_indicator, bins_num) \
        if bayesian_causality else None

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
                        optimizer_type: OptimizerType = OptimizerType.EBM, optimizer=None, sampling_from_model=True,
                        bayesian_causality : bool = False, bins_num: Optional[int] = None, bayesian_model: Optional[DiscreteBayesianNetwork] = None):
    """
       Normalize counterfactual-generation inputs and generate one counterfactual.

       This private wrapper supplies defaults for feature metadata, selects the
       appropriate model-based optimizer, validates the Bayesian causal-loss
       configuration, and delegates the optimization to
       ``__generate_single_cf``.

       If ``categorical_indicator`` is ``None`` or empty, every feature is
       treated as non-categorical. If ``features_order`` is ``None`` or empty,
       feature order is taken from ``bounds``. If ``masked_features`` is
       ``None`` or empty, all features may be modified.

       In Bayesian causal-loss mode, both ``bayesian_model`` and ``bins_num``
       must be provided.

       Raises
       ------
       ValueError
           If Bayesian causal-loss mode is enabled without ``bayesian_model``.

       ValueError
           If Bayesian causal-loss mode is enabled without ``bins_num``.

       NotImplementedError
           If the requested optimizer type is not supported.
       """

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

    if bayesian_causality:
        if bayesian_model is None:
            raise ValueError("bayesian_model must be provided for bayesian causal loss mode")
        if bins_num is None:
            raise ValueError("bins_num must be provided for bayesian causal loss mode")
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
                                masked_features=masked_features, cfs=cfs, X= X, init_points= init_points, n_iter=n_iter, bayesian_causality= bayesian_causality,
                                bins_num=bins_num, bayesian_model = bayesian_model)


def generate_cfs_bayesian(query_instance:np.ndarray, desired_class:int, bayesian_model: DiscreteBayesianNetwork, categorical_bins_num: int,
                          proximity_weight : float, sparsity_weight: float, plausibility_weight: float, diversity_weight: float,
                          bounds:Dict[str, Tuple[float, float]], model:ClassifierMixin, features_order:Optional[List[str]] =None,
                 masked_features:Optional[List[str]] =None, categorical_indicator:Optional[List[bool]] =None, X:Optional[pd.DataFrame] =None,
                 num_cfs:int=1, init_points:int=10, n_iter:int=1000, sampling_from_model:bool=False,
                 optimizer_type : OptimizerType = OptimizerType.EBM, optimizer : ModelBasedCounterOptimizer=None)-> np.ndarray:
    """
    Generate counterfactuals using a Bayesian-network-based causal loss.

    This function generates ``num_cfs`` counterfactual instances by repeatedly
    optimizing a weighted loss function. A separate Optuna study is created
    for each counterfactual.

    Candidate instances are evaluated using the supplied predictive model and
    the Bayesian causal-loss calculation implemented by
    ``compute_loss_bayesian``. Continuous values are discretized using bin
    edges calculated from ``X`` before the Bayesian causal loss is evaluated.

    Counterfactuals generated earlier in the same call are passed to later
    optimization runs. This allows the diversity component of the loss to
    account for previously generated counterfactuals.

    Parameters
    ----------
    query_instance : numpy.ndarray
        Instance for which counterfactual explanations are generated. The
        optimization procedure supports an instance represented as a
        one-dimensional array or as an array with shape ``(1, n_features)``.

        Features not included in ``masked_features`` are restored to their
        values from this instance before a candidate is evaluated and before
        the final counterfactual is returned.

    desired_class : int
        Target class for the generated counterfactuals. The value is passed
        to the predictive model and loss calculation. It is also used to
        select rows from ``X`` for initial sampling.

    bayesian_model : DiscreteBayesianNetwork
        Fitted discrete Bayesian network used by the Bayesian causal-loss
        calculation.

        The model must correspond to the same reference data and the same
        number of discretization bins used during counterfactual generation.
        This compatibility is required by the implementation but is not
        validated by this function.

    categorical_bins_num : int
        Number of bins used to discretize continuous features for the
        Bayesian causal-loss calculation. Bin edges are computed from
        ``X[features_order]`` using this value.

    proximity_weight : float
        Weight of the proximity component passed to
        ``compute_loss_bayesian``.

    sparsity_weight : float
        Weight of the sparsity component passed to
        ``compute_loss_bayesian``.

    plausibility_weight : float
        Weight of the Bayesian-network-based plausibility component passed to
        ``compute_loss_bayesian``.

    diversity_weight : float
        Weight of the diversity component passed to
        ``compute_loss_bayesian``. Previously generated counterfactuals are
        supplied to the loss calculation when available.

    bounds : dict[str, tuple[float, float]]
        Search bounds for individual features. Each key is a feature name and
        each value is a ``(minimum, maximum)`` tuple.

        For categorical features, the bounds are converted to integers and
        used to construct an inclusive sequence of categories. For other
        features, Optuna samples floating-point values between the supplied
        lower and upper bounds.

        Values obtained from model-based initial sampling are clipped to
        these bounds before being added to the Optuna study.

    model : ClassifierMixin
        Predictive classifier used to evaluate counterfactual candidates.

        The implementation calls ``model.predict(X)`` to identify reference
        samples assigned to ``desired_class``. Additional model operations
        may be performed by ``compute_loss_bayesian`` and by the selected
        initial-sampling optimizer.

    features_order : list[str] or None
        Feature order used to align the query instance, bounds, categorical
        indicators, candidate arrays, and columns of ``X``.

        If ``None`` or empty, the function uses ``list(bounds.keys())``.

    masked_features : list[str] or None
        Names of features that may be changed during optimization.

        Features not included in this list are fixed to their corresponding
        values from ``query_instance``. If ``None`` or empty, all features in
        ``features_order`` may be changed.

    categorical_indicator : list[bool] or None
        Boolean indicators specifying which features are categorical. The
        order must correspond to ``features_order``.

        For a categorical feature, Optuna selects an integer category in the
        inclusive range defined by the corresponding bounds. The resulting
        value is rounded and converted to ``int`` when constructing a
        candidate array.

        If ``None`` or empty, all features are treated as non-categorical.

    X : pandas.DataFrame of shape (n_samples, n_features) or None
        Reference dataset used for initial candidate sampling and for
        calculating discretization bin edges.

        Rows predicted by ``model`` as ``desired_class`` are selected and
        duplicate rows are removed before data-based initial sampling.

        This argument is required by the implementation. A ``ValueError`` is
        raised if it is ``None``.

    num_cfs : int
        Number of counterfactual instances to generate. The default is 1.

        Counterfactuals are generated sequentially. Each generated
        counterfactual is added to the collection supplied to the next
        optimization run.

    init_points : int
        Requested number of initial candidate points used to seed each Optuna
        study. The default is 10.

        Up to ``int(init_points * 0.5)`` rows are sampled from reference
        observations predicted as ``desired_class``. The actual number is
        limited by the number of available target-class rows.

        If initial points remain and ``sampling_from_model`` is True,
        additional candidates are requested from the optimizer.

    n_iter : int
        Number of optimization trials requested after initial candidates have
        been enqueued. The study is executed with
        ``n_trials=n_iter + sampled_trials``.

        The default is 1000.

    sampling_from_model : bool
        Whether to request additional initial candidates from the selected
        model-based optimizer after sampling candidates from ``X``.

        Model-based sampling is performed only when fewer than
        ``init_points`` candidates were obtained from the reference data.

    optimizer_type : OptimizerType
        Optimizer type used to construct or select the model-based optimizer
        for initial candidate generation.

        The implementation handles:

        * ``OptimizerType.EBM``, which creates an ``EBMCounterOptimizer``;
        * ``OptimizerType.LinearAdditive``, which creates a
          ``LogisticRegressionCounterOptimizer``;
        * ``OptimizerType.Custom``, which uses the supplied ``optimizer``.

        Other optimizer types cause a ``NotImplementedError``.

    optimizer : ModelBasedCounterOptimizer or None
        Custom model-based optimizer used when ``optimizer_type`` is
        ``OptimizerType.Custom``.

        Its ``optimize_proba`` method is called with ``desired_class`` and a
        randomly selected non-empty subset of ``masked_features`` when
        model-based initial sampling is enabled.

    Returns
    -------
    numpy.ndarray of shape (num_cfs, n_features)
        Generated counterfactual instances. Individual counterfactual arrays
        are vertically stacked before being returned.

    Raises
    ------
    ValueError
        If ``bayesian_model`` is ``None``.

    ValueError
        If ``categorical_bins_num`` is ``None``.

    ValueError
        If ``X`` is ``None``.

    ValueError
        If a model-based optimizer returns a candidate in a format other than
        ``dict``, ``numpy.ndarray``, or ``list``.

    NotImplementedError
        If ``optimizer_type`` is not one of the optimizer types handled by
        the implementation.

    Notes
    -----
    Each counterfactual is generated using a newly created in-memory Optuna
    study with the direction set to ``"maximize"``.

    Initial candidates are sampled from rows in ``X`` for which
    ``model.predict(X) == desired_class``. Duplicate rows are removed before
    sampling. Data-based sampling uses ``pandas.DataFrame.sample`` without an
    explicit ``random_state``.

    When model-based sampling is enabled, the implementation randomly selects
    non-empty subsets of ``masked_features`` and avoids requesting the same
    feature subset more than once during one counterfactual-generation run.

    Candidate points are tracked to avoid enqueuing exact duplicates. A point
    is accepted only when it contains every feature required by
    ``features_order`` and all checked values lie within the corresponding
    bounds.

    No random seed is configured for Python's ``random`` module, pandas
    sampling, NumPy, or Optuna. Results may therefore vary between calls
    unless randomness is controlled externally.

    The function does not independently verify that every returned instance
    is classified as ``desired_class``. Candidate quality is determined by
    the loss function and the best Optuna trial.
    """
    cfs = []
    for _ in range(num_cfs):
        cf = _generate_single_cf(query_instance, desired_class, None,
                                 None, proximity_weight, sparsity_weight,
                                 plausibility_weight, diversity_weight,
                                 bounds, model, categorical_indicator, features_order, masked_features=masked_features,
                                 cfs=cfs, X=X, init_points=init_points, n_iter=n_iter,
                                 sampling_from_model=sampling_from_model, optimizer_type=optimizer_type,
                                 optimizer=optimizer, bayesian_causality=True, bins_num=categorical_bins_num, bayesian_model=bayesian_model)
        cfs.append(cf)

    return np.vstack(cfs)

def generate_cfs(query_instance:np.ndarray, desired_class:int, adjacency_matrix:np.ndarray, causal_order : Sequence[int], proximity_weight : float,
                 sparsity_weight: float, plausibility_weight: float, diversity_weight: float, bounds:Dict[str, Tuple[float, float]],
                 model:ClassifierMixin, features_order:Optional[List[str]] =None,
                 masked_features:Optional[List[str]] =None, categorical_indicator:Optional[List[bool]] =None, X:Optional[pd.DataFrame] =None,
                 num_cfs:int=1, init_points:int=10, n_iter:int=1000, sampling_from_model:bool=False,
                 optimizer_type : OptimizerType = OptimizerType.EBM, optimizer : ModelBasedCounterOptimizer=None) -> np.ndarray:
    """
    Generate counterfactuals using an adjacency-based causal representation.

    This function generates ``num_cfs`` counterfactual instances by repeatedly
    optimizing a weighted loss function. A separate Optuna study is created
    for each counterfactual.

    Candidate instances are evaluated by ``compute_loss`` using the supplied
    predictive model, query instance, feature metadata, causal order, and
    adjacency matrix.

    Counterfactuals generated earlier in the same call are passed to later
    optimization runs. This allows the diversity component of the loss to
    account for the counterfactuals already generated.

    Parameters
    ----------
    query_instance : numpy.ndarray
        Instance for which counterfactual explanations are generated. The
        implementation supports an instance represented as a one-dimensional
        array or an array with shape ``(1, n_features)``.

        Features not included in ``masked_features`` are restored to their
        values from this instance before candidate evaluation and before the
        final counterfactual is returned.

    desired_class : int
        Target class for the generated counterfactuals. It is passed to the
        loss calculation and used to select initial samples from ``X``.

    adjacency_matrix : numpy.ndarray
        Adjacency matrix passed to ``compute_loss`` as the representation of
        the causal structure.

        Structural validation of this matrix is performed by
        ``ACFX.counterfactual`` when plausibility loss is enabled and this
        public function is called through the ``ACFX`` API. This function
        itself does not validate the matrix dimensions.

    causal_order : Sequence[int]
        Order of variables in the causal graph. It is passed to
        ``compute_loss`` together with ``adjacency_matrix``.

    proximity_weight : float
        Weight of the proximity component passed to ``compute_loss``.

    sparsity_weight : float
        Weight of the sparsity component passed to ``compute_loss``.

    plausibility_weight : float
        Weight of the plausibility component passed to ``compute_loss``.

    diversity_weight : float
        Weight of the diversity component passed to ``compute_loss``.
        Previously generated counterfactuals are supplied to the loss
        calculation when available.

    bounds : dict[str, tuple[float, float]]
        Search bounds for individual features. Each key is a feature name and
        each value is a ``(minimum, maximum)`` tuple.

        Categorical bounds are converted to integers and used to construct an
        inclusive sequence of category values. Non-categorical values are
        sampled as floating-point values within their bounds.

        Model-generated initial candidates are clipped to these bounds before
        being added to the Optuna study.

    model : ClassifierMixin
        Predictive classifier used to evaluate counterfactual candidates.

        The implementation calls ``model.predict(X)`` to select reference
        observations assigned to ``desired_class``. Further model operations
        are performed by ``compute_loss`` and may also be performed by the
        selected initial-sampling optimizer.

    features_order : list[str] or None
        Feature order used to align the query instance, bounds, feature
        metadata, candidate arrays, and columns of ``X``.

        If ``None`` or empty, ``list(bounds.keys())`` is used.

    masked_features : list[str] or None
        Names of features that may be changed during optimization.

        Features omitted from this list are fixed to their corresponding
        values in ``query_instance``. If ``None`` or empty, all features in
        ``features_order`` may be changed.

    categorical_indicator : list[bool] or None
        Boolean indicators specifying which features should be treated as
        categorical. Their order must correspond to ``features_order``.

        Categorical candidate values are selected from the inclusive integer
        range given by the corresponding feature bounds. They are rounded and
        converted to ``int`` when the candidate array is constructed.

        If ``None`` or empty, all features are treated as non-categorical.

    X : pandas.DataFrame of shape (n_samples, n_features) or None
        Reference dataset used for initial candidate sampling.

        The implementation selects rows for which
        ``model.predict(X) == desired_class`` and removes duplicates before
        sampling initial points.

        This argument is required by the implementation. A ``ValueError`` is
        raised if it is ``None``.

    num_cfs : int
        Number of counterfactual instances to generate. The default is 1.

        Counterfactuals are generated sequentially and passed to subsequent
        runs through the diversity-loss input.

    init_points : int
        Requested number of initial points used to seed each Optuna study.
        The default is 10.

        Up to ``int(init_points * 0.5)`` target-class observations are sampled
        from ``X``. The number is limited by the amount of available data.

        If fewer points are obtained and ``sampling_from_model`` is True,
        model-based candidates are requested from the selected optimizer.

    n_iter : int
        Number of optimization trials requested in addition to the number of
        successfully enqueued initial candidates. The default is 1000.

    sampling_from_model : bool
        Whether model-based initial sampling is used when data-based sampling
        does not provide the requested number of initial points.

    optimizer_type : OptimizerType
        Optimizer type used for model-based initial candidate generation.

        Supported branches in the implementation are:

        * ``OptimizerType.EBM``;
        * ``OptimizerType.LinearAdditive``;
        * ``OptimizerType.Custom``.

    optimizer : ModelBasedCounterOptimizer or None
        Custom optimizer used when ``optimizer_type`` is
        ``OptimizerType.Custom``.

    Returns
    -------
    numpy.ndarray of shape (num_cfs, n_features)
        Generated counterfactual instances, vertically stacked in generation
        order.

    Raises
    ------
    ValueError
        If ``X`` is ``None``.

    ValueError
        If a model-based optimizer returns a candidate in a format other than
        ``dict``, ``numpy.ndarray``, or ``list``.

    NotImplementedError
        If ``optimizer_type`` is not handled by the optimizer-selection
        implementation.

    Notes
    -----
    Each counterfactual uses a separate in-memory Optuna study configured with
    ``direction="maximize"``.

    The Optuna objective returns ``-loss[0, 1]`` from the array produced by
    ``compute_loss``. This scalar is maximized by the study.

    Each candidate contains one value for every feature in
    ``features_order``. Values for features outside ``masked_features`` are
    replaced by their corresponding values from ``query_instance``.

    Exact duplicate initial candidates are not enqueued more than once within
    one counterfactual-generation run. Initial candidate values must also lie
    within the supplied bounds.

    No random seed is set internally. Data sampling, random selection of
    feature subsets, and optimization may therefore produce different results
    across calls.

    The function does not independently verify that the returned instances
    are predicted as ``desired_class``. It returns the parameters from the
    best Optuna trial according to the supplied loss function.
    """
    cfs = []
    for _ in range(num_cfs):
        cf = _generate_single_cf(query_instance, desired_class, adjacency_matrix,
                                 causal_order, proximity_weight, sparsity_weight,
                                 plausibility_weight, diversity_weight,
                                 bounds, model, categorical_indicator, features_order, masked_features=masked_features,
                                 cfs=cfs, X=X, init_points=init_points, n_iter=n_iter,
                                 sampling_from_model=sampling_from_model, optimizer_type=optimizer_type,
                                 optimizer=optimizer, bayesian_causality=False)
        cfs.append(cf)

    return np.vstack(cfs)

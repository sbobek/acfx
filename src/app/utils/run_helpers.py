import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from acfx import ACFX, AcfxCustom, AcfxEBM, AcfxLinear, RandomSearchCounterOptimizer
from utils.key_helper import get_pbounds_key

MASKED_SUFFIX = "_is_masked"


def categorical_feature_names(feature_types):
    return feature_types.query("Type in ['nominal', 'ordinal']")["Column Name"].tolist()


def continuous_slider_range(low, high):
    """The bounds of a continuous feature may be widened up to 3 data ranges on each side."""
    return float(low - 3 * abs(high - low)), float(high + 3 * abs(high - low))


def pbounds_from_state(state, feature_names):
    return {key: (float(state[get_pbounds_key(key)][0]), float(state[get_pbounds_key(key)][1]))
            for key in feature_names}


def masked_features(state):
    """Features whose '<name>_is_masked' checkbox is on (the '_<name>_is_masked' widget mirrors are skipped)."""
    return [key[:-len(MASKED_SUFFIX)] for key in state.keys()
            if key.endswith(MASKED_SUFFIX) and not key.startswith("_") and state[key] == True]


def acfx_for_classifier(classifier_instance) -> ACFX:
    if isinstance(classifier_instance, LogisticRegression):
        return AcfxLinear(classifier_instance)
    if isinstance(classifier_instance, ExplainableBoostingClassifier):
        return AcfxEBM(classifier_instance)
    if isinstance(classifier_instance, (RandomForestClassifier, KNeighborsClassifier)):
        return AcfxCustom(classifier_instance)
    raise ValueError("classifier_instance out of range")


def causal_order_to_indices(causal_order, all_columns):
    if all(isinstance(item, str) for item in causal_order):
        return [all_columns.index(column) for column in causal_order]
    return causal_order


def fit_explainer(explainer: ACFX, X, pbounds, features_order, masked, categorical_indicator,
                  adjacency_matrix=None, causal_order=None, bayesian_model=None, num_bins=None) -> ACFX:
    """
    Fits the explainer with either the LiNGAM graph (adjacency_matrix + causal_order), the Bayesian network,
    or no causal model. AcfxCustom gets a RandomSearchCounterOptimizer over the same bounds.
    """
    kwargs = dict(X=X, pbounds=pbounds, masked_features=masked, categorical_indicator=categorical_indicator,
                  features_order=features_order, bayesian_causality=bayesian_model is not None)
    if bayesian_model is not None:
        kwargs.update(num_bins=num_bins, bayesian_model=bayesian_model)
    else:
        kwargs.update(adjacency_matrix=adjacency_matrix, causal_order=causal_order)
    if isinstance(explainer, AcfxCustom):
        kwargs['optimizer'] = RandomSearchCounterOptimizer(explainer.blackbox, X, pbounds, 1000)
    return explainer.fit(**kwargs)


def target_class_list(data, classes):
    """Human-readable class names when the dataset provides them, otherwise the classifier classes."""
    if 'target_names' in data:
        assert len(classes) == len(data.target_names)
        return data.target_names
    return classes


def prediction_label(pred, classes, class_names) -> str:
    classes = list(classes)
    if pred in classes:
        return str(class_names[classes.index(pred)])
    return str(pred)


def numeric_desired_class(desired_class, class_names):
    if isinstance(desired_class, str):
        return np.argwhere(class_names == desired_class)[0][0]
    return desired_class


def query_template(all_columns, categorical_indicator) -> pd.DataFrame:
    """A single all-zero query row: int for categorical features, float for continuous ones."""
    df_init = pd.DataFrame([0] * len(all_columns), index=all_columns).T
    for (col_index, col_name) in enumerate(all_columns):
        if categorical_indicator[col_index]:
            df_init[col_name] = df_init[col_name].astype(int)
        else:
            df_init[col_name] = df_init[col_name].astype(float)
    return df_init


def counterfactuals_table(query_instance: pd.Series, cfs, columns) -> pd.DataFrame:
    cfs = pd.DataFrame(cfs, columns=list(columns))
    cfs.index = [f"Counterfactual {i+1}" for i in range(len(cfs))]
    query_instance_with_index = pd.DataFrame([query_instance])
    query_instance_with_index.index = ["Query Instance"]
    return pd.concat([query_instance_with_index, cfs])


def class_probabilities_text(classes, probs, class_names) -> str:
    classes = list(classes)
    return " | ".join(f"**{class_names[classes.index(cls_val)]}**: {prob:.1%}" for cls_val, prob in zip(classes, probs))

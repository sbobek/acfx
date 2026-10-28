from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

CLASSIFIER_LIST = [
    "LogisticRegression (scikit learn)",
    "RandomForestClassifier (scikit learn)",
    "KNeighborsClassifier (scikit learn)",
    "ExplainableBoostingClassifier (glassbox)"
]


def select_features(feature_types, X):
    """Columns marked as available for evaluation (is_on) and their types, in feature_types order."""
    selected_feature_labels = [item["Column Name"] for _, item in feature_types.iterrows()
                               if item["is_on"] and item["Column Name"] in X.columns]
    selected_feature_types = [item["Type"] for _, item in feature_types.iterrows()
                              if item["is_on"] and item["Column Name"] in selected_feature_labels]
    return X[selected_feature_labels], selected_feature_types


def available_classifiers(n_categorical_features):
    """Only EBM handles nominal/ordinal features natively."""
    if n_categorical_features > 0:
        return [CLASSIFIER_LIST[-1]]
    return CLASSIFIER_LIST


def build_classifier(classifier_name, params, selected_feature_types):
    if classifier_name == CLASSIFIER_LIST[0]:
        return LogisticRegression(**params)
    if classifier_name == CLASSIFIER_LIST[1]:
        return RandomForestClassifier(**params)
    if classifier_name == CLASSIFIER_LIST[2]:
        return KNeighborsClassifier(**params)
    if classifier_name == CLASSIFIER_LIST[3]:
        return ExplainableBoostingClassifier(feature_types=selected_feature_types, **params)
    raise IndexError("classifier_name out of range")


def build_logistic_regression_params(tol, fit_intercept, max_iter):
    """The data is not scaled (counterfactuals stay in original units), so convergence may need a high max_iter."""
    return {
        'tol': tol,
        'fit_intercept': fit_intercept,
        'max_iter': max_iter
    }


def build_random_forest_params(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf,
                               min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease):
    """Maps widget values to RandomForestClassifier kwargs: 'None'/0 mean None, values >= 1 are sample counts."""
    if max_features == 'None':
        max_features = None
    if max_depth == 0:
        max_depth = None
    if max_leaf_nodes == 0:
        max_leaf_nodes = None
    if min_samples_leaf >= 1:
        min_samples_leaf = int(min_samples_leaf)
    if min_samples_split >= 1:
        min_samples_split = int(min_samples_split)
    return {
        'n_estimators': n_estimators,
        'criterion': criterion,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'min_weight_fraction_leaf': min_weight_fraction_leaf,
        'max_features': max_features,
        'max_leaf_nodes': max_leaf_nodes,
        'min_impurity_decrease': min_impurity_decrease,
    }


def build_knn_params(n_neighbors, weights, algorithm, leaf_size, p):
    if weights == 'None':
        weights = None
    return {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'algorithm': algorithm,
        'leaf_size': leaf_size,
        'p': p
    }


def parse_interactions(interactions_instance):
    """
    Parses the EBM 'interactions' text input: an int (number of pairs), a float (fraction of features)
    or '<float>x' (multiple of the number of features), passed to EBM as is. Invalid input means 0.
    """
    if interactions_instance is None:
        return 0
    if interactions_instance.isdigit():
        return int(interactions_instance)
    if interactions_instance.endswith("x"):
        try:
            float(interactions_instance[:-1])
            return interactions_instance
        except ValueError:
            return 0
    try:
        return float(interactions_instance)
    except ValueError:
        return 0


def build_ebm_params(random_state, max_bins, max_interaction_bins, interactions, validation_size, outer_bags,
                     inner_bags, learning_rate):
    if random_state == 0:
        random_state = None
    return {
        'random_state': random_state,
        'max_bins': max_bins,
        'max_interaction_bins': max_interaction_bins,
        'interactions': parse_interactions(interactions),
        'validation_size': validation_size,
        'outer_bags': outer_bags,
        'inner_bags': inner_bags,
        'learning_rate': learning_rate,
    }

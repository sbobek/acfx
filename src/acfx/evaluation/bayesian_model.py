import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BIC, MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork


def discretize_dataframe(df: pd.DataFrame, categorical_flags: list[bool], bins: int = 5) -> pd.DataFrame:
    """
    Discretize continuous features in a DataFrame based on categorical flags.

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_flags : list[bool]
        List indicating whether each column is categorical (True) or continuous (False).
    bins : int
        Number of bins for discretization (default = 5).

    Returns:
    -------
    pd.DataFrame
        DataFrame with continuous features discretized into bins.
    """
    if len(categorical_flags) != df.shape[1]:
        raise ValueError("Length of categorical_flags must match number of columns in df.")

    result_df = df.copy()

    for col, is_cat in zip(df.columns, categorical_flags):
        if not is_cat:  # Continuous feature
            # Use pandas.cut to discretize into equal-width bins
            result_df[col] = pd.cut(df[col], bins=bins, labels=False)

    return result_df


import numpy as np

def discretize_ndarray(data: np.ndarray, categorical_flags: list[bool], bins: int = 5) -> np.ndarray:
    """
    Discretize continuous features in a NumPy ndarray based on categorical flags.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array of shape (n_samples, n_features).
    categorical_flags : list[bool]
        List indicating whether each column is categorical (True) or continuous (False).
    bins : int
        Number of bins for discretization (default = 5).

    Returns
    -------
    np.ndarray
        Array with continuous features discretized into integer bins.
    """
    if len(categorical_flags) != data.shape[1]:
        raise ValueError("Length of categorical_flags must match number of columns in data.")

    result = data.copy().astype(object)  # Use object type to allow mixed types if needed

    for col_idx, is_cat in enumerate(categorical_flags):
        if not is_cat:  # Continuous feature
            col = data[:, col_idx]
            # Compute bin edges and assign bin indices
            bin_edges = np.linspace(np.min(col), np.max(col), bins + 1)
            result[:, col_idx] = np.digitize(col, bin_edges[1:-1], right=True)

    return result


def train_bayesian_model(df: pd.DataFrame, categorical_flags: list[bool], bins: int = 5) -> DiscreteBayesianNetwork:
    def fit_dag_structure(discretized_df):
        hc = HillClimbSearch(discretized_df)
        best_model = hc.estimate(scoring_method=BIC(discretized_df))
        best_model.fit(discretized_df, estimator=MaximumLikelihoodEstimator)
        return list(best_model.nodes()), list(best_model.edges())

    discretized_df = discretize_dataframe(df, categorical_flags, bins)
    nodes, edges = fit_dag_structure(discretized_df)
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(edges)
    model.fit(discretized_df, estimator=BayesianEstimator)
    return model
import numpy as np
import pandas as pd

def generate_sample_from_csf(adjacency_matrix, csf, sample_order=None, categorical=None):
    num_vars = adjacency_matrix.shape[0]
    samples = np.zeros((1, num_vars))

    if categorical is None:
        categorical = [False] * num_vars  # Default to all variables being non-categorical

    if sample_order is None:
        sample_order = range(num_vars)

    # Assign noise values to variables without parents

    for j in range(num_vars):
        if np.all(adjacency_matrix[j, :] == 0):  # If no parents, assign noise
            samples[0, j] = csf[j]
            if categorical[j]:  # Round the value if the variable is categorical
                samples[0, j] = np.round(samples[0, j])

    # Generate samples based on the specified order
    for j in sample_order:
        parents = np.where(adjacency_matrix[j, :] != 0)[0]
        if len(parents) > 0:
            # Predicted values based on parents
            predicted_values = np.dot(samples[:, parents], adjacency_matrix[j, parents])
            samples[:, j] = predicted_values
            if categorical[j]:  # Round the value if the variable is categorical
                samples[:, j] = np.round(samples[:, j])

    return samples

def get_parents(adjacency_matrix):
    parents = []
    num_vars = adjacency_matrix.shape[0]
    for j in range(num_vars):
        if np.all(adjacency_matrix[j, :] == 0):
            parents.append(j)
    return parents


def generate_samples(adjacency_matrix, num_samples, bounds=None, sample_order=None, categorical=None):
    num_vars = adjacency_matrix.shape[0]
    samples = np.zeros((num_samples, num_vars))

    if categorical is None:
        categorical = [False] * num_vars  # Default to all variables being non-categorical

    if sample_order is None:
        sample_order = range(num_vars)

    # Assign noise values to variables without parents
    for i in range(num_samples):
        if bounds is None:
            noise = np.random.normal(size=num_vars)  # assign_csf
        else:
            noise = np.random.normal(size=num_vars)
        for j in range(num_vars):
            if np.all(adjacency_matrix[j, :] == 0):  # If no parents, assign noise
                samples[i, j] = noise[j]
                if categorical[j]:  # Round the value if the variable is categorical
                    samples[i, j] = np.round(samples[i, j])

    # Generate samples based on the specified order
    for j in sample_order:
        parents = np.where(adjacency_matrix[j, :] != 0)[0]
        if len(parents) > 0:
            # Predicted values based on parents
            predicted_values = np.dot(samples[:, parents], adjacency_matrix[j, parents])
            samples[:, j] = predicted_values
            if categorical[j]:  # Round the value if the variable is categorical
                samples[:, j] = np.round(samples[:, j])

    return samples



def print_cfs(qi, cfs, features):
    """
    Construct a DataFrame from a list of instances cfs,
    filling fields that are the same as in instance qi with '-'.

    Parameters:
    qi (dict): An instance represented as a dictionary.
    cfs (list of dicts): A list of instances, each represented as a dictionary.

    Returns:
    pd.DataFrame: Transformed DataFrame with matching fields replaced by '-'.
    """
    df = pd.DataFrame(cfs, columns=features)

    # Replace fields that are the same as in qi with '-'
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: '-' if x == qi[df.columns.get_loc(col)] else f'{qi[df.columns.get_loc(col)]}->{x}')

    return df
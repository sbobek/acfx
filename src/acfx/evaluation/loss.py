import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from scipy.spatial.distance import pdist, squareform

def compute_causal_penalty_bayesian_adjacency(samples, model : DiscreteBayesianNetwork):
    """
    Compute causal penalty for counterfactual samples based on Bayesian Network CPDs.

    Parameters:
    - samples: np.ndarray or pd.DataFrame (num_samples x num_features). All features must be categorical
    - model: fitted BayesianNetwork

    Returns:
    - penalty: float, average negative log-likelihood of samples under the model
    """
    penalty = 0.0
    n_samples = len(samples)
    nodes = list(model.nodes())
    for idx in range(n_samples):
        sample = samples[idx]
        sample_penalty = 0.0

        for node in nodes:
            parents = model.get_parents(node)
            if not parents:
                continue
            cpd = model.get_cpds(node)
            if not cpd:
                continue

            # Get parent values for this sample
            parent_values = [int(sample[nodes.index(p)]) for p in parents]
            node_value = sample[nodes.index(node)]

            # Find probability from CPD
            prob = cpd.get_value(**{node: int(node_value), **dict(zip(parents, parent_values))})
            if np.abs(prob) < 1e-8:
                prob = 1e-8
            sample_penalty += -np.log(prob)

        penalty += sample_penalty

    return penalty / n_samples

def compute_causal_penalty(samples, adjacency_matrix, sample_order, categorical=None, skip_categorical=True):
    """
    Calculate the inconsistency of samples with the given adjacency matrix.

    Parameters:
    - samples: np.ndarray, counterfactual samples to evaluate the causal penalty for
    - adjacency_matrix: np.ndarray, the adjacency matrix from DirectLiNGAM
    - samples: np.ndarray, the samples to evaluate (num_samples x num_features)
    - categorical: bool, categorical or continuous indicator
    - skip_categorical: bool, skip categorical indicator - when adjacency matrix is not categorical-feature effective.
    Even if the flag is true, we want the sample_order to be the same length as number of features in samples to keep the indexing
    Returns:
    - inconsistency: float, a measure of how inconsistent the samples are with the adjacency matrix
    """
    num_samples, num_features = samples.shape
    inconsistency = 0.0

    if categorical is None:
        categorical = [False] * len(sample_order)
    # Iterate through each feature and its causal parents
    included_to_loss = 0
    for i in sample_order:
        if skip_categorical and categorical[i]:
            continue
        included_to_loss += 1
        parents = np.where(adjacency_matrix[i, :] != 0)[0]
        if len(parents) > 0:
            # Predicted values based on parents
            predicted_values = np.dot(samples[:, parents], adjacency_matrix[i, parents])
            # Calculate the inconsistency as the mean squared error
            if categorical[i]:
                mse = (samples[:, i].round() != predicted_values.round()).astype(float)
            else:
                mse = (samples[:, i] - predicted_values) ** 2
            inconsistency += mse

    return np.sqrt(np.mean(inconsistency)) / max(1, included_to_loss)


def compute_yloss(model, cfs, desired_class):
    predicted_value = np.array(model.predict_proba(cfs))
    maxvalue = np.full((len(predicted_value)), -np.inf)
    for c in range(len(model.classes_)):
        if c != desired_class:
            maxvalue = np.maximum(maxvalue, predicted_value[:, c])
    yloss = np.maximum(0, maxvalue - predicted_value[:, int(desired_class)])
    return yloss,maxvalue


def compute_proximity_loss(cfs, query_instance, pbounds, features_order=None, feature_weights=None, categorical=None):
    """Compute weighted distance between two vectors."""
    # If feature_weights is None, assign an array of ones with the size of cfs.shape[1]

    if feature_weights is None:
        feature_weights = np.ones(cfs.shape[1])
    if categorical is None:
        categorical = [False] * cfs.shape[1]

    diff = abs(cfs - query_instance)
    diff = [1 if categorical[i] and v > 0 else v for i, v in enumerate(diff[0, :])]

    diff = [
        v / (pbounds[features_order[i]][1] - pbounds[features_order[i]][0]) if not categorical[i] and features_order[
            i] in pbounds.keys() else v for i, v in enumerate(diff)]
    product = np.multiply(
        (diff),
        feature_weights)
    product = product.reshape(-1, product.shape[-1])
    proximity_loss = np.sum(product, axis=1)

    # Dividing by the sum of feature weights to normalize proximity loss
    return proximity_loss / sum(feature_weights)

def compute_sparsity_loss(cfs, query_instance):
    """Compute weighted distance between two vectors."""
    sparsity_loss = np.count_nonzero(cfs - query_instance, axis=1)
    return sparsity_loss / cfs.shape[1]  # Dividing by the number of features to normalize sparsity loss


def compute_diversity_loss(cfs, low=1e-6, high=1e-5):
    # Compute pairwise distances
    pairwise_distances = pdist(cfs)

    # Convert the distances to a square matrix form
    # TODO: gower distance?
    distance_matrix = squareform(pairwise_distances)

    perturbations = np.random.uniform(low=low, high=high, size=distance_matrix.shape[0])

    # Add the random perturbations to the diagonal elements of the transformed matrix
    np.fill_diagonal(distance_matrix, distance_matrix.diagonal() + perturbations)

    transformed_matrix = 1 / (1 + distance_matrix)

    # Calculate the determinant of the transformed matrix
    determinant = np.linalg.det(transformed_matrix)
    return determinant

def _compute_loss_common(
    yloss, conditional_term,
    proximity_loss, sparsity_loss, plausibility_loss, diversity_loss,
    proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
    cfs
):
    """Helper to compute final loss array."""
    total_loss = yloss + conditional_term * (
        (diversity_loss * diversity_weight) +
        (proximity_weight * proximity_loss) +
        (sparsity_weight * sparsity_loss) +
        (plausibility_loss * plausibility_weight)
    )
    loss = np.reshape(np.array(total_loss), (-1, 1))
    index = np.reshape(np.arange(len(cfs)), (-1, 1))
    return np.concatenate([index, loss], axis=1)


def compute_loss_generic(
    model, cfs, query_instance, desired_class,
    proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
    pbounds, features_order, categorical=None, allcfs=None,
    plausibility_fn=None
):
    """Generic loss computation with injected plausibility function."""
    if allcfs is None:
        allcfs = []

    yloss, _ = compute_yloss(model, cfs, desired_class)
    conditional_term = 1.0 / (
        yloss + 1 + sum([proximity_weight, sparsity_weight, plausibility_weight, diversity_weight])
    )

    proximity_loss = compute_proximity_loss(
        cfs, query_instance, pbounds, features_order=features_order, categorical=categorical
    ) if proximity_weight > 0 else 0.0

    sparsity_loss = compute_sparsity_loss(cfs, query_instance) if sparsity_weight > 0 else 0.0

    plausibility_loss = plausibility_fn() if plausibility_weight > 0 else 0.0

    diversity_loss = 1 - np.abs(compute_diversity_loss(allcfs)) if len(allcfs) > 1 and diversity_weight > 0 else 0.0

    return _compute_loss_common(
        yloss, conditional_term,
        proximity_loss, sparsity_loss, plausibility_loss, diversity_loss,
        proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
        cfs
    )

def compute_loss_bayesian(model, cfs, cfs_categorized, query_instance, desired_class, bayesian_model : DiscreteBayesianNetwork,
                          proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
                          pbounds, features_order, masked_features, categorical=None, allcfs=None):
    return compute_loss_generic(
        model, cfs, query_instance, desired_class,
        proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
        pbounds, features_order, categorical, allcfs,
        plausibility_fn=lambda: compute_causal_penalty_bayesian_adjacency(cfs_categorized, bayesian_model)
    )

def compute_loss(model, cfs, query_instance, desired_class, adjency_matrix, causal_order,
                 proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
                 pbounds, features_order, masked_features, categorical=None, allcfs=None):
    return compute_loss_generic(
        model, cfs, query_instance, desired_class,
        proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
        pbounds, features_order, categorical, allcfs,
        plausibility_fn=lambda: compute_causal_penalty(cfs, adjency_matrix, causal_order, categorical=categorical)
    )


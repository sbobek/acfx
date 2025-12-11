import numpy as np
import pandas as pd

from ...evaluation.loss import compute_causal_penalty, compute_yloss, compute_proximity_loss, \
    compute_sparsity_loss, compute_diversity_loss


class CCStats:
    def __init__(self):
        self.stats = []

    @staticmethod
    def get_columns():
        return ['model_name', 'dataset', 'fidelity', 'probability', 'loss', 'proximity_loss', 'sparsity_loss',
                'causality_loss', 'diversity',
                'no_ov_cfs', 'ov_loss', 'ov_proximity_loss', 'ov_sparsity_loss', 'ov_causality_loss', 'ov_diversity'
            , 'execution_time']

    def get_stats_for_dataset(self, name):
        datasum = pd.DataFrame(self.stats, columns=self.get_columns())

        return datasum[datasum['dataset'] == name].groupby('model_name').mean().to_string(header=True)

    def get_total_stats(self):
        return pd.DataFrame(self.stats, columns=self.get_columns())

    def save_total_stats(self, filename):
        self.get_total_stats().to_csv(filename, index=False)

    def append_stats(self, method, cfs, dataset, explain_instance, model_clf, causal_model, desired_class, pbounds,
                     execution_time):
        """
        Append statistical evaluation metrics to the global `stats` list.

        This function evaluates various metrics for a given set of counterfactuals (cfs) and appends the results to a global list named `stats`. The evaluation metrics include accuracy, y-loss, proximity loss, sparsity loss, causal penalty, and diversity loss.

        Parameters:
        name (str): The name or identifier for the set of counterfactuals being evaluated.
        cfs (np.ndarray): An array of counterfactual instances to be evaluated.

        Metrics Evaluated:
        - Accuracy: The mean accuracy of the model's predictions on the counterfactuals.
        - y-Loss: The mean y-loss, which measures the difference between the model's predicted and desired tools.
        - Proximity Loss: The mean proximity loss, which measures the distance between the counterfactuals and the original instance being explained.
        - Sparsity Loss: The mean sparsity loss, which measures how many features differ between the counterfactuals and the original instance.
        - Causal Penalty: The mean causal penalty, which assesses the impact of the counterfactuals on a causal model.
        - Diversity: A measure of diversity among the counterfactuals.

        The results are appended as a list to the global `stats` list in the following order:
        [name, dataset,accuracy, mean_y_loss, mean_proximity_loss, mean_sparsity_loss, mean_causal_penalty, diversity,execution_time]

        Returns:
        List

        Example:
        >>> name = "Counterfactual Set 1"
        >>> cfs = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> append_stats(name, cfs)
        """
        categorical_indicator = dataset.categorical_indicator

        record = [method, dataset.name]
        if cfs is not None:
            record.append(np.mean(desired_class == model_clf.predict(cfs)))
            record.append(np.mean([p[desired_class] for p in model_clf.predict_proba(cfs)]))
            self._append_loss(causal_model, categorical_indicator, cfs, dataset, desired_class, explain_instance,
                              model_clf, pbounds, record)

            mask = (desired_class == model_clf.predict(cfs))
            ovcfs = [cfs[i] for i in range(len(cfs)) if mask[i]]

            if len(ovcfs) > 0:
                record.append(len(ovcfs))
                self._append_loss(causal_model, categorical_indicator, ovcfs, dataset, desired_class, explain_instance,
                                  model_clf, pbounds, record)
            else:
                record.append(0)
                record.extend([np.nan] * 5)

            record.append(execution_time)
        else:
            record.extend([np.nan] * 14)

        self.stats.append(record)

        return record

    @staticmethod
    def _append_loss(causal_model, categorical_indicator, cfs, dataset, desired_class, explain_instance, model_clf,
                     pbounds, record):
        record.append(np.mean([compute_yloss(model_clf, ce, desired_class)[0] for ce in cfs]))
        record.append(np.mean([compute_proximity_loss(ce.reshape(1, -1), explain_instance, pbounds=pbounds,
                                                      features_order=dataset.features,
                                                      categorical=categorical_indicator) for ce in cfs]))
        record.append(np.mean([compute_sparsity_loss(ce.reshape(1, -1), explain_instance) for ce in cfs]))
        record.append(np.mean([compute_causal_penalty(ce.reshape(1, -1), causal_model.adjacency_matrix_,
                                                      causal_model.causal_order_, categorical=categorical_indicator)
                               for ce in cfs]))

        cfs_as_np = np.array(cfs)
        # if cfs empty
        if not cfs_as_np.any():
            record.append(0)
        # 1D distance matrix: det(K) = 1
        elif len(cfs_as_np.shape) < 2:
            record.append(1)
        else:
            record.append(compute_diversity_loss(cfs))
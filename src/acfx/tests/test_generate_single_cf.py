from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.acfx.abstract.OptimizerType import OptimizerType
from src.acfx.evaluation.ccfs import _generate_single_cf


class TestGenerateSingleCF:
    @pytest.fixture
    def setup_teardown(self):
        query_instance = np.array([0.5, 1.5, 2.5])
        desired_class = 1
        adjacency_matrix = np.array([[0, 1], [1, 0]])
        causal_order = [0, 1]
        proximity_weight = 0.5
        sparsity_weight = 0.5
        plausibility_weight = 0.5
        diversity_weight = 0.5
        bounds = {'feature1': (0, 1), 'feature2': (1, 2), 'feature3': (2, 3)}
        model = Mock()
        model.predict.return_value = np.array([1, 0, 1, 0, 1])  # Mock prediction to match desired_class
        categorical_indicator = [False, False, False]
        features_order = ['feature1', 'feature2', 'feature3']
        masked_features = ['feature1', 'feature2']
        cfs = []
        init_points = 5
        n_iter = 1000
        optimizer_type = OptimizerType.EBM
        X = pd.DataFrame({
            'feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature2': [1.1, 1.2, 1.3, 1.4, 1.5],
            'feature3': [2.1, 2.2, 2.3, 2.4, 2.5]
        })


        yield (query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
               plausibility_weight, diversity_weight, bounds, model, categorical_indicator, features_order,
               masked_features, cfs, X, init_points, n_iter, optimizer_type)

    @patch('optuna.create_study')
    @patch('src.refactor.evaluation.casual_counterfactuals.compute_loss')
    @patch('src.refactor.model.ccfs.get_ebm_optimizer')
    def test_generate_single_cf_valid_input(self,mock_get_ebm_optimizer, mock_compute_loss, mock_create_study, setup_teardown):
        (query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
         plausibility_weight, diversity_weight, bounds, model, categorical_indicator, features_order,
         masked_features, cfs, X, init_points, n_iter, optimizer_type) = setup_teardown

        mock_study = Mock()
        mock_create_study.return_value = mock_study
        mock_study.best_params = {'feature1': 0.5, 'feature2': 1.5, 'feature3': 2.5}
        mock_compute_loss.return_value = np.array([[0, -1]])

        best_cf = _generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight,
                                      sparsity_weight, plausibility_weight, diversity_weight, bounds, model,
                                      categorical_indicator, features_order, masked_features, cfs, X, init_points,
                                      n_iter, optimizer_type)

        expected_cf = np.array([[0.5, 1.5, 2.5]])
        assert np.array_equal(best_cf, expected_cf)

    @patch('optuna.create_study')
    @patch('src.refactor.evaluation.casual_counterfactuals.compute_loss')
    @patch('src.refactor.model.ccfs.get_ebm_optimizer')
    def test_generate_single_cf_invalid_bounds(self,mock_get_ebm_optimizer, mock_compute_loss, mock_create_study, setup_teardown):
        (query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
         plausibility_weight, diversity_weight, bounds, model, categorical_indicator, features_order,
         masked_features, cfs, X, init_points, n_iter, optimizer_type) = setup_teardown

        bounds = {'feature1': (0, 1), 'feature2': (1, 2), 'feature4': (2, 3)}  # Invalid feature in bounds
        with pytest.raises(KeyError):
            _generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight,
                                sparsity_weight, plausibility_weight, diversity_weight, bounds, model,
                                categorical_indicator, features_order, masked_features, cfs, X, init_points,
                                n_iter, optimizer_type)

    @patch('optuna.create_study')
    @patch('src.refactor.evaluation.casual_counterfactuals.compute_loss')
    @patch('src.refactor.model.ccfs.get_ebm_optimizer')
    def test_generate_single_cf_invalid_optimizer_type(self,mock_get_ebm_optimizer, mock_compute_loss, mock_create_study, setup_teardown):
        (query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
         plausibility_weight, diversity_weight, bounds, model, categorical_indicator, features_order,
         masked_features, cfs, X, init_points, n_iter, optimizer_type) = setup_teardown

        optimizer_type = None
        with pytest.raises(NotImplementedError):
            _generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight,
                                sparsity_weight, plausibility_weight, diversity_weight, bounds, model,
                                categorical_indicator, features_order, masked_features, cfs, X, init_points,
                                n_iter, optimizer_type)

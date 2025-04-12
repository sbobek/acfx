import os

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from src.refactor.abstract.OptimizerType import OptimizerType
from src.refactor.model.bayesian_optimization import optimizer_iteration, generate_single_cf
from src.refactor.evaluation.multi_dataset_evaluation import DEFAULT_LOG_PATH

LOGFILEPATH = os.path.join(os.getcwd(), DEFAULT_LOG_PATH)

class TestOptimizerIteration:
    @pytest.fixture(scope='function')
    def setup_teardown_logfile(self):
        try:
            # setup
            with open(LOGFILEPATH, 'a'):
                pass
        except Exception as e:
            print(f"Error occurred while trying to create the file: {e}")
        yield
        # teardown
        try:
            if os.path.isfile(LOGFILEPATH):
                os.remove(LOGFILEPATH)
        except Exception as e:
            print(f"Error occurred while trying to remove the logfile: {e}")

    @patch('random.randint')
    @patch('random.sample')
    def test_optimizer_iteration_new_set(self,mock_sample, mock_randint):
        mock_randint.return_value = 2
        mock_sample.return_value = ['feature1', 'feature2']
        masked_features = ['feature1', 'feature2', 'feature3']
        total_lists = [['feature3']]
        optimizer = Mock()
        optimizer.optimize_proba.return_value = (None, 'counterfactual')
        desired_class = 'desired_class'

        result = optimizer_iteration(masked_features, total_lists, optimizer, desired_class)

        assert result == 'counterfactual'
        assert ['feature1', 'feature2'] in total_lists
        optimizer.optimize_proba.assert_called_once_with('desired_class', feature_masked=['feature1', 'feature2'])

    @patch('random.randint')
    @patch('random.sample')
    def test_optimizer_iteration_existing_set(self,mock_sample, mock_randint):
        mock_randint.return_value = 2
        mock_sample.return_value = ['feature1', 'feature2']
        masked_features = ['feature1', 'feature2', 'feature3']
        total_lists = [['feature1', 'feature2']]
        optimizer = Mock()
        desired_class = 'desired_class'

        result = optimizer_iteration(masked_features, total_lists, optimizer, desired_class)

        assert result is None
        assert len(total_lists) == 1
        optimizer.optimize_proba.assert_not_called()

    @patch('random.randint')
    @patch('random.sample')
    def test_optimizer_iteration_exception(self,mock_sample, mock_randint, setup_teardown_logfile):
        mock_randint.return_value = 2
        mock_sample.return_value = ['feature1', 'feature2']
        masked_features = ['feature1', 'feature2', 'feature3']
        total_lists = [['feature3']]
        optimizer = Mock()
        optimizer.optimize_proba.side_effect = Exception('Test exception')
        desired_class = 'desired_class'

        result = optimizer_iteration(masked_features, total_lists, optimizer, desired_class)
        assert result is None
        optimizer.optimize_proba.assert_called_once_with('desired_class', feature_masked=['feature1', 'feature2'])


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
    @patch('src.refactor.model.bayesian_optimization.get_ebm_optimizer')
    def test_generate_single_cf_valid_input(self,mock_get_ebm_optimizer, mock_compute_loss, mock_create_study, setup_teardown):
        (query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
         plausibility_weight, diversity_weight, bounds, model, categorical_indicator, features_order,
         masked_features, cfs, X, init_points, n_iter, optimizer_type) = setup_teardown

        mock_study = Mock()
        mock_create_study.return_value = mock_study
        mock_study.best_params = {'feature1': 0.5, 'feature2': 1.5, 'feature3': 2.5}
        mock_compute_loss.return_value = np.array([[0, -1]])

        best_cf = generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight,
                                     sparsity_weight, plausibility_weight, diversity_weight, bounds, model,
                                     categorical_indicator, features_order, masked_features, cfs, X, init_points,
                                     n_iter, optimizer_type)

        expected_cf = np.array([[0.5, 1.5, 2.5]])
        assert np.array_equal(best_cf, expected_cf)

    @patch('optuna.create_study')
    @patch('src.refactor.evaluation.casual_counterfactuals.compute_loss')
    @patch('src.refactor.model.bayesian_optimization.get_ebm_optimizer')
    def test_generate_single_cf_invalid_bounds(self,mock_get_ebm_optimizer, mock_compute_loss, mock_create_study, setup_teardown):
        (query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
         plausibility_weight, diversity_weight, bounds, model, categorical_indicator, features_order,
         masked_features, cfs, X, init_points, n_iter, optimizer_type) = setup_teardown

        bounds = {'feature1': (0, 1), 'feature2': (1, 2), 'feature4': (2, 3)}  # Invalid feature in bounds
        with pytest.raises(KeyError):
            generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight,
                               sparsity_weight, plausibility_weight, diversity_weight, bounds, model,
                               categorical_indicator, features_order, masked_features, cfs, X, init_points,
                               n_iter, optimizer_type)

    @patch('optuna.create_study')
    @patch('src.refactor.evaluation.casual_counterfactuals.compute_loss')
    @patch('src.refactor.model.bayesian_optimization.get_ebm_optimizer')
    def test_generate_single_cf_invalid_optimizer_type(self,mock_get_ebm_optimizer, mock_compute_loss, mock_create_study, setup_teardown):
        (query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight, sparsity_weight,
         plausibility_weight, diversity_weight, bounds, model, categorical_indicator, features_order,
         masked_features, cfs, X, init_points, n_iter, optimizer_type) = setup_teardown

        optimizer_type = None
        with pytest.raises(NotImplementedError):
            generate_single_cf(query_instance, desired_class, adjacency_matrix, causal_order, proximity_weight,
                               sparsity_weight, plausibility_weight, diversity_weight, bounds, model,
                               categorical_indicator, features_order, masked_features, cfs, X, init_points,
                               n_iter, optimizer_type)
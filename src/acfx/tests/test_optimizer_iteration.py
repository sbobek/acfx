import os
from unittest.mock import patch, Mock

import pytest

from src.acfx.evaluation.ccfs import optimizer_iteration
from src.acfx.tests.consts import LOGFILEPATH


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

import os

from src.refactor.evaluation.multi_dataset_evaluation import DEFAULT_LOG_PATH

LOGFILEPATH = os.path.join(os.getcwd(), DEFAULT_LOG_PATH)

from src.refactor.model.ccfs import generate_cfs
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

class TestGenerateCfs:
    @pytest.fixture
    def mock_inputs(self):
        query_instance = np.array([0.5, 1.0, 2.0])
        desired_class = 1
        adjacency_matrix = np.zeros((3, 3))
        causal_order = [0, 1, 2]
        proximity_weight = 1.0
        sparsity_weight = 1.0
        plausibility_weight = 1.0
        diversity_weight = 1.0
        bounds = {'f1': (0, 1), 'f2': (0, 2), 'f3': (1, 3)}
        model = MagicMock()
        features_order = ['f1', 'f2', 'f3']
        masked_features = ['f1', 'f2', 'f3']
        categorical_indicator = [False, False, False]
        X = None
        return (query_instance, desired_class, adjacency_matrix, causal_order,
                proximity_weight, sparsity_weight, plausibility_weight, diversity_weight,
                bounds, model, features_order, masked_features, categorical_indicator, X)

    @patch('src.refactor.model.ccfs.generate_single_cf')
    def test_generate_cfs_output_shape(self, mock_generate_single_cf, mock_inputs):
        mock_generate_single_cf.return_value = np.array([[0.6, 1.1, 2.1]])
        num_cfs = 5
        result = generate_cfs(*mock_inputs, num_cfs=num_cfs, init_points=2, n_iter=10)
        assert isinstance(result, np.ndarray)
        assert result.shape == (num_cfs, 3)

    @patch('src.refactor.model.ccfs.generate_single_cf')
    def test_generate_cfs_multiple_calls(self, mock_generate_single_cf, mock_inputs):
        mock_generate_single_cf.side_effect = [
            np.array([[0.6, 1.1, 2.1]]),
            np.array([[0.7, 1.2, 2.2]]),
            np.array([[0.8, 1.3, 2.3]])
        ]
        num_cfs = 3
        result = generate_cfs(*mock_inputs, num_cfs=num_cfs, init_points=2, n_iter=10)
        assert result.shape == (num_cfs, 3)
        assert np.allclose(result[0], [0.6, 1.1, 2.1])
        assert np.allclose(result[1], [0.7, 1.2, 2.2])
        assert np.allclose(result[2], [0.8, 1.3, 2.3])
        assert mock_generate_single_cf.call_count == num_cfs

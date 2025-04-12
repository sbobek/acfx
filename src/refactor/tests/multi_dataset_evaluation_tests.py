import os
import pickle

import pytest
from unittest.mock import Mock, patch, mock_open
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from src.refactor.evaluation.multi_dataset_evaluation import train_ebm_model, preprocess_ds, load_or_dump_cached_file


class DatasetMock:
    def __init__(self, df_train, df_test, target, features, feature_types):
        self.df_train = df_train
        self.df_test = df_test
        self.target = target
        self.features = features
        self.feature_types = feature_types

class TestPreprocessDs:
    @pytest.fixture
    def dataset_fixture(self):
        df_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'target': [0, 1, 0, 1]
        })
        df_test = pd.DataFrame({
            'feature1': [2, 3],
            'feature2': [6, 7],
            'target': [1, 0]
        })
        target = 'target'
        features = ['feature1', 'feature2']
        feature_types = ['continuous', 'continuous']
        ds = DatasetMock(df_train, df_test, target, features, feature_types)
        return ds

    def test_preprocess_ds_no_missing_classes(self,dataset_fixture):
        ds = dataset_fixture
        ds_df_train, ds_df_test = preprocess_ds(ds)

        assert ds_df_train.equals(ds.df_train)
        assert ds_df_test.equals(ds.df_test)


    def test_preprocess_ds_missing_classes_in_test(self,dataset_fixture):
        ds = dataset_fixture
        ds.df_test = ds.df_test[ds.df_test['target'] == 1]  # Remove class 0 from test set
        ds_df_train, ds_df_test = preprocess_ds(ds)

        assert set(ds_df_train['target']) == set(ds_df_test['target'])


    def test_preprocess_ds_missing_classes_in_train(self,dataset_fixture):
        ds = dataset_fixture
        ds.df_train = ds.df_train[ds.df_train['target'] == 1]  # Remove class 0 from train set
        ds_df_train, ds_df_test = preprocess_ds(ds)

        assert set(ds_df_train['target']) == set(ds_df_test['target'])


    @patch('interpret.glassbox.ExplainableBoostingClassifier.fit')
    @patch('interpret.glassbox.ExplainableBoostingClassifier.predict')
    def test_train_ebm_model(self,mock_predict, mock_fit, dataset_fixture):
        ds = dataset_fixture
        random_state = 42
        mock_predict.return_value = ds.df_test['target']  # Mock prediction to match the test target

        model = train_ebm_model(ds, random_state)

        mock_fit.assert_called_once()
        mock_predict.assert_called_once()
        assert isinstance(model, ExplainableBoostingClassifier)


class TestLoadOrDumpCachedFile:
    @pytest.fixture
    def cache_dir(self,tmp_path):
        return str(tmp_path)

    @pytest.fixture
    def setup_teardown_pickle(self,cache_dir):
        cached_file_name = 'test_cache.pkl'
        path_pickle = os.path.join(cache_dir, cached_file_name)
        with open(path_pickle, 'wb') as f:
            pickle.dump('value', f)
        yield cached_file_name
        if os.path.exists(path_pickle):
            os.remove(path_pickle)

    def test_load_or_dump_cached_file_dump_data(self,cache_dir, setup_teardown_pickle):
        cached_file_name = setup_teardown_pickle
        cached_data = {'key': 'value'}

        with patch('builtins.open', mock_open()) as mocked_file:
            result = load_or_dump_cached_file(cache_dir, cached_file_name, cached_data)

            mocked_file.assert_called_once_with(os.path.join(cache_dir, cached_file_name), 'wb')
            mocked_file().write.assert_called_once()
            assert result == cached_data

    def test_load_or_dump_cached_file_load_data(self,cache_dir, setup_teardown_pickle):
        cached_file_name = setup_teardown_pickle
        cached_data = {'key': 'value'}

        with patch('builtins.open', mock_open(read_data=pickle.dumps(cached_data))) as mocked_file:
            result = load_or_dump_cached_file(cache_dir, cached_file_name)

            mocked_file.assert_called_once_with(os.path.join(cache_dir, cached_file_name), 'rb')
            assert result == cached_data

    def test_load_or_dump_cached_file_load_data_failure(self,cache_dir, setup_teardown_pickle):
        cached_file_name = setup_teardown_pickle

        with patch('builtins.open', mock_open(read_data='data')) as mocked_file:
            mocked_file.side_effect = pickle.UnpicklingError
            result = load_or_dump_cached_file(cache_dir, cached_file_name)

            mocked_file.assert_called_once_with(os.path.join(cache_dir, cached_file_name), 'rb')
            assert result is None

    def test_load_or_dump_cached_file_no_file(self,cache_dir):
        cached_file_name = 'non_existent_cache.pkl'

        result = load_or_dump_cached_file(cache_dir, cached_file_name)

        assert result is None

    def test_load_or_dump_cached_file_create_directory(self,cache_dir, setup_teardown_pickle):
        cached_file_name = 'test_cache.pkl'
        cached_data = {'key': 'value'}
        cache_dir = cache_dir + '_'
        with patch('os.makedirs') as mocked_makedirs, patch('builtins.open', mock_open()) as mocked_file:
            result = load_or_dump_cached_file(cache_dir, cached_file_name, cached_data)

            mocked_makedirs.assert_called_once_with(cache_dir)
            mocked_file.assert_called_once_with(os.path.join(cache_dir, cached_file_name), 'wb')
            assert result == cached_data

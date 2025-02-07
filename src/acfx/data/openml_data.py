import numpy as np
import pandas as pd

from .data import Data
from sklearn.model_selection import train_test_split


class OpenmlData(Data):
    def __init__(self, dataset, recipe=None):
        super().__init__(dataset, recipe)
        self.name = dataset.name

        data, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='array', target=dataset.default_target_attribute
        )
        
        df = pd.DataFrame(data, columns=attribute_names)
        df[dataset.default_target_attribute] = y
        self.target_class = dataset.default_target_attribute
        
        missing_percentages = df.isnull().mean() * 100
        columns_to_keep = missing_percentages[missing_percentages <= 75].index
        df_filtered = df[columns_to_keep]
        
        #fill-in missing values with means
        df_filtered = df_filtered.apply(lambda x: x.fillna(x.mean()), axis=0)

        features = [f for f in attribute_names if f in df_filtered.columns]
        
        self.features_cat = [f for i, f in enumerate(features) if categorical_indicator[i] and f in df_filtered.columns]
        self.features_num = [f for i, f in enumerate(features) if not categorical_indicator[i] and f in df_filtered.columns]

        self.feature_types = ['nominal' for _ in self.features_cat]
        self.feature_types += ['continuous' for _ in self.features_num]

        self._origin = df_filtered
        self._origin_features = [col for col in df_filtered.columns if col != self.target]
        self._features = self.features_cat + self.features_num
        self._categorical_indicator = [True for _ in self.categorical] + [False for _ in self.continuous]
        self._dataset = self.transform(df_filtered)
        self._dataset.loc[:, self.target] = self._origin[self.target]

        self._dataset_train, self._dataset_test = train_test_split(self._dataset, test_size=self.test_size, random_state=self.random_state)
        
    @property
    def features(self):
        return self._features
    
    @property
    def origin_features(self):
        return self._origin_features

    # List of all categorical features
    @property
    def categorical(self):
        return self.features_cat
    
    @property
    def categorical_indicator(self):
        return self._categorical_indicator

    # List of all continuous features
    @property
    def continuous(self):
        return self.features_num

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        return []

    # Feature name of the target column
    @property
    def target(self):
        return self.target_class

    # The full dataset
    @property
    def df(self):
        return self._dataset.copy()
    
    @property
    def df_origin(self):
        return self._origin

    # The training split of the dataset
    @property
    def df_train(self):
        return self._dataset_train.copy()

    # The test split of the dataset
    @property
    def df_test(self):
         return self._dataset_test.copy()

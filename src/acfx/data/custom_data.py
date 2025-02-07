import numpy as np
import pandas as pd

from .data import Data
from sklearn.model_selection import train_test_split


class CustomData(Data):
    def __init__(self, dataset, recipe=None):
        super().__init__(dataset, recipe)

        self.target_class = self.recipe.target
        
        df = dataset.copy()
        self._origin_features = self.recipe.features
        self.features_cat = self.recipe.features_cat
        self.features_num = self.recipe.features_num
        self.feature_types = self.recipe.features_types

        self._origin = df
        self._features = self._origin_features
        self._categorical_indicator = [col in self.features_cat for col in self._features]
        self._dataset = self.transform(df)
        self._dataset.loc[:, self.target] = df[self.target]

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

from abc import ABC, abstractmethod

import json
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

TEST_SIZE = .33
RANDOM_STATE = 42

class DataRecipe:
    def __init__(self, target=None, features=None, features_num=None, features_cat=None, features_types=None, name=None, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.target = target
        self.features = features
        self.features_num = features_num
        self.features_cat = features_cat
        self.features_types = features_types
        self.name = name
        self.test_size = test_size
        self.random_state = random_state
        
    @classmethod
    def from_json_file(cls, filepath):
        data = json.load(open(filepath, 'r'))
        return cls(
            data.get('target'),
            data.get('features'),
            data.get('features_num'),
            data.get('features_cat'),
            data.get('features_types'),
            data.get('name', filepath[filepath.rfind('/')+1:filepath.rfind('.')]),
            data.get('test_size', TEST_SIZE),
            data.get('random_state', RANDOM_STATE)
        )

class Data(ABC):
    def __init__(self, dataset, recipe=None):
        self.recipe = recipe if isinstance(recipe, DataRecipe) else DataRecipe()
        
        self.test_size = self.recipe.test_size
        self.random_state = self.recipe.random_state
        self.name = self.recipe.name

        self.scalers = {}
        self.encoders = {}
        
    def __repr__(self):
        return str(f'<Data> Dataset name: "{self.name}"; Shape: {self.df.shape}')

    @property
    @abstractmethod
    def features(self):
        """
        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass
    
    @property
    @abstractmethod
    def origin_features(self):
        """
        Returns
        -------
        list of Strings
            List of all categorical columns
        """
        pass
    
    @property
    @abstractmethod
    def categorical(self):
        """
        Provides the column names of categorical data.
        Column names do not contain encoded information as provided by a get_dummy() method (e.g., sex_female)

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all categorical columns
            
        """
        pass
    
    @property
    @abstractmethod
    def categorical_indicator(self):
        return self._categorical_indicator

    @property
    @abstractmethod
    def continuous(self):
        """
        Provides the column names of continuous data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all continuous columns
        """
        pass

    @property
    @abstractmethod
    def immutables(self):
        """
        Provides the column names of immutable data.

        Label name is not included.

        Returns
        -------
        list of Strings
            List of all immutable columns
        """
        pass

    @property
    @abstractmethod
    def target(self):
        """
        Provides the name of the label column.

        Returns
        -------
        str
            Target label name
        """
        pass
    
    @property
    @abstractmethod
    def df_origin(self):
        """
        The origin Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df(self):
        """
        The full Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_train(self):
        """
        The training split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def df_test(self):
        """
        The testing split Dataframe.

        Returns
        -------
        pd.DataFrame
        """
        pass
    
    def sample_test(self, n):
        for frac in self.df_test[self.features].sample(n, random_state=self.random_state).values:
            yield frac

    def initialize_scalers(self, df):
        for feature in self.continuous:
            if feature not in self.scalers:
                scaler = StandardScaler()
                scaler.fit(df[[feature]])
                self.scalers[feature] = scaler

    def initialize_encoders(self, df):
        for feature in self.categorical:
            if feature not in self.encoders:
                label_encoder = LabelEncoder()
                label_encoder.fit(df[feature])
                self.encoders[feature] = label_encoder

    def transform(self, df):
        transformed_df = df.copy()
        
        self.initialize_scalers(transformed_df)
        self.initialize_encoders(transformed_df)  # Use LabelEncoder instead of OneHotEncoder
        
        for feature in self.continuous:
            transformed_df[feature] = self.scalers[feature].transform(transformed_df[[feature]])
        
        for feature in self.categorical:
            transformed_df[feature] = self.encoders[feature].transform(transformed_df[feature])
        
        return transformed_df[self.features]

    def inverse_transform(self, df):
        original_df = df.copy()
        
        for feature in self.categorical:
            original_df[feature] = self.encoders[feature].inverse_transform(original_df[feature])
        
        for feature in self.continuous:
            original_df[feature] = self.scalers[feature].inverse_transform(original_df[[feature]])
            original_df[feature] = original_df[feature].round(10)
        
        return original_df[self.origin_features]

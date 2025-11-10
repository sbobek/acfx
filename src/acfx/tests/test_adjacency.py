import os
import warnings

from ...acfx.AcfxLinear import AcfxLinear
from ...acfx.evaluation.loss import compute_causal_penalty
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import lingam

def is_categorical(df: pd.DataFrame) -> list[bool]:
    result = []
    for col in df.columns:
        # Drop NaNs to avoid type confusion
        non_null = df[col].dropna()

        if non_null.empty:
            raise AttributeError(col)
        elif all(float(x).is_integer() for x in non_null if isinstance(x, (int, float))):
            result.append(True)
        elif all(isinstance(x, float) for x in non_null):
            result.append(False)
        else:
            raise AttributeError(col)
    return result

@pytest.fixture
def sample_data():
    csv = pd.read_csv(os.path.join(os.path.dirname(__file__), 'transformed_ready_for_acfx.csv'))
    csv.dropna(inplace=True)
    target = 'cat_value'
    X = csv.drop(target, axis=1)
    y = csv[target]
    return train_test_split(X, y, test_size=0.2, random_state=42), is_categorical(X)


def test_counterfactual_has_casual_penalty(sample_data):
    (X_train, X_test, y_train, y_test), categorical_indicator = sample_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    explainer = AcfxLinear(model)

    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(X_train)
    adjacency_matrix = causal_model.adjacency_matrix_
    causal_order = causal_model.causal_order_

    query_instance = X_test.iloc[0].values

    pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}
    features_order = X_train.columns.tolist()

    explainer.fit(X=X_test, adjacency_matrix=adjacency_matrix, casual_order=causal_order, pbounds=pbounds,
                  features_order=features_order, categorical_indicator=categorical_indicator)

    with warnings.catch_warnings(record=True) as w:
        compute_causal_penalty(query_instance.reshape(1,-1), adjacency_matrix,
                                                    sample_order=causal_order,
                                                    features_order=features_order, categorical=categorical_indicator)

        warnings_for_categorical_causality = [warn for warn in w if str(warn.message)
            .startswith('Categorical feature is skipped, but the adjacency indicates causability ')]
        assert len(warnings_for_categorical_causality) == 20


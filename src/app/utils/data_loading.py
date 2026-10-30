from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.utils import Bunch

BUILTIN_DATASETS = ['iris', 'wine', 'breast cancer', 'german credit']

GERMAN_CREDIT_TYPES = {
    "Status of existing checking account": "ordinal",
    "Duration": "continuous",
    "Credit history": "ordinal",
    "Purpose": "nominal",
    "Credit amount": "continuous",
    "Savings account/bonds": "ordinal",
    "Present employment since": "ordinal",
    "Installment rate in percentage of disposable income": "continuous",
    "Personal status and sex": "nominal",
    "Other debtors / guarantors": "nominal",
    "Present residence since": "continuous",
    "Property": "nominal",
    "Age": "continuous",
    "Other installment plans": "nominal",
    "Housing": "nominal",
    "Number of existing credits at this bank": "ordinal",
    "Job": "ordinal",
    "Number of people being liable to provide maintenance for": "ordinal",  # there are only two options: 1 and 2
    "Telephone": "nominal",
    "foreign worker": "nominal",
    "Credit granted": "nominal",
}


def is_categorical_like(series, max_unique_ratio=0.1, max_unique_absolute=50):
    def is_integer_like():
        return pd.api.types.is_float_dtype(series) and (series.dropna() % 1 == 0).all()

    if isinstance(series.dtype, pd.CategoricalDtype):
        return True
    # pandas>=3 stores text in the dedicated 'str' dtype instead of object
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) \
            or pd.api.types.is_integer_dtype(series) or is_integer_like():
        unique_count = series.nunique(dropna=True)
        total_count = len(series)
        return (unique_count / total_count <= max_unique_ratio) or (unique_count <= max_unique_absolute)
    return False


def infer_feature_types(data_instances, feature_names, predefined_types=None) -> pd.DataFrame:
    """
    Builds the 'Column Name' / 'Type' / 'is_on' table shown on the data selection page.
    Types from predefined_types win; other numeric columns are 'nominal' when categorical-like, else 'continuous'.
    Only continuous features are available for evaluation (is_on) by default.
    """
    predefined_types = predefined_types or {}
    feature_types = []
    for col in feature_names:
        if col in predefined_types:
            feature_types.append({'Column Name': col, 'Type': predefined_types[col],
                                  "is_on": bool(predefined_types[col] == 'continuous')})
        elif is_numeric_dtype(data_instances[col]):
            if is_categorical_like(data_instances[col]):
                feature_types.append({'Column Name': col, 'Type': 'nominal', "is_on": False})
            else:
                feature_types.append({'Column Name': col, 'Type': 'continuous', "is_on": True})
        else:
            raise ValueError(f"non-numerical feature found: {col}. For any kind of categorical feature, you need to preprocess first")
    return pd.DataFrame(feature_types)


def load_german_credit_data() -> Bunch:
    file_path = Path(__file__).parent.parent / 'german_credit_numeric.csv'

    if not file_path.exists():
        raise FileNotFoundError(f"german credit dataset not found: {file_path}")

    bunch_df = pd.read_csv(file_path)
    target_column = 'Credit granted'
    if target_column not in bunch_df.columns:
        raise ValueError(f"german credit csv dataset is invalid: '{target_column}' column missing. Columns available: {list(bunch_df.columns)}")

    bunch_X = bunch_df.drop(columns=[target_column])
    bunch_y = bunch_df[target_column]

    return Bunch(
        data=bunch_X,
        target=bunch_y,
        frame=bunch_df,
        feature_names=list(bunch_X.columns),
        target_names=np.array(['Not Granted', 'Granted'])
    )


def load_builtin_dataset(data_source_name) -> Bunch:
    if data_source_name == 'iris':
        return load_iris(as_frame=True)
    if data_source_name == 'wine':
        return load_wine(as_frame=True)
    if data_source_name == 'breast cancer':
        return load_breast_cancer(as_frame=True)
    if data_source_name == 'german credit':
        return load_german_credit_data()
    raise ValueError(f"data_source_name: {data_source_name} is out of range")


def builtin_feature_types(data_source_name, data: Bunch) -> pd.DataFrame:
    predefined_types = GERMAN_CREDIT_TYPES if data_source_name == 'german credit' else None
    return infer_feature_types(data.data, data.feature_names, predefined_types)


def split_features_and_label(data: pd.DataFrame, label_column):
    return data.drop(columns=[label_column]), data[label_column]

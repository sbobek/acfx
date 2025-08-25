import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from ACFX import ACFX
from AcfxLinear import AcfxLinear
from AcfxEBM import AcfxEBM
from src.acfx import AcfxCustom
from utils.key_helper import get_pbounds_key

def get_pbounds() -> dict[str,tuple[float,float]]:
    return {key: (st.session_state[get_pbounds_key(key)[0]], st.session_state[get_pbounds_key(key)[1]])
            for key in st.session_state.pbounds.keys()}

def get_masked_features() -> list[str]:
    suffix = "_is_masked"
    masked_features_keys = [key for key in st.session_state.keys()
                            if key.endswith(suffix) and st.session_state[key] == True]
    return [feature_masked[:-len(suffix)] for feature_masked in masked_features_keys]

def get_categorical_indicator() -> list[bool]:
    is_categorical = {row['Column Name']: row['Type'] != 'continuous'  for _, row in st.session_state.feature_types.iterrows()}
    assert st.session_state.data.columns == list(is_categorical.keys())
    return [is_categorical[feature_name] for feature_name in st.session_state.X.columns]

def get_acfx():
    classifier_instance = st.session_state.classifier_instance
    if isinstance(classifier_instance, LogisticRegression):
        acfx = AcfxLinear(classifier_instance)
    elif isinstance(classifier_instance, ExplainableBoostingClassifier):
        acfx = AcfxEBM(classifier_instance)
    elif (isinstance(classifier_instance, RandomForestClassifier)
          or isinstance(classifier_instance, KNeighborsClassifier)):
        acfx = AcfxCustom(classifier_instance)
    else:
        raise ValueError("classifier_instance out of range")
    return acfx

def fit_acfx(acfx:ACFX, query_instance) -> ACFX:
    adjacency_matrix = None
    casual_order = None
    if 'adjacency_matrix' in st.session_state and st.session_state.adjacency_matrix is not None:
        adjacency_matrix = st.session_state.adjacency_matrix
    if 'casual_order' in st.session_state and st.session_state.casual_order is not None:
        casual_order = st.session_state.casual_order

    if 'pbounds' not in st.session_state or not isinstance(st.session_state.pbounds, dict):
        raise TypeError("pbounds must be initialized in session state and be dict")
    if (casual_order is None and adjacency_matrix is not None) or (casual_order is not None and adjacency_matrix is not None):
        raise KeyError('casual_order and adjacency_matrix must be specified together')
    elif casual_order is not None and not isinstance(casual_order,list):
        raise TypeError('casual_order must be a list')
    elif adjacency_matrix is not None and not isinstance(adjacency_matrix,np.ndarray):
        raise TypeError('adjacency_matrix must be a numpy array')
    elif 'feature_types' not in st.session_state or st.session_state.feature_types is None:
        raise KeyError('feature_types must be initialized in session state here')

    return acfx.fit(X = st.session_state.selected_X, query_instance=query_instance, adjacency_matrix=adjacency_matrix,
             casual_order=casual_order,
             pbounds=get_pbounds(), y=st.session_state.y,
             masked_features=get_masked_features(),
             categorical_indicator=get_categorical_indicator(),
             features_order=st.session_state.casual_order)


    # def fit(self, X, query_instance: np.ndarray, adjacency_matrix:Optional[np.ndarray], casual_order:Optional[Sequence[int]],
    #         pbounds:Dict[str, Tuple[float, float]],y=None, masked_features:Optional[List[str]] = None,
    #         categorical_indicator:Optional[List[bool]] =None, features_order:Optional[List[str]] =None):

if 'proximity_weight' not in st.session_state \
    or 'diversity_weight' not in st.session_state \
    or 'sparsity_weight' not in st.session_state:
        st.warning("⚠️ Start by running 'Evaluation Settings'")
else:
    if 'classifier_name' not in st.session_state or st.session_state.classifier_name is None:
        raise ValueError('classifier_name must be in session state')
    if 'classifier_instance' not in st.session_state or st.session_state.classifier_instance is None:
        raise ValueError('classifier_instance must be in session state')
    query_instance = st.data_editor(
        st.session_state.df,
        num_rows=1,
        use_container_width=True
    )
    acfx = get_acfx()
    if st.checkbox(label="EVALUATE"):
        if query_instance is None:
            st.warning("⚠️ Query instance must be provided first")
        else:
            acfx = fit_acfx(acfx,query_instance)
            acfx.counterfactual(desired_class=?, )


            # def counterfactual(self, desired_class: int, num_counterfactuals: int = 1, proximity_weight: float = 1,
            #                    sparsity_weight: float = 1, plausibility_weight: float = 0, diversity_weight: float = 1,
            #                    init_points: int = 10,
            #                    n_iter: int = 1000, sampling_from_model: bool = True) -> np.ndarray:




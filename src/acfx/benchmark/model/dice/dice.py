from typing import Any, Dict

import dice_ml
import pandas as pd


class Dice:
    """
    Implementation of Dice from Mothilal et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to initialize.
        Please make sure to pass all values as dict with the following keys.

        * "num": int, default: 1
            Number of counterfactuals per factual to generate
        * "desired_class": int, default: 1
            Given a binary class label, the desired class a counterfactual should have (e.g., 0 or 1)
        * "posthoc_sparsity_param": float, default: 0.1
            Fraction of post-hoc preprocessing steps.
    - Restrictions:
        *   Only the model agnostic approach (backend: sklearn) is used in our implementation.
        *   ML model needs to have a transformation pipeline for normalization, encoding and feature order.
            See pipelining at carla/models/catalog/catalog.py for an example ML model class implementation

    .. [1] R. K. Mothilal, Amit Sharma, and Chenhao Tan. 2020. Explaining machine learning classifiers
            through diverse counterfactual explanations
    """

    _DEFAULT_HYPERPARAMS = {"num": 1, "desired_class": 1, "posthoc_sparsity_param": 0.1}

    def __init__(self, mlmodel, data, hyperparams=None) -> None:
        self._continuous = data.continuous
        self._categorical = data.categorical
        self._target = data.target
        self._model = mlmodel
        self._data = data
        
        if hyperparams is None:
            hyperparams = self._DEFAULT_HYPERPARAMS

        # Prepare data for dice data structure
        self._dice_data = dice_ml.Data(
            dataframe=data.df,
            continuous_features=self._continuous,
            outcome_name=self._target,
        )

        self._dice_model = dice_ml.Model(model=mlmodel, backend="sklearn")

        self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method="random")
        self._num = hyperparams["num"]
        self._desired_class = hyperparams["desired_class"]
        self._post_hoc_sparsity_param = hyperparams["posthoc_sparsity_param"]

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        # Prepare factuals
        querry_instances = factuals.copy()
        querry_instances = querry_instances[self._data.features]

        # check if querry_instances are not empty
        if not querry_instances.shape[0] > 0:
            raise ValueError("Factuals should not be empty")

        # Generate counterfactuals
        dice_exp = self._dice.generate_counterfactuals(
            querry_instances,
            total_CFs=self._num,
            desired_class=self._desired_class,
            posthoc_sparsity_param=self._post_hoc_sparsity_param,
        )

        list_cfs = dice_exp.cf_examples_list
        df_cfs = pd.concat([cf.final_cfs_df for cf in list_cfs], ignore_index=True)
        
#         df_cfs = e1.cf_examples_list[0].final_cfs_df[self._data.features].values # sbk
        
        return df_cfs[self._data.features]


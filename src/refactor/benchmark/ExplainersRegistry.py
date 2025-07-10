import traceback
# ignore error reported by static code analysis -- this package is loaded from local path
from cfnow import find_tabular
from time import time as tstime
from alibi.explainers import CounterfactualProto
from alibi.explainers import CEM
from alibi.explainers import Counterfactual
import pandas as pd
import numpy as np
import signal
from tensorflow.python.keras import backend as keras_backend
from src.refactor.model.ccfs import run_ccf
from src.refactor.evaluation.EBMCounterOptimizer import EBMCounterOptimizer


class ExplainersRegistry:
    def __init__(self, model_clf, causal_model, dataset, num_cfs, stats, time_limit=1800, n_iter=100, init_points=500):
        self.model_clf = model_clf
        self.causal_model = causal_model
        self.ds = dataset
        self.num_cfs = num_cfs
        self.stats = stats
        self.time_limit = time_limit
        self.n_iter = n_iter
        self.init_points = init_points

        self.pbounds = self.calc_pbounds()
        self.desired_class = None

        self.recipes = {}
        self.register_explainer('CCF', self._method_ccf)
        self.register_explainer('CCF-no-causal', self._method_ccf_no_causal)
        self.register_explainer('Baseline', self._method_baseline)
        self.register_explainer('cfnow', self._method_cfnow)

    def _method_baseline(self):
        ebcf = EBMCounterOptimizer(self.model_clf,
                                   pd.DataFrame(self.explain_instance.reshape(1, -1), columns=self.ds.features))
        cfs = []
        for i in range(self.num_cfs):
            proba, cf = ebcf.optimize_proba(self.desired_class, feature_masked=self.ds.features)
            cfs.append(cf)
        return pd.DataFrame(cfs).values

    def _method_ccf(self):
        return run_ccf(self.explain_instance, self.model_clf, self.ds,
                       self.desired_class, self.num_cfs, self.causal_model, self.pbounds,
                       init_points=self.init_points, n_iter=self.n_iter)

    def _method_ccf_no_causal(self):
        return run_ccf(self.explain_instance, self.model_clf, self.ds,
                       self.desired_class, self.num_cfs, self.causal_model, self.pbounds, as_causal=False,
                       init_points=self.init_points, n_iter=self.n_iter)

    def _method_cfnow(self):
        try:
            local_time_limit = int(self.cfgen_time_agg / self.cfgen_methods_no)
            if local_time_limit < 20:
                # make it reasonable limit for this kind of search algorithm
                local_time_limit = 20
        except:
            local_time_limit = 20

        cf_obj = find_tabular(
            factual=pd.Series(self.explain_instance),
            count_cf=self.num_cfs,
            feat_types={i: 'cat' if self.ds.categorical_indicator[i] else 'cont' for i in
                        range(len(self.ds.categorical_indicator))},
            model_predict_proba=self.model_clf.predict_proba,
            limit_seconds=local_time_limit)
        cfs = list(cf_obj.cfs[:self.num_cfs]) if cf_obj.total_cf >= self.num_cfs else list(cf_obj.cfs)
        return cfs

    def register_explainer(self, name, func):
        self.recipes.update({name: func})

    def calc_pbounds(self):
        return {f: (self.ds.df[f].min(), self.ds.df[f].max()) for f in self.ds.features}

    def set_desired_class(self, explain_instance, desired_class=None):
        if desired_class:
            self.desired_class = desired_class
        else:
            self.desired_class = (self.model_clf.predict(explain_instance)[0] + 1) % self.ds.df_train[
                self.ds.target].nunique()

    def run_explainers(self, explain_instance):
        self.set_desired_class(explain_instance)
        self.explain_instance = explain_instance
        self.cfgen_methods_no = 0
        self.cfgen_time_agg = 0
        keras_backend.clear_session()
        for method_name, method_func in self.recipes.items():
            print(f'Running {method_name}...')
            try:
                ts = tstime()
                signal.alarm(self.time_limit)
                cfs = method_func()
                te = tstime() - ts
                self.cfgen_methods_no += 1
                self.cfgen_time_agg += te
                self.stats.append_stats(method_name, cfs, self.ds, self.explain_instance, self.model_clf,
                                        self.causal_model, int(self.desired_class), self.pbounds, te)
                print('Done (in {:.2f}s)'.format(te))
            except TimeoutError:
                print('Timeout...')
                self.stats.append_stats(method_name, None, self.ds, self.explain_instance, self.model_clf,
                                        self.causal_model, int(self.desired_class), self.pbounds, np.nan)
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc()  # This prints the full traceback

                self.stats.append_stats(method_name, None, self.ds, self.explain_instance, self.model_clf,
                                        self.causal_model, int(self.desired_class), self.pbounds, np.nan)
            signal.alarm(0)
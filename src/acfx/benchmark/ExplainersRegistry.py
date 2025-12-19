import traceback

from alibi.explainers import CEM
from alibi.explainers.counterfactual import Counterfactual
from cfnow import find_tabular
from time import time as tstime
import pandas as pd
import numpy as np
import signal

from lux.lux import LUX
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import backend as keras_backend

from ..evaluation.multi_dataset_evaluation import DEFAULT_ERROR_LOG_PATH
from ..evaluation.ccfs import generate_cfs

from .model import Dice
from ..evaluation.EBMCounterOptimizer import EBMCounterOptimizer
from .model.LORE import lore
from .model.LORE.neighbor_generator import genetic_neighborhood
from .tools.utils import prepare_ds_lore, log2file
from alibi.explainers import CounterfactualProto

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
        self.register_explainer('Dice', self._method_dice)
        self.register_explainer('LUX', self._method_lux)
        self.register_explainer('LORE', self._method_lore)
        self.register_explainer('CEM', self._method_cem)

        self.register_explainer('CFProto', self._method_protocf)
        self.register_explainer('Wachter', self._method_wachter)
        self.register_explainer('cfnow', self._method_cfnow)

    def _method_baseline(self):
        ebcf = EBMCounterOptimizer(self.model_clf,
                                   pd.DataFrame(self.explain_instance.reshape(1, -1), columns=self.ds.features))
        cfs = []
        for i in range(self.num_cfs):
            cf = ebcf.optimize_proba(self.desired_class, feature_masked=self.ds.features)
            cfs.append(cf)
        return pd.DataFrame(cfs).values

    def _method_ccf(self):
        return run_ccf_causal(self.explain_instance, self.model_clf, self.ds,
                              self.desired_class, self.num_cfs, self.causal_model.adjacency_matrix_,
                              self.causal_model.causal_order_,self.pbounds, init_points=self.init_points, n_iter=self.n_iter)

    def _method_ccf_no_causal(self):
        return run_ccf_nocausal(self.explain_instance, self.model_clf, self.ds,
                              self.desired_class, self.num_cfs, self.causal_model.adjacency_matrix_,
                            self.causal_model.causal_order_, self.pbounds,
                              init_points=self.init_points, n_iter=self.n_iter)

    def _method_cem(self):
        Xtr = self.ds.df_train[self.ds.features].values
        mode = 'PN'  # 'PN' (pertinent negative) or 'PP' (pertinent positive)
        shape = (1,) + Xtr.shape[1:]  # instance shape
        kappa = .2  # minimum difference needed between the prediction probability for the perturbed instance on the
                    # class predicted by the original instance and the max probability on the other tools
                    # in order for the first loss term to be minimized
        beta = .1  # weight of the L1 loss term
        c_init = 10.  # initial weight c of the loss term encouraging to predict a different class (PN) or
                      # the same class (PP) for the perturbed instance compared to the original instance to be explained
        c_steps = 10  # nb of updates for c
        max_iterations = 1000  # nb of iterations per value of c
        feature_range = (Xtr.min(axis=0).reshape(shape)-.1,  # feature range for the perturbed instance
                         Xtr.max(axis=0).reshape(shape)+.1)  # can be either a float or array of shape (1xfeatures)
        clip = (-1000.,1000.)  # gradient clipping
        lr_init = 1e-2  # initial learning rate

        cem = CEM(self.model_clf.predict_proba,
          mode,
          shape,
          kappa=kappa,
          beta=beta,
          feature_range=feature_range,
          max_iterations=max_iterations,
          c_init=c_init,
          c_steps=c_steps,
          learning_rate_init=lr_init,
          clip=clip)

        cem.fit(Xtr, no_info_type='median')  # we need to define what feature values contain the least
                                                 # info wrt predictions
                                                 # here we will naively assume that the feature-wise median
                                                 # contains no info; domain knowledge helps!
        explanation = cem.explain(self.explain_instance.reshape(1,-1), verbose=False)
        if explanation['PN'] is not None:
            cfs = [explanation['PN'].ravel() for _ in range(self.num_cfs)]
        else:
            print('CEM did not return any CF')
            log2file('CEM did not return any CF')
            raise Exception('No CF found')
        return cfs

    def _method_protocf(self):
        Xtr = self.ds.df_train[self.ds.features].values
        shape = (1,) + Xtr.shape[1:]
        beta = .01
        c_init = 1.
        c_steps = 5
        max_iterations = 500
        rng = (-1., 1.)  # scale features between -1 and 1
        rng_shape = (1,) + Xtr.shape[1:]
        feature_range = (Xtr.min(axis=0).reshape(shape) - .1,  # feature range for the perturbed instance
                         Xtr.max(axis=0).reshape(shape) + .1)  # can be either a float or array of shape (1xfeatures)
        cat_vars_ord = {}
        for i, ci in enumerate(self.ds.categorical_indicator):
            if ci:
                cat_vars_ord[i] = len(np.unique(Xtr[:, i]))
        cf = CounterfactualProto(self.model_clf.predict_proba,
                                 shape,
                                 beta=beta,
                                 cat_vars=cat_vars_ord,
                                 max_iterations=max_iterations,
                                 feature_range=feature_range,
                                 c_init=c_init,
                                 c_steps=c_steps,
                                 eps=(.01, .01)  # perturbation size for numerical gradients
                                 )
        cf.fit(Xtr, d_type='abdm', disc_perc=[25, 50, 75])
        explanation = cf.explain(self.explain_instance.reshape(1, -1),
                                 target_class=[self.desired_class])
        if explanation['cf'] is not None:
            cfs = [explanation['cf']['X'].ravel()]
        else:
            print('Protocf did not return any CF')
            log2file('Protocf did not return any CF')
            raise Exception('No CF found')
        return cfs

    def _method_wachter(self):
        Xtr = self.ds.df_train[self.ds.features].values
        shape = (1,) + Xtr.shape[1:]
        beta = .01
        c_init = 1.
        c_steps = 5
        max_iterations = 500
        rng = (-1., 1.)  # scale features between -1 and 1
        rng_shape = (1,) + Xtr.shape[1:]
        feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32),
                         (np.ones(rng_shape) * rng[1]).astype(np.float32))

        cf = Counterfactual(self.model_clf.predict_proba, shape, distance_fn='l1', target_proba=1.0,
                            target_class=int(self.desired_class), max_iter=1000, early_stop=50, lam_init=1e-1,
                            max_lam_steps=10, tol=0.05, learning_rate_init=0.1,
                            feature_range=feature_range, eps=0.01, init='identity',
                            decay=True, write_dir=None, debug=False)

        explanation = cf.explain(self.explain_instance.reshape(1, -1))
        if explanation['cf'] is not None:
            cfs = [explanation['cf']['X'].ravel()]
        else:
            print('Wachter did not return results')
            log2file('Wachter did not return results')
            raise Exception('No CF found')
        return cfs

    def _method_lux(self):
        Xtr, _ = train_test_split(self.ds.df_train.copy(), train_size=min(self.ds.df_train.shape[0], 2000))
        cols = [f.replace('-', '') for f in Xtr.columns]
        Xtr.columns = cols

        features = [f for f in cols if f not in self.ds.target]
        explain_instance = self.explain_instance.reshape(1, -1)
        lux = LUX(predict_proba=self.model_clf.predict_proba,
                  # classifier=model_clf,
                  neighborhood_size=0.1)
        lux.fit(Xtr[features], Xtr[self.ds.target], instance_to_explain=explain_instance,
                categorical=self.ds._categorical_indicator, n_jobs=-1)

        _, Xs = train_test_split(Xtr[features], stratify=Xtr[self.ds.target], test_size=1000)

        cf_lux = lux.counterfactual(np.array(explain_instance), Xs[features], counterfactual_representative='nearest',
                                    topn=self.num_cfs)
        cfs = [np.array(c['counterfactual']) for c in cf_lux]

        return cfs

    def _method_dice(self):
        hyperparams = {"num": self.num_cfs, "desired_class": int(self.desired_class), "posthoc_sparsity_param": 0.1}
        dice = Dice(self.model_clf, data=self.ds, hyperparams=hyperparams)
        try:
            return dice.get_counterfactuals(pd.DataFrame([self.explain_instance], columns=self.ds.features)).values
        except Exception as e:
            print(f'DICE failed with exception: {e}')
            stack_trace = traceback.format_exc()
            log2file(f'DICE failed with exception: {e}. Stack trace: {stack_trace}', filepath=DEFAULT_ERROR_LOG_PATH)
            raise

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

    def _method_lore(self):
        Xtr = self.ds.df_train.copy()
        cols = [f.replace('-', '') for f in Xtr.columns]
        Xtr.columns = cols
        features = [f for f in cols if f not in self.ds.target]

        myds = prepare_ds_lore(Xtr, discrete=self.ds.categorical_indicator, class_name=self.ds.target_class)
        X_explain = np.concatenate((self.explain_instance.reshape(1, -1), myds['X']))
        try:
            exp_LORE, info_LORE = lore.explain(0, X_explain,
                                               myds, self.model_clf,
                                               ng_function=genetic_neighborhood,
                                               discrete_use_probabilities=True,
                                               continuous_function_estimation=False,
                                               returns_infos=True, path='acfx/benchmark/model/LORE/yadt/',
                                               sep=';', log=True, depth=100)
        except:
            try:
                # Try numerical only
                myds = prepare_ds_lore(Xtr, class_name=self.ds.target_class)
                X_explain = np.concatenate((self.explain_instance.reshape(1, -1), myds['X']))
                exp_LORE, info_LORE = lore.explain(0, X_explain,
                                                   myds, self.model_clf,
                                                   ng_function=genetic_neighborhood,
                                                   discrete_use_probabilities=True,
                                                   continuous_function_estimation=False,
                                                   returns_infos=True, path='acfx/benchmark/model/LORE/yadt/',
                                                   sep=';', log=True, depth=100)
            except Exception as e:
                print(f'LORE failed with exception: {e}')
                stack_trace = traceback.format_exc()
                log2file(f'LORE failed with exception: {e}. Stack trace: {stack_trace}', DEFAULT_ERROR_LOG_PATH)
                raise
        if len(exp_LORE) < 2 or len(exp_LORE[1]) < 1:
            print('LORE did not return results')
            log2file('LORE did not return results')
            raise Exception('No CF found')
        query = ' & '.join(
            [f'({str(a)} {str(b)})' if str(a) not in str(b) else f'({b})' for a, b in exp_LORE[1][0].items()])
        cfs = Xtr.query(query)
        fcfs = cfs.sample(min(len(cfs), self.num_cfs))
        return list(fcfs[features].values)

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


def run_ccf_causal(explain_instance, model_clf, dataset, desired_class, num_cfs, adjacency_matrix, causal_order,
                   pbounds, as_causal=True, masked_features=None, init_points=500, n_iter=100):
    return generate_cfs(explain_instance, desired_class=desired_class, adjacency_matrix=adjacency_matrix,
                        causal_order=causal_order,
                        proximity_weight=1,
                        sparsity_weight=1,
                        categorical_indicator = dataset._categorical_indicator,
                        plausibility_weight=int(as_causal),
                        diversity_weight = 1,
                        bounds=pbounds,
                        model=model_clf,
                        features_order = dataset.features,
                        masked_features = masked_features,
                        num_cfs=num_cfs,
                        X=dataset.df_train,
                        init_points=init_points,
                        n_iter=n_iter)

def run_ccf_nocausal(explain_instance, model_clf, dataset, desired_class, num_cfs, adjacency_matrix, causal_order,
            pbounds, masked_features=None, init_points=500, n_iter=100):
    return generate_cfs(explain_instance, desired_class=desired_class, adjacency_matrix=adjacency_matrix,
                        causal_order=causal_order,
                        proximity_weight=1,
                        sparsity_weight=1,
                        categorical_indicator = dataset._categorical_indicator,
                        plausibility_weight=0,
                        diversity_weight = 1,
                        bounds=pbounds,
                        model=model_clf,
                        features_order = dataset.features,
                        masked_features = masked_features,
                        num_cfs=num_cfs,
                        X=dataset.df_train,
                        init_points=init_points,
                        n_iter=n_iter)
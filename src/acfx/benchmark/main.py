import logging
import sys
import random
import numpy as np
import tensorflow as tf
import warnings
import signal
import openml
import traceback

from .data import OpenmlData
from .data.consts import RANDOM_STATE, TIME_LIMIT, MODEL_TIME_LIMIT
from src.acfx.evaluation.multi_dataset_evaluation import train_ebm_model, train_causal_model, \
    timeout_handler
from .tools.utils import log2file, load_or_dump_cached_file
from .ExplainersRegistry import ExplainersRegistry
from .tools.CCStats import CCStats
from tensorflow.python.keras import backend as keras_backend

def set_session():
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    keras_backend.set_session(sess)

def set_model_path():
    import os
    cwd = os.getcwd()
    sys.path.append(f'{cwd}/acfx/benchmark/model/LORE')
    sys.path.append(f'{cwd}/acfx/benchmark/model')

def evaluate(use_suite:bool, time_limit:int, model_time_limit:int):
    suite = openml.study.get_suite(99)
    print(suite)
    tasks = suite.tasks
    stats = CCStats()
    log2file('', clear=True)
    SAMPLE_SIZE = 10
    NUM_CFS = 3
    init_points = 200
    n_iter = int(0.2 * init_points)

    if not use_suite:
        all_datasets = openml.datasets.list_datasets(output_format='dataframe')
        classification_datasets = all_datasets[
            (all_datasets['NumberOfClasses'] > 1) & (all_datasets['NumberOfInstances'] > 2000) &
            (all_datasets['NumberOfMissingValues'] == 0) &
            # (all_datasets['NumberOfInstances']< 10000)&
            (all_datasets['NumberOfFeatures'] < 30)].drop_duplicates(subset=['name'])
        classification_datasets['name'] = classification_datasets['name'].apply(lambda x: x.split('_')[0])
        classification_datasets = classification_datasets.drop_duplicates(subset=['name'])

        tasks = classification_datasets['did']
    # OpenML datasets
    warnings.filterwarnings('ignore')
    cache_dir = 'cache/'
    for task_id in tasks:
        try:
            ds = load_or_dump_cached_file(cache_dir, f'dataset-{task_id}.pkl', cached_data=None)
            if ds is None:
                ds = OpenmlData(openml.datasets.get_dataset(task_id))
                load_or_dump_cached_file(cache_dir, f'dataset-{task_id}.pkl', cached_data=ds)

            log2file(f'Processing ID: {task_id} for name: {ds.name}\n')
            print(f'{"*" * 50} {ds.name} {"*" * 50}')

            print('Training classification model...', end='')
            model_clf = load_or_dump_cached_file(cache_dir, f'model_clf-{task_id}.pkl', cached_data=None)
            if model_clf is None:
                model_clf = train_ebm_model(ds, RANDOM_STATE, debug=True)
                load_or_dump_cached_file(cache_dir, f'model_clf-{task_id}.pkl', cached_data=model_clf)
            print('Done')

            try:
                signal.alarm(model_time_limit)  # one hour for causal model
                print('Training casual model...', end='')

                causal_model = load_or_dump_cached_file(cache_dir, f'causal_model-{task_id}.pkl', cached_data=None)
                if causal_model is None:
                    causal_model = train_causal_model(ds)
                    load_or_dump_cached_file(cache_dir, f'causal_model-{task_id}.pkl', cached_data=causal_model)
                print('Done.')
                signal.alarm(0)
            except TimeoutError:
                print('Timeout, aborting, moving to another dataset...')
                continue
            explainers_registry = ExplainersRegistry(model_clf, causal_model, ds, NUM_CFS, stats, time_limit,
                                                     init_points=init_points, n_iter=n_iter)
            print('Calculating counterfactuals...')
            for explain_instance in ds.sample_test(SAMPLE_SIZE):
                explainers_registry.run_explainers(explain_instance)
            log2file(stats.get_stats_for_dataset(ds.name))
        except Exception as e:
            # Capture the traceback as a string
            stack_trace = traceback.format_exc()
            log2file(stack_trace)
            print(stack_trace)


def set_seed():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

def init_logger():
    logger = logging.getLogger('openml')
    logger.handlers.clear()
    logger.propagate = False
    logging.getLogger('openml').setLevel(logging.ERROR)

if __name__ == '__main__':
    set_model_path()
    set_session()

    # Register the timeout handler for the SIGALRM signal
    signal.signal(signal.SIGALRM, timeout_handler)

    init_logger()
    warnings.filterwarnings("ignore")
    set_seed()

    evaluate(False, TIME_LIMIT, MODEL_TIME_LIMIT)

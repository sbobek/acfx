import os
import pickle
import lingam
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score


def timeout_handler(signum, frame):
    raise TimeoutError("Calculation timed out!")

def log2file(output, filepath='progress.txt', clear=False):
    with open(filepath, 'w' if clear else 'a') as f:
        f.write(output)

def train_ebm_model(ds, random_state, n_jobs=-1, interactions=0, debug=False):
    model_clf = ExplainableBoostingClassifier(random_state=random_state, n_jobs=-1, feature_types=ds.feature_types,
                                              interactions=0)
    model_clf.fit(ds.df_train[ds.features], ds.df_train[ds.target])

    y_pred = model_clf.predict(ds.df_test[ds.features])
    y_test = ds.df_test[ds.target]
    if debug:
        print(f"Model accuracy: {round(accuracy_score(y_test, y_pred), 2)}")

    return model_clf

def train_causal_model(dataset):
    causal_model = lingam.DirectLiNGAM()
    causal_model.fit(dataset.df_train[dataset.features])

    return causal_model

def load_or_dump_cached_file(cache_dir, cached_file_name, cached_data=None):
    """
    Load cached data from the specified file, or if cached_data is provided,
    dump it into the file in the specified directory.

    If the file does not exist or loading fails, and cached_data is not provided, return None.

    Parameters:
    cache_dir (str): Directory where the cache file is stored.
    cached_file_name (str): Name of the cache file.
    cached_data (object, optional): Data to be cached (default is None).

    Returns:
    object or None: The loaded cached data, or None if loading is not possible and cached_data is None.
    """
    # Ensure the directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Construct the full file path
    file_path = os.path.join(cache_dir, cached_file_name)

    # If cached_data is provided, dump it into the file
    if cached_data is not None:
        with open(file_path, 'wb') as file:
            pickle.dump(cached_data, file)
        return cached_data

    # Otherwise, try loading the cached data
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            return None
    else:
        return None

Release notes
=============
0.3.8 (2026-10-29)
------------------
Library

* Support for Python 3.12–3.14
* Fixed counterfactual generation for query instances given as 2D arrays of shape (1, n_features)
* Fixed EBM counter-optimizer for binary classification
* Initial points sampled from data are now taken only from instances predicted as the desired class
* Initial points from data and from the counter-optimizer are checked against the feature bounds
* Every Optuna study gets a unique name, so consecutive searches do not interfere
* Faster LogisticRegression counter-optimizer (vectorized, same results)
* Bayesian plausibility loss: feature values are rounded to the nearest state and states never observed in the data get the maximal penalty, instead of an error or a probability read from a wrong state

Streamlit app

* German credit built-in dataset with predefined feature types
* "Mark all as available for evaluation" button on the data selection page
* Interactive causal graph visualization (pyvis)
* PREDICT button showing the blackbox prediction with class probabilities
* ``max_iter`` setting for LogisticRegression (the data is not scaled, so convergence may need more iterations)
* Ordinal features are treated as categorical in the evaluation settings and in the query template
* DirectLiNGAM is the default adjacency generator only for purely continuous data, the discrete Bayesian network otherwise
* Fixed: changing RandomForest criterion or KNN weights/algorithm raised an error
* Fixed: EBM ``interactions`` given as ``"<float>x"`` was ignored
* Fixed: masked features list contained duplicated internal keys
* Fixed: selecting an ordinal feature type on the data selection page
* ``pyarrow<25`` pinned for Python<3.12 (newer versions do not work with numpy 1.26)

Testing and documentation

* Automated test suite (pytest) for the library and the Streamlit app (pages tested with Streamlit ``AppTest``)
* Code coverage measurement with a 90% threshold, described in ``TESTING.md``
* GitHub issue and pull request templates, extended contribution guide

0.3.6 (2026-03-19)
------------------
* Streamlit app
* Support for Python>3.12

0.1.0 (2025-02-14)
------------------
* Added pypl installation
* Added sphinx documentation
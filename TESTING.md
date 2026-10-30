# Testing ACFX

The repository contains an automated [pytest](https://docs.pytest.org/) test-suite for:

* **the ACFX library** (`src/acfx`): loss components, Bayesian-network plausibility, counter-optimizers,
  the counterfactual generation engine and the public explainer API (`AcfxLinear`, `AcfxEBM`, `AcfxCustom`),
* **the Streamlit GUI** (`src/app`): the data/classifier/causal-graph logic used by the pages, and the pages
  themselves, driven headlessly with Streamlit's [`AppTest`](https://docs.streamlit.io/develop/api-reference/app-testing).

The tests are deterministic (fixed seeds for NumPy, `random` and the Optuna sampler), so a failure is reproducible
and indicates a regression. Test names describe the expected behaviour, so the test files also document the code.

## 1. Set up the environment

Python 3.10 – 3.14 is supported. From the repository root:

```bash
cd src
python -m venv .venv && source .venv/bin/activate   # optional, any virtual environment works
pip install -e ".[test,streamlit-app]"
```

* `-e` (editable install) makes the tests and the coverage report use the code from this repository,
  not a copy installed from PyPI.
* `test` installs pytest, pytest-cov and the libraries used by the tests (lingam, networkx).
* `streamlit-app` installs the GUI dependencies required by the tests in `src/app/tests`.

## 2. Run the tests

All commands are run from the `src` directory (the configuration lives in `src/pyproject.toml`).

| Command | What it runs | 
|---|---|
| `pytest` | the whole suite | 
| `pytest -m "not slow"` | everything except end-to-end optimisation runs |
| `pytest acfx/tests` | only the library | 
| `pytest app/tests` | only the Streamlit GUI | 
| `pytest acfx/tests/test_loss.py -v` | a single file, listing every test name | 

Tests marked `slow` run the complete counterfactual search (README examples, Bayesian mode on a real dataset,
full GUI flow through all pages). They are part of the default run; use `-m "not slow"` for quick feedback
during development.

## 3. Measure code coverage

Coverage is measured with [coverage.py](https://coverage.readthedocs.io/) through the `pytest-cov` plugin:

```bash
pytest --cov                              # percentage per file in the terminal, with missing lines
pytest --cov --cov-report=html            # additionally writes htmlcov/index.html
pytest --cov --cov-report=xml             # coverage.xml, e.g. for CI services
```




# ACFX — Actionable Counterfactual eXplainer

[![PyPI version](https://img.shields.io/pypi/v/acfx.svg)](https://pypi.org/project/acfx/)
[![Python versions](https://img.shields.io/pypi/pyversions/acfx.svg)](https://pypi.org/project/acfx/)
[![License](https://img.shields.io/github/license/sbobek/acfx.svg)](https://github.com/sbobek/acfx/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/acfx/badge/?version=latest)](https://acfx.readthedocs.io/en/latest/)
[![GitHub Stars](https://img.shields.io/github/stars/sbobek/acfx.svg)](https://github.com/sbobek/acfx/stargazers)

**ACFX (Actionable Counterfactual eXplainer)** is a **model-agnostic Explainable AI (XAI)** framework for generating **actionable counterfactual explanations** for machine learning models.

It answers the question:

> *What minimal and feasible changes to the input would lead to a desired prediction outcome?*



## Key Features

* Model-agnostic counterfactual explanations
* Actionability and feasibility constraints
* Support for causal structures and expert knowledge
* Built-in benchmarking framework
* Python API and graphical user interface (GUI)



## Installation

### From PyPI

```bash
conda create --name acfx_env python=3.11
conda activate acfx_env
conda install pip
pip install acfx
```

### From source

```bash
git clone https://github.com/sbobek/acfx
cd acfx/src
pip install .
```



## Quick Start

```python
from acfx import AcfxEBM
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize model and explainer
model = ExplainableBoostingClassifier()
explainer = AcfxEBM(model)

# Load sample data
data = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define feature bounds
pbounds = {col: (X_train[col].min(), X_train[col].max()) for col in X_train.columns}

# Example adjacency matrix and causal order
adjacency_matrix = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.8, 0.0, 0.0, 0.0],
    [0.0, 0.6, 0.0, 0.0],
    [0.5, 0.0, 0.7, 0.0]
])
causal_order = [0, 1, 2, 3]

# Fit explainer
explainer.fit(
    X=X_train,
    adjacency_matrix=adjacency_matrix,
    causal_order=causal_order,
    pbounds=pbounds,
    y=y_train,
    features_order=X_train.columns.tolist()
)

# Generate counterfactual
query_instance = X_test.iloc[0].values
original_class = model.predict([query_instance])[0]
cf = explainer.counterfactual(desired_class=original_class, query_instance=query_instance)
print(cf)
```

> ⚠️ The adjacency matrix above is a simple example. In practice, it can be provided by expert knowledge or learned using tools like DirectLiNGAM.



## Usage Tutorials

Learn more and explore advanced examples:

* **Documentation**: [ACFX Docs](https://acfx.readthedocs.io/en/latest/)
* **Python API (Colab Notebooks)**: [Interactive Examples](https://colab.research.google.com/drive/1Hj6yH4UIrAp1Jp6B1U542vcdSkXuZHUd?usp=sharing)
* **Graphical User Interface (GUI)**: [ACFX GUI](https://acfx.readthedocs.io/en/latest/)



## Benchmarking

### Install dependencies

```bash
pip install acfx[benchmark]
```

### Run benchmark

```bash
git clone https://github.com/sbobek/acfx
cd acfx/src
python -m acfx.benchmark.main
```

> ⚠️ Always run as a module from `acfx/src` using `python -m`. Running directly or from another folder may cause import or path errors.



## Citation

TBD





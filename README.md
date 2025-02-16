# Actionable Counterfactual Explanations
The repository is a software implementation and set of additional utility tools for a CCF (Constrained Counterfactuals) method.
## Installation

First install all python packages
``` bash
conda create --name ccfenv python=3.11
conda activate ccfenv
conda install pip
pip install -r requirements.txt
```

If you want to run benchmarks, you need to install the following packages that are required by LORE:
``` bash
sudo apt-get update
sudo apt-get install -y libtbb2
```

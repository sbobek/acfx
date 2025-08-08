Welcome to the ACFX documentation
---------------------------------
**ACFX** is Actionable Counterfactual eXplainer. It is a model-agnostic XAI algorithm that provides actionable counterfactual explanations for any type of machine-learning model.

## Installation
To be able to use the acfx package, install it using pip from PyPi
``` bash
conda create --name acfx_env --python 3.11
conda activate acfx_env
conda install pip
pip install acfx
```
...or from source code
``` bash
git clone https://github.com/sbobek/acfx
cd acfx/src
pip install .
```
Welcome to the ACFX documentation
---------------------------------
**ACFX** is Actionable Counterfactual eXplainer. It is a model-agnostic XAI algorithm that provides actionable counterfactual explanations for any type of machine-learning model.

## Installation
To be able to use the acfx package, install it using pip from PyPi
``` bash
conda create --name acfx_env python=3.11
conda activate acfx_env
conda install pip
pip install acfx
```
...or directly from source code
``` bash
git clone https://github.com/sbobek/acfx
cd acfx/src
pip install .
```

## Benchmark
To be able to run benchmark, install dependencies by running
``` bash
conda create --name acfx_env python=3.11
conda activate acfx_env
conda install pip
pip install acfx[benchmark]
```
...or directly from source code
``` bash
git clone https://github.com/sbobek/acfx
cd acfx/src
pip install .[benchmark]
```
To run benchmark, type:
``` bash
git clone https://github.com/sbobek/acfx
cd acfx/src/
python -m acfx.benchmark.main
```
Make sure to run the main.py file as a module using python -m 
and execute it from the acfx/src/acfx directory. 
This ensures that Python treats the acfx folder as a package, allowing relative imports inside 
the script to work correctly. Running the script directly (e.g., python benchmark/main.py) 
or from a different location may result in import errors due to incorrect module resolution (and) 
or invalid filepaths given in some of the benchmark methods.
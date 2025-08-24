# ACFX Counterfactuals discovery app

This project is a Streamlit-based web application for interactive counterfactual discovery using ACFX API

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/counterfactual-discovery-app.git
cd counterfactual-discovery-app
```
2. Build environment
```bash
cd ./src/app
conda create --name acfx_env --python 3.11
conda activate acfx_env
conda install pip
pip install acfx
pip install -r requirements.txt
```

## Evaluation
To start the streamlit app, run:
```bash
streamlit run "Welcome to the ACFX.py"
```
This will run a streamlit's local server for the application. 
In console output, you will see URL to the start page. 
It will be automatically opened after running the command.
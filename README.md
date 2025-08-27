# Uniformly Scaling Flows for Anomaly Detection
This repository contains experiments
investigating uniformly scaling normalizing flows for anomaly
detection. We use the [USFlows](https://github.com/aai-institute/USFlows) 
implementation of uniformly scaling flows. 

# Installation
1) Clone the repository
2) Clone the VeriFlow repository
```bash
git clone git@github.com:aai-institute/USFlows.git
```
3) Create NF4AD poetry environment. 
```bash
  cd path/to/nf4ad/project/directory
  poetry install
```
4) Install the USFlows library
```bash
poetry run pip install -e path/to/usflows/project/directory
```

# Run experiments
The experiments for uniformly scaling normalizing flows can be run with the
following command from within the nf4ad project directory:
```bash
python scripts/run-experiment.py --config experiments/<experiment>/<expconfig>.yaml
```
where `<experiment>` is the name of the experiment and `<expconfig>` is the name
of the configuration file for the experiment. The configuration files are
located in the `experiments` directory.

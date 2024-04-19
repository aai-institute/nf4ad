# nf4ad
Normalizing flows for anomaly detection. This repository contains experiments
investigating the use of uniformly scaling normalizing flows for anomaly
detection. 

# Installation
1) Clone the repository
2) Clone the VeriFlow repository
```bash
git clone git@github.com:aai-institute/VeriFlow.git
```
3) Create a new virtual environment, e.g. with Conda
```bash
conda create -n nf4ad python=3.10
conda activate nf4ad
```
4) Install the requirements
```bash
pip install path/to/veriflow/project/directory
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
```

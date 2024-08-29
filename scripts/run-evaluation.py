import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import sys
sys.path.append(os.getcwd())

import typing as T

import click

from src.explib.config_parser import read_config
 
from nf4ad.eval import Evaluation
Pathable = T.Union[str, os.PathLike]  # In principle one can cast it to os.path.Path
import torch 
torch.autograd.set_detect_anomaly(True)

@click.command()
@click.option("--report_dir", default="./reports", help="Report file")
@click.option("--config", default="./config.yaml", help="Prefix for config items") 
def run(report_dir: Pathable, config: Pathable):
    """Loads an experiment from config file conducts the experiment it.

    Args:
        report_dir (str): Directory to save report to.
        config (str): Path to config file. The report is expected to be specified in .yaml format with
        support to some special key functionalities (see :func:`~laplace_flows.experiments.utils.read_config)
        Defaults to "./config.yaml".
    """
    sepline = "\n" + ("-" * 80) + "\n" + ("-" * 80) + "\n"
    print(f"{sepline}Parsing config file:{sepline}")
    config = os.path.abspath(config)
    experiment = read_config(config)
     
    print(f"{sepline}Done.{sepline}")
      
    print(f"{sepline}Conducting evaluation{sepline}")
    # Conduct evaluation
    
    evaluation = Evaluation(experiment)
    evaluation.conduct(report_dir)
    print(f"{sepline}Done.{sepline}")


if __name__ == "__main__":
    run()

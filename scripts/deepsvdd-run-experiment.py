from pathlib import Path

import yaml
import subprocess
import argparse

ROOT_DIR = Path(__file__).parent.parent
MAIN_PATH = ROOT_DIR / 'deepsvdd/src/main.py'


def read_yaml(file_path):
    """Read the YAML configuration file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def construct_command(config):
    """Construct the command to run the main script with parameters from the YAML config."""
    command = [
        "python", str(MAIN_PATH),
        config['dataset_name'],
        config['net_name'],
        config['xp_path'],
        config['data_path'],
        "--device", str(config['device']),
        "--n_jobs_dataloader", str(config['n_jobs_dataloader']),
        "--seed", str(config['seed']),
        "--objective", config['objective'],
        "--nu", str(config['nu']),
        "--lr", str(config['lr']),
        "--n_epochs", str(config['n_epochs']),
        "--lr_milestone", str(config['lr_milestone']),
        "--batch_size", str(config['batch_size']),
        "--weight_decay", str(config['weight_decay']),
        "--pretrain", str(config['pretrain']),
        "--ae_lr", str(config['ae_lr']),
        "--ae_n_epochs", str(config['ae_n_epochs']),
        "--ae_lr_milestone", str(config['ae_lr_milestone']),
        "--ae_batch_size", str(config['ae_batch_size']),
        "--ae_weight_decay", str(config['ae_weight_decay']),
        "--normal_class", str(config['normal_class'])
    ]
    return " ".join(command)


def main():
    parser = argparse.ArgumentParser(description="Run DeepSVDD experiments with YAML configuration")
    parser.add_argument('-d', '--dataset',
                        type=str,
                        choices=['mnist', 'fashion', 'cifar-10'],
                        default='fashion ',
                        help='Specify dataset for path to the YAML configuration file')
    dataset = parser.parse_args().dataset

    if dataset == 'fashion':
        config_path = ROOT_DIR / 'experiments/fashion/deepsvdd-fashion.yaml'
    elif dataset == 'cifar-10':
        config_path = ROOT_DIR / 'experiments/cifar-10/deepsvdd-cifar-10.yaml'
    elif dataset == 'mnist':
        config_path = ROOT_DIR / 'experiments/mnist/deepsvdd-mnist.yaml'
    else:
        raise Exception('Invalid dataset_name, this should never happen!')

    config = read_yaml(config_path)
    command = construct_command(config)
    print("Executing command:", command)
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()

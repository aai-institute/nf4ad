"""
Comprehensive benchmarking system for Flow models on ADBench datasets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

from .adbench_wrapper import ADBenchFlow
from .flows import NonUSFlow
import pyro.distributions as dist

# Try to import USFlow from usflows
try:
    from src.usflows.flows import USFlow
    USFLOW_AVAILABLE = True
except ImportError:
    USFLOW_AVAILABLE = False
    USFlow = None


# ---------------------------------------------------------------------------
# Configuration and Data Classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    data_dir: Path = Path("./data/adbench")
    output_dir: Path = Path("./results/adbench")
    random_state: int = 42
    test_size: float = 0.5
    n_jobs: int = 1
    verbose: bool = True
    save_models: bool = False
    device: str = "cuda"


@dataclass
class FlowConfig:
    """Configuration for Flow model."""
    coupling_blocks: int = 8
    hidden_dim: int = 128
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    patience: Optional[int] = 10
    min_delta: float = 1e-4
    gradient_clip: Optional[float] = None
    clamp: float = 5.0
    # USFlow-specific parameters
    lu_transform: int = 1
    householder: int = 0
    affine_conjugation: bool = True
    prior_scale: Optional[float] = 1.0
    masktype: str = "checkerboard"  # "checkerboard" or "channel"
    flow_type: str = "nonusflow"  # "nonusflow" or "usflow"


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    dataset: str
    n_features: int
    n_train: int
    n_test: int
    n_anomalies_train: int
    n_anomalies_test: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    training_time: float
    inference_time: float
    n_epochs_trained: int
    training_losses: List[float]
    timestamp: str


# ---------------------------------------------------------------------------
# ADBench Dataset Catalog
# ---------------------------------------------------------------------------

ADBENCH_CLASSICAL_DATASETS = [
    "1_ALOI",
    "2_annthyroid",
    "3_backdoor",
    "4_breastw",
    "5_campaign",
    "6_cardio",
    "7_Cardiotocography",
    "8_celeba",
    "9_census",
    "10_cover",
    "11_donors",
    "12_fault",
    "13_fraud",
    "14_glass",
    "15_Hepatitis",
    "16_http",
    "17_InternetAds",
    "18_Ionosphere",
    "19_landsat",
    "20_letter",
    "21_Lymphography",
    "22_magic.gamma",
    "23_mammography",
    "24_mnist",
    "25_musk",
    "26_optdigits",
    "27_PageBlocks",
    "28_pendigits",
    "29_Pima",
    "30_satellite",
    "31_satimage-2",
    "32_shuttle",
    "33_skin",
    "34_smtp",
    "35_SpamBase",
    "36_speech",
    "37_Stamps",
    "38_thyroid",
    "39_vertebral",
    "40_vowels",
    "41_Waveform",
    "42_WBC",
    "43_WDBC",
    "44_Wilt",
    "45_wine",
    "46_WPBC",
    "47_yeast",
]


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------

def load_adbench_dataset(
    dataset_name: str,
    data_dir: Path,
    test_size: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and split an ADBench dataset."""
    # Find the .npz file
    dataset_name = dataset_name.strip()
    data_dir = Path(data_dir)
    
    # Try direct match first
    npz_path = data_dir / dataset_name
    if not npz_path.exists() and not dataset_name.endswith('.npz'):
        npz_path = data_dir / f"{dataset_name}.npz"
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset {dataset_name} not found in {data_dir}")
    
    # Load data
    npz = np.load(npz_path, allow_pickle=True)
    X = npz["X"]
    y = npz["y"].astype(int)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Flow Model Creation
# ---------------------------------------------------------------------------

def create_nonusflow_model(n_features: int, config: FlowConfig, device: str) -> NonUSFlow:
    """Create a NonUSFlow model for anomaly detection."""
    device = torch.device(device)
    
    base_dist = dist.Normal(
        torch.zeros(n_features).to(device),
        torch.ones(n_features).to(device)
    )
    
    class SimpleConditioner(nn.Module):
        def __init__(self, in_dim, out_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        
        def forward(self, x):
            return self.net(x)
    
    flow = NonUSFlow(
        in_dims=[n_features],
        device=device,
        coupling_blocks=config.coupling_blocks,
        base_distribution=base_dist,
        prior_scale=config.prior_scale,
        affine_conjugation=config.affine_conjugation,
        clamp=config.clamp,
        conditioner_cls=SimpleConditioner,
        conditioner_args={
            'in_dim': n_features,
            'out_dim': n_features * 2,
            'hidden_dim': config.hidden_dim,
        },
        nonlinearity=nn.ReLU(),
    )
    
    return flow


def create_usflow_model(n_features: int, config: FlowConfig, device: str):
    """Create a USFlow model for anomaly detection."""
    if not USFLOW_AVAILABLE:
        raise ImportError(
            "USFlow is not available. Please install usflows: "
            "pip install usflows or clone from https://github.com/your-repo/USFlows"
        )
    
    device = torch.device(device)
    
    base_dist = dist.Normal(
        torch.zeros(n_features).to(device),
        torch.ones(n_features).to(device)
    )
    
    class SimpleConditioner(nn.Module):
        def __init__(self, in_dims, **kwargs):
            super().__init__()
            in_dim = in_dims[0]
            hidden_dim = kwargs.get('hidden_dim', 128)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, in_dim),  # USFlow expects output = input dim for additive coupling
            )
        
        def forward(self, x):
            return self.net(x)
    
    flow = USFlow(
        in_dims=[n_features],
        coupling_blocks=config.coupling_blocks,
        base_distribution=base_dist,
        prior_scale=config.prior_scale,
        affine_conjugation=config.affine_conjugation,
        lu_transform=config.lu_transform,
        householder=config.householder,
        masktype=config.masktype,
        conditioner_cls=SimpleConditioner,
        conditioner_args={
            'in_dims': [n_features],
            'hidden_dim': config.hidden_dim,
        },
        nonlinearity=nn.ReLU(),
        device=device,
    )
    
    return flow


def create_flow_model(n_features: int, config: FlowConfig, device: str) -> Union[NonUSFlow, 'USFlow']:
    """Create a Flow model for anomaly detection based on config.flow_type."""
    if config.flow_type.lower() == "usflow":
        return create_usflow_model(n_features, config, device)
    elif config.flow_type.lower() == "nonusflow":
        return create_nonusflow_model(n_features, config, device)
    else:
        raise ValueError(
            f"Unknown flow_type: {config.flow_type}. "
            "Supported types: 'nonusflow', 'usflow'"
        )


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_single_experiment(
    dataset_name: str,
    flow_config: FlowConfig,
    benchmark_config: BenchmarkConfig,
) -> ExperimentResult:
    """Run a single experiment on one dataset with one configuration."""
    if benchmark_config.verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment on {dataset_name}")
        print(f"Flow type: {flow_config.flow_type}")
        print(f"{'='*60}")
    
    # Load data
    X_train, X_test, y_train, y_test = load_adbench_dataset(
        dataset_name,
        benchmark_config.data_dir,
        benchmark_config.test_size,
        benchmark_config.random_state,
    )
    
    n_features = X_train.shape[1]
    
    if benchmark_config.verbose:
        print(f"Dataset: {dataset_name}")
        print(f"Features: {n_features}")
        print(f"Train: {len(X_train)} samples ({y_train.sum()} anomalies)")
        print(f"Test: {len(X_test)} samples ({y_test.sum()} anomalies)")
    
    # Create model
    flow_model = create_flow_model(n_features, flow_config, benchmark_config.device)
    wrapper = ADBenchFlow(
        flow_model=flow_model,
        batch_size=flow_config.batch_size,
        epochs=flow_config.epochs,
        lr=flow_config.lr,
        device=benchmark_config.device,
        gradient_clip=flow_config.gradient_clip,
        verbose=benchmark_config.verbose,
        patience=flow_config.patience,
        min_delta=flow_config.min_delta,
    )
    
    # Train
    start_time = time.time()
    wrapper.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Inference
    start_time = time.time()
    scores = wrapper.predict_score(X_test)
    inference_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_metrics(y_test, scores)
    
    # Create result
    result = ExperimentResult(
        dataset=dataset_name,
        n_features=n_features,
        n_train=len(X_train),
        n_test=len(X_test),
        n_anomalies_train=int(y_train.sum()),
        n_anomalies_test=int(y_test.sum()),
        config=asdict(flow_config),
        metrics=metrics,
        training_time=training_time,
        inference_time=inference_time,
        n_epochs_trained=len(wrapper.training_losses_),
        training_losses=wrapper.training_losses_,
        timestamp=pd.Timestamp.now().isoformat(),
    )
    
    if benchmark_config.verbose:
        print(f"\nResults:")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")
        print(f"  Best F1: {metrics['best_f1']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Epochs: {len(wrapper.training_losses_)}")
    
    return result


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute anomaly detection metrics."""
    metrics = {}
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, scores))
    except:
        metrics['roc_auc'] = float('nan')
    
    # PR-AUC
    try:
        metrics['pr_auc'] = float(average_precision_score(y_true, scores))
    except:
        metrics['pr_auc'] = float('nan')
    
    # Best F1
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        best_idx = int(np.argmax(f1))
        metrics['best_f1'] = float(f1[best_idx])
        if best_idx < len(thresholds):
            metrics['best_threshold'] = float(thresholds[best_idx])
        else:
            metrics['best_threshold'] = float('nan')
    except:
        metrics['best_f1'] = float('nan')
        metrics['best_threshold'] = float('nan')
    
    return metrics


# ---------------------------------------------------------------------------
# Batch Benchmarking
# ---------------------------------------------------------------------------

class ADBenchBenchmark:
    """Main benchmarking class."""
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.results: List[ExperimentResult] = []
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run_on_datasets(
        self,
        datasets: List[str],
        flow_config: FlowConfig,
        parallel: bool = False,
    ) -> List[ExperimentResult]:
        """Run experiments on multiple datasets."""
        results = []
        
        if parallel and self.config.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                futures = {
                    executor.submit(
                        run_single_experiment,
                        dataset,
                        flow_config,
                        self.config,
                    ): dataset
                    for dataset in datasets
                }
                
                for future in as_completed(futures):
                    dataset = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.results.append(result)
                    except Exception as e:
                        print(f"Error on dataset {dataset}: {e}")
        else:
            for dataset in datasets:
                try:
                    result = run_single_experiment(dataset, flow_config, self.config)
                    results.append(result)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error on dataset {dataset}: {e}")
        
        return results
    
    def hyperparameter_search(
        self,
        datasets: List[str],
        param_grid: Dict[str, List[Any]],
        n_trials: Optional[int] = None,
    ) -> pd.DataFrame:
        """Run hyperparameter search."""
        print(f"\n{'='*60}")
        print("Starting Hyperparameter Search")
        print(f"{'='*60}")
        print(f"Datasets: {len(datasets)}")
        print(f"Parameter grid size: {len(list(ParameterGrid(param_grid)))}")
        
        # Generate configurations
        configs = []
        for params in ParameterGrid(param_grid):
            config = FlowConfig(**params)
            configs.append(config)
        
        # Limit number of trials if specified
        if n_trials is not None and n_trials < len(configs):
            import random
            random.seed(self.config.random_state)
            configs = random.sample(configs, n_trials)
        
        print(f"Running {len(configs)} configurations")
        
        # Run all combinations
        all_results = []
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Configuration: {asdict(config)}")
            
            for dataset in datasets:
                try:
                    result = run_single_experiment(dataset, config, self.config)
                    all_results.append(result)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error on {dataset} with config {i}: {e}")
        
        # Convert to DataFrame
        df = self.results_to_dataframe(all_results)
        
        # Save results
        output_path = self.config.output_dir / f"hyperparam_search_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        return df
    
    def results_to_dataframe(self, results: Optional[List[ExperimentResult]] = None) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        if results is None:
            results = self.results
        
        rows = []
        for r in results:
            row = {
                'dataset': r.dataset,
                'n_features': r.n_features,
                'n_train': r.n_train,
                'n_test': r.n_test,
                'n_anomalies_test': r.n_anomalies_test,
                'roc_auc': r.metrics['roc_auc'],
                'pr_auc': r.metrics['pr_auc'],
                'best_f1': r.metrics['best_f1'],
                'training_time': r.training_time,
                'inference_time': r.inference_time,
                'n_epochs': r.n_epochs_trained,
                **{f'config_{k}': v for k, v in r.config.items()},
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self, filename: Optional[str] = None):
        """Save all results to JSON and CSV."""
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_results_{timestamp}"
        
        # Save as JSON (full details)
        json_path = self.config.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        # Save as CSV (summary)
        csv_path = self.config.output_dir / f"{filename}.csv"
        df = self.results_to_dataframe()
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
    
    def summary(self) -> pd.DataFrame:
        """Get summary statistics across all results."""
        df = self.results_to_dataframe()
        
        summary = df.groupby('dataset').agg({
            'roc_auc': ['mean', 'std', 'max'],
            'pr_auc': ['mean', 'std', 'max'],
            'best_f1': ['mean', 'std', 'max'],
            'training_time': ['mean', 'std'],
            'n_epochs': ['mean', 'std'],
        }).round(4)
        
        return summary

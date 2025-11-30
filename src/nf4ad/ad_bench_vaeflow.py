from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, List
import time
import json

import numpy as np
import pandas as pd
import torch
from numpy.typing import ArrayLike

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

from src.usflows.flows import USFlow
from src.usflows.networks import ConvNet

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
DATA_DIR: Path = Path("./data/adbench")
RESULTS_DIR: Path = Path("./results/vaeflow_benchmark")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# All ADBench classical datasets
ADBENCH_CLASSICAL_DATASETS = [
    "1_ALOI", "2_annthyroid", "3_backdoor", "4_breastw",
    "5_campaign", "6_cardio", "7_Cardiotocography", "8_celeba",
    "9_census", "10_cover", "11_donors", "12_fault",
    "13_fraud", "14_glass", "15_Hepatitis", "16_http",
    "17_InternetAds", "18_Ionosphere", "19_landsat", "20_letter",
    "21_Lymphography", "22_magic.gamma", "23_mammography", "24_mnist",
    "25_musk", "26_optdigits", "27_PageBlocks", "28_pendigits",
    "29_Pima", "30_satellite", "31_satimage-2", "32_shuttle",
    "33_skin", "34_smtp", "35_SpamBase", "36_speech",
    "37_Stamps", "38_thyroid", "39_vertebral", "40_vowels",
    "41_Waveform", "42_WBC", "43_WDBC", "44_Wilt",
    "45_wine", "46_WPBC", "47_yeast"
]


# ---------------------------------------------------------------------------
# Dataset resolution & loading
# ---------------------------------------------------------------------------

def resolve_npz_path(dataset_name: str, data_dir: Path = DATA_DIR) -> Path:
    dataset_name = dataset_name.strip()
    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory {data_dir.resolve()} does not exist. "
            "Make sure you created it and copied the ADBench .npz files into it."
        )

    candidate = data_dir / dataset_name
    if candidate.is_file():
        return candidate

    if not dataset_name.endswith(".npz"):
        candidate_with_ext = data_dir / f"{dataset_name}.npz"
        if candidate_with_ext.is_file():
            return candidate_with_ext

    candidates = []

    if dataset_name.isdigit():
        prefix = f"{dataset_name}_"
        candidates = [p for p in data_dir.glob("*.npz") if p.name.startswith(prefix)]
    else:
        norm = dataset_name.lower()
        for p in data_dir.glob("*.npz"):
            stem = p.stem.lower()
            if stem == norm:
                candidates.append(p)
                continue
            if "_" in stem:
                _, suffix = stem.split("_", 1)
                if suffix == norm:
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not match dataset name '{dataset_name}' to any .npz file in "
            f"{data_dir.resolve()}."
        )

    if len(candidates) > 1:
        names = ", ".join(sorted(p.name for p in candidates))
        raise RuntimeError(
            f"Dataset name '{dataset_name}' is ambiguous; it matches multiple files: "
            f"{names}. Please specify a more precise name."
        )

    return candidates[0]


def load_classical_dataset(
    dataset_name: str,
    data_dir: Path = DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = resolve_npz_path(dataset_name, data_dir)
    npz = np.load(npz_path, allow_pickle=True)

    X = npz["X"]
    y = npz["y"].astype(int)

    print(
        f"Loaded {npz_path.name}: X.shape={X.shape}, "
        f"y.shape={y.shape}, anomaly_ratio={y.mean():.4f}"
    )

    return X, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_anomaly_scores(
    y_true: ArrayLike,
    scores: ArrayLike,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError(
            f"y_true and scores must have the same length, "
            f"got {y_true.shape[0]} and {scores.shape[0]}."
        )

    if np.unique(y_true).size < 2:
        raise ValueError(
            "y_true must contain both normal (0) and anomalous (1) labels."
        )

    metrics: Dict[str, float] = {}
    metrics["auc_roc"] = float(roc_auc_score(y_true, scores))
    metrics["auc_pr"] = float(average_precision_score(y_true, scores))

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1))
    metrics["best_f1"] = float(f1[best_idx])

    if thresholds.size > 0 and best_idx < thresholds.size:
        metrics["best_f1_threshold"] = float(thresholds[best_idx])
    else:
        metrics["best_f1_threshold"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Flow prior creation
# ---------------------------------------------------------------------------

def create_flow_prior(latent_dim: int, us: bool, device: torch.device):
    """Helper to create flow prior with specified latent dimension."""
    from nf4ad.flows import NonUSFlow
    import pyro.distributions as dist
    import torch.nn as nn
    
    base_dist = dist.Normal(
        torch.zeros(latent_dim).to(device),
        torch.ones(latent_dim).to(device)
    )
    
    if us:
        flow = USFlow(
            in_dims=[latent_dim],
            device=device,
            coupling_blocks=4,
            base_distribution=base_dist,
            prior_scale=1.0,
            affine_conjugation=True,
            conditioner_cls=ConvNet,
            conditioner_args={
                'in_dims': [latent_dim],
                'c_hidden': [256, 256, 256, 256],
                'c_out': latent_dim,
                'nonlinearity': nn.ReLU(),
                'normalize_layers': True,
                'gating': True,
            },
            nonlinearity=nn.ReLU(),
        )
    else:
        flow = NonUSFlow(
            in_dims=[latent_dim],
            device=device,
            coupling_blocks=4,
            base_distribution=base_dist,
            prior_scale=1.0,
            affine_conjugation=True,
            conditioner_cls=ConvNet,
            conditioner_args={
                'in_dims': [latent_dim],
                'c_hidden': [256, 256, 256, 256],
                'c_out': latent_dim * 2,
                'nonlinearity': nn.ReLU(),
                'normalize_layers': True,
                'gating': True,
            },
            nonlinearity=nn.ReLU(),
        )
    
    return flow


# ---------------------------------------------------------------------------
# VAEFlow benchmark function
# ---------------------------------------------------------------------------

def run_vaeflow_on_dataset(
    dataset_name: str,
    data_dir: Path = DATA_DIR,
    device: str = "cpu",
    us: bool = True,
) -> Dict[str, Any]:
    """
    Run VAEFlow on a single dataset with adaptive architecture.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing dataset files
        device: Device to use ('cpu', 'cuda', or 'mps')
    
    Returns:
        Dictionary with results including metrics and timing
    """
    print(f"\n{'='*80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*80}")
    
    # Load dataset
    X, y = load_classical_dataset(dataset_name, data_dir=data_dir)
    n_features = X.shape[1]
    
    # Adaptive latent dimension based on feature count
    latent_dim = 2 if n_features < 128 else 16
    print(f"Features: {n_features}, Using latent_dim: {latent_dim}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Import wrapper
    from nf4ad.adbench_wrapper import ADBenchVAEFlow
    
    # Create model with fixed architecture
    device_obj = torch.device(device)
    vaeflow = ADBenchVAEFlow(
        flow_prior=create_flow_prior(latent_dim=latent_dim, us=us, device=device_obj),
        input_dim=n_features,
        latent_dim=latent_dim,
        encoder_hidden_dims=[512, 256, 128],  # Fixed architecture
        decoder_hidden_dims=[128, 256, 512],  # Fixed architecture
        dropout=0.2,
        use_batchnorm=True,
        epochs=100,
        batch_size=64,
        lr=1e-3,
        patience=10,
        verbose=True,
        device=device,
    )
    
    # Train
    start_time = time.time()
    vaeflow.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Predict
    scores = vaeflow.predict_score(X_test_scaled)
    
    # Evaluate
    metrics = evaluate_anomaly_scores(y_test, scores)
    
    # Prepare result
    result = {
        "dataset": dataset_name,
        "n_features": int(n_features),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_anomalies_test": int(y_test.sum()),
        "anomaly_ratio": float(y.mean()),
        "latent_dim": latent_dim,
        "training_time": float(training_time),
        "n_epochs": len(vaeflow.training_losses_) if vaeflow.training_losses_ else 0,
        "final_loss": float(vaeflow.training_losses_[-1]) if vaeflow.training_losses_ else float('nan'),
        "roc_auc": metrics["auc_roc"],
        "pr_auc": metrics["auc_pr"],
        "best_f1": metrics["best_f1"],
        "best_f1_threshold": metrics["best_f1_threshold"],
    }
    
    print(f"\nResults:")
    print(f"  ROC-AUC: {metrics['auc_roc']:.4f}")
    print(f"  PR-AUC:  {metrics['auc_pr']:.4f}")
    print(f"  Best F1: {metrics['best_f1']:.4f}")
    print(f"  Training time: {training_time:.2f}s")
    print(f"  Epochs: {result['n_epochs']}")
    
    return result


# ---------------------------------------------------------------------------
# Main benchmark script
# ---------------------------------------------------------------------------

def run_full_benchmark(
    datasets: List[str] = None,
    data_dir: Path = DATA_DIR,
    results_dir: Path = RESULTS_DIR,
    us: bool = True,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Run VAEFlow benchmark on all specified datasets.
    
    Args:
        datasets: List of dataset names (default: all ADBENCH datasets)
        data_dir: Directory containing dataset files
        results_dir: Directory to save results
        device: Device to use
    
    Returns:
        DataFrame with all results
    """
    if datasets is None:
        datasets = ADBENCH_CLASSICAL_DATASETS
    
    print(f"{'='*80}")
    print(f"VAEFlow Benchmark on ADBench Classical Datasets")
    print(f"{'='*80}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Device: {device}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")
    
    results = []
    failed = []

    arch = "usf" if us else "non-usf"
    
    for i, dataset_name in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing {dataset_name}...")
        
        try:
            result = run_vaeflow_on_dataset(
                dataset_name=dataset_name,
                data_dir=data_dir,
                device=device,
                us=us
            )
            results.append(result)
            
        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            failed.append({"dataset": dataset_name, "error": str(e)})
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = results_dir / f"vaeflow_{arch}_results_{timestamp}.csv"
    json_file = results_dir / f"vaeflow_{arch}_results_{timestamp}.json"
    
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved to: {csv_file}")
    
    # Save detailed JSON
    with open(json_file, 'w') as f:
        json.dump({
            "results": results,
            "failed": failed,
            "config": {
                "encoder_hidden_dims": [512, 256, 128],
                "decoder_hidden_dims": [128, 256, 512],
                "dropout": 0.2,
                "use_batchnorm": True,
                "epochs": 100,
                "batch_size": 64,
                "lr": 1e-3,
                "patience": 10,
                "arch": arch,
            }
        }, f, indent=2)
    print(f"Detailed results saved to: {json_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Successful: {len(results)}/{len(datasets)}")
    print(f"Failed: {len(failed)}/{len(datasets)}")
    
    if results:
        print(f"\nPerformance Statistics:")
        print(f"  ROC-AUC:  {df['roc_auc'].mean():.4f} ± {df['roc_auc'].std():.4f}")
        print(f"  PR-AUC:   {df['pr_auc'].mean():.4f} ± {df['pr_auc'].std():.4f}")
        print(f"  Best F1:  {df['best_f1'].mean():.4f} ± {df['best_f1'].std():.4f}")
        print(f"\nTiming:")
        print(f"  Total time: {df['training_time'].sum()/3600:.2f} hours")
        print(f"  Avg time per dataset: {df['training_time'].mean():.2f}s")
        print(f"  Avg epochs: {df['n_epochs'].mean():.1f}")
        
        print(f"\nTop 5 datasets by ROC-AUC:")
        top5 = df.nlargest(5, 'roc_auc')[['dataset', 'roc_auc', 'pr_auc', 'best_f1']]
        print(top5.to_string(index=False))
    
    if failed:
        print(f"\nFailed datasets:")
        for fail in failed:
            print(f"  - {fail['dataset']}: {fail['error']}")
    
    print(f"\n{'='*80}\n")
    
    return df


# ---------------------------------------------------------------------------
# Run the benchmark
# ---------------------------------------------------------------------------

# Run on all datasets (or specify a subset for testing)
# For quick test, uncomment the line below:
# test_datasets = ["6_cardio", "2_annthyroid", "38_thyroid"]
# df_results = run_full_benchmark(datasets=test_datasets, device="cpu")

# For full benchmark:
df_results = run_full_benchmark(us=True, device="cuda")

# Display results
print("\nFinal Results DataFrame:")
print(df_results[['dataset', 'n_features', 'latent_dim', 'roc_auc', 'pr_auc', 'best_f1', 'training_time', 'n_epochs']])

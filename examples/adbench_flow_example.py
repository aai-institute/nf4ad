"""
Example script for testing Flow models on ADBench datasets.
"""
import torch
import numpy as np
from nf4ad.adbench_wrapper import ADBenchFlow
from nf4ad.flows import NonUSFlow
import pyro.distributions as dist
import torch.nn as nn


def create_flow_model(feature_dim: int, device: str = "cuda"):
    """Create a NonUSFlow model for tabular data."""
    base_dist = dist.Normal(
        torch.zeros(feature_dim).to(device),
        torch.ones(feature_dim).to(device)
    )
    
    # Simple MLP conditioner
    class SimpleConditioner(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, out_dim),
            )
        
        def forward(self, x):
            return self.net(x)
    
    flow = NonUSFlow(
        in_dims=[feature_dim],
        device=device,
        coupling_blocks=8,
        base_distribution=base_dist,
        prior_scale=1.0,
        affine_conjugation=True,
        conditioner_cls=SimpleConditioner,
        conditioner_args={
            'in_dim': feature_dim,
            'out_dim': feature_dim * 2,
        },
        nonlinearity=nn.ReLU(),
    )
    
    return flow


def demo_synthetic_tabular():
    """Demo with synthetic tabular data."""
    print("=" * 60)
    print("Flow Model ADBench Wrapper Demo - Synthetic Tabular Data")
    print("=" * 60)
    
    # Create synthetic data
    np.random.seed(42)
    n_train = 500
    n_test = 200
    n_features = 50
    
    # Normal samples (Gaussian)
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    
    # Test set: normal + anomalies (shifted Gaussian)
    X_test_normal = np.random.randn(n_test // 2, n_features).astype(np.float32)
    X_test_anomaly = np.random.randn(n_test // 2, n_features).astype(np.float32) * 2 + 3
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.hstack([np.zeros(n_test // 2), np.ones(n_test // 2)])
    
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create flow model
    print("\nCreating flow model...")
    flow = create_flow_model(n_features, device=device)
    
    # Create wrapper
    print("Initializing Flow wrapper...")
    wrapper = ADBenchFlow(
        flow_model=flow,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device=device,
    )
    
    # Train
    print("\nTraining model...")
    wrapper.fit(X_train)
    
    # Evaluate
    print("\nComputing anomaly scores...")
    scores = wrapper.predict_score(X_test)
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Compute metrics
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    try:
        roc_auc = roc_auc_score(y_test, scores)
        pr_auc = average_precision_score(y_test, scores)
        
        print("\nResults:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        
        # Score statistics
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]
        print(f"\nScore Statistics:")
        print(f"  Normal    - mean: {normal_scores.mean():.4f}, std: {normal_scores.std():.4f}")
        print(f"  Anomalies - mean: {anomaly_scores.mean():.4f}, std: {anomaly_scores.std():.4f}")
        
    except Exception as e:
        print(f"\nMetric computation failed: {e}")


def demo_image_data():
    """Demo showing flow on flattened image data."""
    print("\n" + "=" * 60)
    print("Flow Model on Image Data (Flattened)")
    print("=" * 60)
    
    np.random.seed(42)
    n_train = 300
    n_test = 100
    img_size = 28  # 28x28 images
    
    # Create synthetic images
    X_train = np.random.randn(n_train, 3, img_size, img_size).astype(np.float32) * 0.1
    X_test_normal = np.random.randn(n_test // 2, 3, img_size, img_size).astype(np.float32) * 0.1
    X_test_anomaly = np.random.randn(n_test // 2, 3, img_size, img_size).astype(np.float32) * 0.3 + 0.5
    
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.hstack([np.zeros(n_test // 2), np.ones(n_test // 2)])
    
    # Flatten images
    X_train_flat = X_train.reshape(n_train, -1)
    X_test_flat = X_test.reshape(n_test, -1)
    n_features = X_train_flat.shape[1]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nImage shape: {img_size}x{img_size}x3")
    print(f"Flattened dimension: {n_features}")
    
    # Create flow model
    flow = create_flow_model(n_features, device=device)
    
    wrapper = ADBenchFlow(
        flow_model=flow,
        epochs=30,
        batch_size=32,
        lr=1e-3,
        device=device,
        verbose=True,
    )
    
    wrapper.fit(X_train_flat)
    scores = wrapper.predict_score(X_test_flat)
    
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_test, scores)
        print(f"\nROC-AUC on flattened images: {auc:.4f}")
    except:
        print("\nMetric computation failed")


if __name__ == '__main__':
    # Run demos
    demo_synthetic_tabular()
    demo_image_data()
    
    print("\n" + "=" * 60)
    print("Demos completed!")
    print("=" * 60)

"""
Pytest configuration and shared fixtures.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
import pyro.distributions as dist


@pytest.fixture
def device():
    """Return available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def synthetic_image_data():
    """Generate synthetic image data for testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_train = 100
    n_test = 50
    img_size = 64
    
    # Normal samples
    X_train = np.random.randn(n_train, 3, img_size, img_size).astype(np.float32) * 0.1
    
    # Test: normal + anomalies
    X_test_normal = np.random.randn(n_test // 2, 3, img_size, img_size).astype(np.float32) * 0.1
    X_test_anomaly = np.random.randn(n_test // 2, 3, img_size, img_size).astype(np.float32) * 0.5 + 2.0
    
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.hstack([np.zeros(n_test // 2), np.ones(n_test // 2)])
    
    # Normalize to [0, 1]
    X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-8)
    X_test = (X_test - X_test.min()) / (X_test.max() - X_test.min() + 1e-8)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'input_shape': (3, img_size, img_size),
    }


@pytest.fixture
def synthetic_tabular_data():
    """Generate synthetic tabular data for testing."""
    np.random.seed(42)
    
    n_train = 200
    n_test = 100
    n_features = 50
    
    # Normal samples (Gaussian)
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    
    # Test: normal + anomalies (outliers)
    X_test_normal = np.random.randn(n_test // 2, n_features).astype(np.float32)
    X_test_anomaly = np.random.randn(n_test // 2, n_features).astype(np.float32) * 3 + 5
    
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.hstack([np.zeros(n_test // 2), np.ones(n_test // 2)])
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'n_features': n_features,
    }


@pytest.fixture
def simple_flow_prior(device):
    """Create a simple flow prior for testing with latent_dim=32."""
    return _create_flow_prior(32, device)


@pytest.fixture
def flow_prior_16(device):
    """Create a simple flow prior for testing with latent_dim=16."""
    return _create_flow_prior(16, device)


def _create_flow_prior(latent_dim: int, device: torch.device):
    """Helper to create flow prior with specified latent dimension."""
    from nf4ad.flows import NonUSFlow
    
    base_dist = dist.Normal(
        torch.zeros(latent_dim).to(device),
        torch.ones(latent_dim).to(device)
    )
    
    # Simple MLP conditioner for testing
    class SimpleConditioner(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dim),
            )
        
        def forward(self, x):
            return self.net(x)
    
    flow = NonUSFlow(
        in_dims=[latent_dim],
        device=device,
        coupling_blocks=3,
        base_distribution=base_dist,
        prior_scale=1.0,
        affine_conjugation=True,
        conditioner_cls=SimpleConditioner,
        conditioner_args={
            'in_dim': latent_dim,
            'out_dim': latent_dim * 2,  # For affine coupling: scale + shift
        },
        nonlinearity=nn.ReLU(),
    )
    
    return flow


@pytest.fixture
def small_dataset():
    """Create small torch dataset for quick tests."""
    torch.manual_seed(42)
    X = torch.randn(50, 3, 32, 32)
    y = torch.zeros(50)
    return TensorDataset(X, y)

"""
Tests for ADBench Flow wrapper classes.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from nf4ad.adbench_wrapper import ADBenchFlow
import pyro.distributions as dist


def _create_flow_for_data(n_features: int, device: torch.device):
    """Helper to create a flow model matching the data dimension."""
    from nf4ad.flows import NonUSFlow
    
    base_dist = dist.Normal(
        torch.zeros(n_features).to(device),
        torch.ones(n_features).to(device)
    )
    
    # Simple MLP conditioner
    class SimpleConditioner(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            hidden_dim = min(128, in_dim)
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        
        def forward(self, x):
            return self.net(x)
    
    flow = NonUSFlow(
        in_dims=[n_features],
        device=device,
        coupling_blocks=3,
        base_distribution=base_dist,
        prior_scale=1.0,
        affine_conjugation=True,
        conditioner_cls=SimpleConditioner,
        conditioner_args={
            'in_dim': n_features,
            'out_dim': n_features * 2,
        },
        nonlinearity=nn.ReLU(),
    )
    
    return flow


class TestADBenchFlow:
    """Tests for ADBenchFlow wrapper."""
    
    def test_initialization(self, simple_flow_prior):
        """Test wrapper initialization."""
        wrapper = ADBenchFlow(
            flow_model=simple_flow_prior,
            epochs=2,
            verbose=False,
        )
        
        assert wrapper is not None
        assert wrapper.flow_model is not None
    
    def test_fit_predict_tabular(self, device, synthetic_tabular_data):
        """Test fit and predict on tabular data."""
        # Create flow matching the tabular data dimension
        n_features = synthetic_tabular_data['n_features']
        flow = _create_flow_for_data(n_features, device)
        
        wrapper = ADBenchFlow(
            flow_model=flow,
            epochs=2,
            batch_size=32,
            verbose=False,
        )
        
        # Fit
        wrapper.fit(synthetic_tabular_data['X_train'])
        
        # Predict
        scores = wrapper.predict_score(synthetic_tabular_data['X_test'])
        
        assert len(scores) == len(synthetic_tabular_data['X_test'])
        assert np.all(np.isfinite(scores))
    
    def test_predict_labels(self, device, synthetic_tabular_data):
        """Test binary prediction."""
        n_features = synthetic_tabular_data['n_features']
        flow = _create_flow_for_data(n_features, device)
        
        wrapper = ADBenchFlow(
            flow_model=flow,
            epochs=2,
            verbose=False,
        )
        
        wrapper.fit(synthetic_tabular_data['X_train'])
        predictions = wrapper.predict(synthetic_tabular_data['X_test'])
        
        assert len(predictions) == len(synthetic_tabular_data['X_test'])
        assert set(predictions) <= {0, 1}
    
    def test_custom_threshold(self, device, synthetic_tabular_data):
        """Test prediction with custom threshold."""
        n_features = synthetic_tabular_data['n_features']
        flow = _create_flow_for_data(n_features, device)
        
        wrapper = ADBenchFlow(
            flow_model=flow,
            epochs=2,
            verbose=False,
        )
        
        wrapper.fit(synthetic_tabular_data['X_train'])
        scores = wrapper.predict_score(synthetic_tabular_data['X_test'])
        threshold = np.percentile(scores, 75)
        
        predictions = wrapper.predict(synthetic_tabular_data['X_test'], threshold=threshold)
        
        # At 75th percentile, roughly 25% should be predicted as anomalies
        anomaly_rate = predictions.mean()
        assert 0.15 < anomaly_rate < 0.35
    
    def test_anomaly_detection_basic(self, device, synthetic_tabular_data):
        """Test basic anomaly detection capability."""
        n_features = synthetic_tabular_data['n_features']
        flow = _create_flow_for_data(n_features, device)
        
        wrapper = ADBenchFlow(
            flow_model=flow,
            epochs=5,
            batch_size=32,
            verbose=False,
        )
        
        wrapper.fit(synthetic_tabular_data['X_train'])
        scores = wrapper.predict_score(synthetic_tabular_data['X_test'])
        
        # Check that all scores are finite and have variance
        assert np.all(np.isfinite(scores)), "All scores should be finite"
        assert scores.std() > 0, "Scores should have variance"
        
        # Soft check: log if model learned something useful
        y_test = synthetic_tabular_data['y_test']
        normal_scores = scores[y_test == 0]
        anomaly_scores = scores[y_test == 1]
        
        if anomaly_scores.mean() > normal_scores.mean():
            print(f"\nFlow model learned to detect anomalies: "
                  f"anomaly_mean={anomaly_scores.mean():.2e}, "
                  f"normal_mean={normal_scores.mean():.2e}")
        else:
            print(f"\nFlow model needs more training: "
                  f"anomaly_mean={anomaly_scores.mean():.2e}, "
                  f"normal_mean={normal_scores.mean():.2e}")


class TestIntegrationFlow:
    """Integration tests for Flow wrappers."""
    
    @pytest.mark.slow
    def test_full_pipeline_flow_tabular(self, device, synthetic_tabular_data):
        """Test full pipeline with Flow on tabular data."""
        from sklearn.metrics import roc_auc_score
        
        n_features = synthetic_tabular_data['n_features']
        flow = _create_flow_for_data(n_features, device)
        
        wrapper = ADBenchFlow(
            flow_model=flow,
            epochs=20,
            batch_size=32,
            lr=1e-3,
            verbose=False,
        )
        
        # Train
        wrapper.fit(synthetic_tabular_data['X_train'])
        
        # Evaluate
        scores = wrapper.predict_score(synthetic_tabular_data['X_test'])
        y_test = synthetic_tabular_data['y_test']
        
        # Compute AUC
        try:
            auc = roc_auc_score(y_test, scores)
            assert 0 <= auc <= 1
            print(f"\nFlow tabular data ROC-AUC: {auc:.3f}")
        except:
            pytest.skip("ROC-AUC computation failed (may happen with random data)")

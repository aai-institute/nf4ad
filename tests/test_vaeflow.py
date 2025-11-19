"""
Unit tests for VAEFlow components.
"""
import pytest
import torch
import numpy as np
from nf4ad.vaeflow import (
    VAEFlow,
    PreTrainedEncoder,
    SimpleDecoder,
    VAEFlowEvaluator,
)


class TestPreTrainedEncoder:
    """Tests for PreTrainedEncoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = PreTrainedEncoder(
            arch='resnet18',
            latent_dim=64,
            pretrained=False,
            trainable=False,
        )
        assert encoder is not None
        assert hasattr(encoder, 'encoder')
        assert hasattr(encoder, 'fc_mu')
        assert hasattr(encoder, 'fc_logvar')
    
    def test_forward_pass(self):
        """Test encoder forward pass."""
        encoder = PreTrainedEncoder(arch='resnet18', latent_dim=64, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        mu, logvar = encoder(x)
        
        assert mu.shape == (2, 64)
        assert logvar.shape == (2, 64)
    
    def test_trainable_parameter(self):
        """Test trainable parameter setting."""
        encoder_frozen = PreTrainedEncoder(arch='resnet18', latent_dim=64, trainable=False)
        encoder_trainable = PreTrainedEncoder(arch='resnet18', latent_dim=64, trainable=True)
        
        # Check frozen encoder
        for param in encoder_frozen.encoder.parameters():
            assert not param.requires_grad
        
        # Check trainable encoder
        trainable_params = sum(p.requires_grad for p in encoder_trainable.encoder.parameters())
        assert trainable_params > 0


class TestSimpleDecoder:
    """Tests for SimpleDecoder."""
    
    def test_initialization(self):
        """Test decoder initialization."""
        decoder = SimpleDecoder(
            latent_dim=64,
            output_channels=3,
            output_size=(224, 224),
        )
        assert decoder is not None
    
    def test_forward_pass(self):
        """Test decoder forward pass."""
        decoder = SimpleDecoder(latent_dim=64, output_size=(64, 64))
        z = torch.randn(4, 64)
        x_recon = decoder(z)
        
        assert x_recon.shape == (4, 3, 64, 64)
        assert torch.all((x_recon >= 0) & (x_recon <= 1))  # Sigmoid output
    
    def test_various_output_sizes(self):
        """Test decoder with different output sizes."""
        sizes = [(32, 32), (64, 64), (128, 128), (100, 100)]
        
        for size in sizes:
            decoder = SimpleDecoder(latent_dim=32, output_size=size)
            z = torch.randn(2, 32)
            x_recon = decoder(z)
            assert x_recon.shape[-2:] == size


class TestVAEFlow:
    """Tests for VAEFlow model."""
    
    def test_initialization(self, simple_flow_prior, device):
        """Test VAEFlow initialization."""
        model = VAEFlow(
            flow_prior=simple_flow_prior,
            latent_dim=32,
            encoder_arch='resnet18',
            encoder_trainable=False,
        )
        assert model is not None
        assert model.latent_dim == 32
    
    def test_forward_pass(self, simple_flow_prior, device):
        """Test VAEFlow forward pass."""
        model = VAEFlow(
            flow_prior=simple_flow_prior,
            latent_dim=32,
            encoder_arch='resnet18',
            encoder_trainable=False,
        ).to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        x_recon, mu, logvar = model(x)
        
        assert x_recon.shape == x.shape
        assert mu.shape == (2, 32)
        assert logvar.shape == (2, 32)
    
    def test_reparameterize(self, simple_flow_prior):
        """Test reparameterization trick."""
        model = VAEFlow(flow_prior=simple_flow_prior, latent_dim=32)
        
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)
        z = model.reparameterize(mu, logvar)
        
        assert z.shape == (4, 32)
        # With mu=0, logvar=0, z should be ~ N(0,1)
        assert z.abs().mean() < 2.0  # Rough check
    
    def test_loss_function(self, simple_flow_prior, device):
        """Test loss computation."""
        model = VAEFlow(
            flow_prior=simple_flow_prior,
            latent_dim=32,
        ).to(device)
        
        x = torch.randn(4, 3, 224, 224).to(device)
        x_recon, mu, logvar = model(x)
        z = model.reparameterize(mu, logvar)
        
        loss = model.loss_function(x, x_recon, mu, logvar, z, beta=1.0)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_sample(self, simple_flow_prior, device):
        """Test sampling from model."""
        model = VAEFlow(
            flow_prior=simple_flow_prior,
            latent_dim=32,
        ).to(device)
        
        samples = model.sample(n_samples=5)
        assert samples.shape[0] == 5
        assert samples.shape[1:] == (3, 224, 224)
    
    def test_anomaly_score(self, simple_flow_prior, device):
        """Test anomaly score computation."""
        model = VAEFlow(
            flow_prior=simple_flow_prior,
            latent_dim=32,
        ).to(device)
        
        x = torch.randn(3, 3, 224, 224).to(device)
        scores = model.anomaly_score(x)
        
        assert scores.shape == (3,)
        assert torch.all(torch.isfinite(scores))
    
    def test_fit_method(self, simple_flow_prior, device):
        """Test model fitting."""
        # Create dataset with correct input size for default decoder (224x224)
        torch.manual_seed(42)
        X = torch.randn(50, 3, 224, 224)
        y = torch.zeros(50)
        from torch.utils.data import TensorDataset
        small_dataset = TensorDataset(X, y)
        
        model = VAEFlow(
            flow_prior=simple_flow_prior,
            latent_dim=32,
        ).to(device)
        
        losses = model.fit(
            data_train=small_dataset,
            batch_size=10,
            epochs=2,
            device=device,
        )
        
        assert len(losses) == 2
        assert all(isinstance(l, float) for l in losses)
    
    def test_elbo_computation(self, simple_flow_prior, device):
        """Test ELBO estimation."""
        model = VAEFlow(
            flow_prior=simple_flow_prior,
            latent_dim=32,
        ).to(device)
        
        x = torch.randn(5, 3, 224, 224).to(device)
        
        # Test different reductions
        elbo_mean = model.elbo(x, nsamples=2, reduction='mean', device=device)
        elbo_sum = model.elbo(x, nsamples=2, reduction='sum', device=device)
        elbo_none = model.elbo(x, nsamples=2, reduction='none', device=device)
        
        assert elbo_mean.dim() == 0  # Scalar
        assert elbo_sum.dim() == 0  # Scalar
        assert elbo_none.shape == (5,)  # Per-sample


class TestVAEFlowEvaluator:
    """Tests for VAEFlowEvaluator."""
    
    def test_initialization(self, simple_flow_prior, small_dataset, device):
        """Test evaluator initialization."""
        model = VAEFlow(flow_prior=simple_flow_prior, latent_dim=32).to(device)
        
        evaluator = VAEFlowEvaluator(
            model=model,
            dataset=small_dataset,
            device=device,
            batch_size=10,
        )
        
        assert evaluator is not None
        assert evaluator.model == model
    
    def test_scores_and_labels(self, simple_flow_prior, device):
        """Test score computation with labels."""
        model = VAEFlow(flow_prior=simple_flow_prior, latent_dim=32).to(device)
        
        # Create simple dataset with labels
        X = torch.randn(30, 3, 224, 224)
        y = torch.cat([torch.zeros(15), torch.ones(15)])
        
        class SimpleDataset:
            def __len__(self):
                return len(X)
            def __getitem__(self, idx):
                return X[idx], int(y[idx])
        
        dataset = SimpleDataset()
        evaluator = VAEFlowEvaluator(model, dataset, device=device, batch_size=10)
        
        scores, labels = evaluator.scores_and_labels()
        
        assert len(scores) == 30
        assert len(labels) == 30
        assert set(labels) == {0, 1}
    
    def test_compute_roc_pr(self, simple_flow_prior, device):
        """Test ROC and PR AUC computation."""
        model = VAEFlow(flow_prior=simple_flow_prior, latent_dim=32).to(device)
        
        X = torch.randn(40, 3, 224, 224)
        y = torch.cat([torch.zeros(20), torch.ones(20)])
        
        class SimpleDataset:
            def __len__(self):
                return len(X)
            def __getitem__(self, idx):
                return X[idx], int(y[idx])
        
        dataset = SimpleDataset()
        evaluator = VAEFlowEvaluator(model, dataset, device=device)
        
        roc_auc, pr_auc = evaluator.compute_roc_pr(return_curve=False)
        
        assert 0 <= roc_auc <= 1 or np.isnan(roc_auc)
        assert 0 <= pr_auc <= 1 or np.isnan(pr_auc)
    
    def test_summary(self, simple_flow_prior, device):
        """Test summary metrics."""
        model = VAEFlow(flow_prior=simple_flow_prior, latent_dim=32).to(device)
        
        X = torch.randn(50, 3, 224, 224)
        y = torch.cat([torch.zeros(25), torch.ones(25)])
        
        class SimpleDataset:
            def __len__(self):
                return len(X)
            def __getitem__(self, idx):
                return X[idx], int(y[idx])
        
        dataset = SimpleDataset()
        evaluator = VAEFlowEvaluator(model, dataset, device=device)
        
        summary = evaluator.summary()
        
        assert 'roc_auc' in summary
        assert 'pr_auc' in summary
        assert 'best_f1' in summary
        assert 'score_type' in summary

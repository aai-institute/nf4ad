"""
Tests for flow components.
"""
import pytest
import torch
import torch.nn as nn
from nf4ad.flows import NonUSFlow
import pyro.distributions as dist


class TestNonUSFlow:
    """Basic tests for NonUSFlow."""
    
    def test_initialization(self, device):
        """Test flow initialization."""
        latent_dim = 32
        base_dist = dist.Normal(
            torch.zeros(latent_dim).to(device),
            torch.ones(latent_dim).to(device)
        )
        
        # Simple conditioner for testing
        class SimpleConditioner(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, out_dim),
                )
            
            def forward(self, x):
                return self.net(x)
        
        flow = NonUSFlow(
            in_dims=[latent_dim],
            device=device,
            coupling_blocks=5,
            base_distribution=base_dist,
            conditioner_cls=SimpleConditioner,
            conditioner_args={
                'in_dim': latent_dim,
                'out_dim': latent_dim * 2,
            },
            nonlinearity=nn.ReLU(),
        )
        
        assert flow is not None
    
    def test_sample(self, simple_flow_prior, device):
        """Test sampling from flow."""
        samples = simple_flow_prior.sample([10])
        
        assert samples.shape[0] == 10
        assert samples.device.type == device.type
    
    def test_log_prob(self, simple_flow_prior, device):
        """Test log probability computation."""
        z = torch.randn(5, 32).to(device)
        log_prob = simple_flow_prior.log_prob(z)
        
        assert log_prob.shape in [(5,), ()]  # Per-sample or total
        assert torch.all(torch.isfinite(log_prob))

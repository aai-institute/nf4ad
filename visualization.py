from typing import Dict, Iterable, Literal
from matplotlib import pyplot as plt
import numpy as np
from src.explib import datasets
import torch

from src.veriflow.flows import Flow
from src.explib.visualization import norm
from src.nf4ad.feature_encoder import FeatureEncoder, feature_encoder_transform

Norm = Literal[-1, 1, 2]
SampleType = Literal["conditional", "boundary", "boundary_basis"]

class FakeModel(torch.nn.Module):
    """A fake model that samples from a dataset.
    
    Args:
        dataset: The dataset to sample from.
    """
    def __init__(self, dataset: datasets, feature_encoder: FeatureEncoder):
        super().__init__()
        self.dataset = dataset
        self.n = len(dataset)
        self.feature_encoder = feature_encoder
    
    def sample(self, shape):

        """Samples from the dataset.
        
        Args:
            shape: The shape of the samples.
        
        Returns:
            A tensor of shape `shape`.
        """
        data = self.dataset[np.random.choice(self.n, shape)][0]
        data = feature_encoder_transform(data).unsqueeze(1)
         
        data = self.feature_encoder(feature_encoder_transform(data)) 
        print(f"DATA SHAPE {data.shape}")
        return data


def latent_radial_qqplot(models: Dict[str, Flow], data: datasets, p, n_samples, save_to=None):
    """Plots a QQ-plot of the empirical and theoretical distribution of the L_p norm of the latent variables.
    
    Args:
        model: The model to visualize.
        p: The norm to use.
        n_samples: The number of samples to draw from the base distribution.
        n_bins: The number of bins to use in the histogram.
        save_to: If not None, the plot is saved to this path.
    """
    fakemodels = {}
    for name, model in models.items():
        fakemodels[name] = FakeModel(data, model.feature_encoder)
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel("Latent radial quantiles of the true distribution under the model")
    ax.set_ylabel("Latent radial quantiles of the learned distribution under the model")
    curves = {"Optimal": plt.plot([0, 1], [0, 1])}
    for name, model in models.items():
        true_samples = fakemodels[name].sample((n_samples,))
        learned_samples =  model.flow.sample((n_samples,))
        model.flow.export = "backward"

        true_latent_norms = norm(model.flow.forward(true_samples), p).sort()[0].detach()
        learned_latent_norms = norm(model.flow.forward(learned_samples), p).sort()[0].detach()

        def cdf(r, samples):
            return (samples <= r).sum()/samples.shape[0]

        tqs = [cdf(r, true_latent_norms).detach() for r in true_latent_norms] 
        lqs = [cdf(r, learned_latent_norms).detach() for r in true_latent_norms] 


        curves[name] = ax.plot([0.] + list(tqs) + [1.], [0.] + list(lqs) + [1.])
    
    plt.legend(curves.keys())
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
    plt.show()
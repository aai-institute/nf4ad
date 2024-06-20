from typing import Dict, Iterable, Literal
from matplotlib import pyplot as plt
import numpy as np
from src.explib import datasets
import math
import torch 
import torchvision
from src.veriflow.flows import Flow
from src.veriflow.distributions import RadialDistribution
from src.explib.visualization import norm, FakeModel
from src.nf4ad.feature_encoder import FeatureEncoder, feature_encoder_transform
from src.nf4ad.flows import FeatureFlow
Norm = Literal[-1, 1, 2]
SampleType = Literal["conditional", "boundary", "boundary_basis"]

class FakeModelWithFeatureEncoder(torch.nn.Module):
    """A fake model that samples from a dataset.
    
    Args:
        dataset: The dataset to sample from.
    """
    def __init__(self, dataset: datasets, feature_encoder: FeatureEncoder, device: torch.device):
        super().__init__()
        self.dataset = dataset
        self.n = len(dataset)
        self.feature_encoder = feature_encoder
        self.device = device
    
    def sample(self, shape):

        """Samples from the dataset.
        
        Args:
            shape: The shape of the samples.
        
        Returns:
            A tensor of shape `shape`.
        """
 
        data = self.dataset[np.random.choice(self.n, shape)][0]
        # TODO: understand how to pass the digit number (and image shape) to find the right min max values for normalization. need to change something in the base class fit module
        data = data.reshape(data.shape[0], 28, 28).unsqueeze(1).to(self.device)
        data = self.feature_encoder(feature_encoder_transform(data, digit=3))
             
        return data


def latent_radial_qqplot(models: Dict[str, Flow], data: datasets, n_samples, save_to=None):
    """Plots a QQ-plot of the empirical and theoretical distribution of the L_p norm of the latent variables.
    
    Args:
        model: The model to visualize.
        n_samples: The number of samples to draw from the base distribution.
        n_bins: The number of bins to use in the histogram.
        save_to: If not None, the plot is saved to this path.
    """
    
    # This fakemodel will be used only for Flow models to generate true samples
    fakemodel = FakeModel(data)
    true_samples = fakemodel.sample((n_samples,))
     
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel("Latent radial quantiles of the true distribution under the model")
    ax.set_ylabel("Latent radial quantiles of the learned distribution under the model")
    curves = {"Optimal": plt.plot([0, 1], [0, 1])}
    for name, model in models.items():
        
        if isinstance(model.base_distribution, RadialDistribution):
            p = model.base_distribution.p
        else:
            p = 1 
        if isinstance(model, FeatureFlow):
            fakemodel = FakeModelWithFeatureEncoder(data, model.feature_encoder, model.device)
            true_samples = fakemodel.sample((n_samples,))

            learned_samples =  model.flow.sample((n_samples,))
             
            model.flow.export = "backward"
            true_latent_norms = norm(model.flow.forward(true_samples.to(model.device)), p).sort()[0].cpu().detach()
            learned_latent_norms = norm(model.flow.forward(learned_samples), p).sort()[0].cpu().detach()
        else:
            learned_samples =  model.sample((n_samples,))
            model.export = "backward"  
            true_latent_norms = norm(model.forward(true_samples.to(model.device)), p).sort()[0].cpu().detach()
            learned_latent_norms = norm(model.forward(learned_samples), p).sort()[0].cpu().detach()
            
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
    
def show_imgs(imgs, saveto=None, title=None, row_size=10): # TODO: decide default row_size and pass it from the main code based on number of samples
     
    # Form a grid of pictures (we use max. 8 columns)
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
   
    is_int = imgs.dtype==torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype==torch.int32
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
   
    np_imgs = imgs.cpu().numpy()

    # Plot the grid
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2, 0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    
    if saveto:
        print(saveto + title)
        plt.savefig(saveto + title)
    
    plt.show()
    plt.close()
    
    return np.transpose(np_imgs, (1,2, 0))

# TODO: plot_digits exists in USFlows/explib. However the nsample function is not defined anywhere. Needs to understand it.
# For the moment we use the following function (that can be moved in the nf4d/visualization module)
def plot_digits(models: dict[str, Flow], n_samples=100, im_shape=(28, 28), save_to=None): #sqrtn: int
    """ Plot the samples from the models in a grid.
    
    Args:
         # TODO
    """
    with torch.no_grad():
            
        ncols = len(models)
        nrows = 1 # todo: do the sqrt
        figsize = (7 * ncols, 25)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
             
        for exp, ax in zip(models.keys(), axes.T):
              
            model = models[exp]
            
            samples = model.sample(sample_shape=[n_samples]).cpu().detach().numpy()  
            # TODO: remove comment once we fix import of nf4ad module
            if 0: #isinstance(model, FeatureFlow):
                samples = samples.squeeze()
            else:
                samples = samples.reshape(-1, *im_shape)

            samples = np.uint8(np.clip(samples, 0, 1) * 255)
            imgs_to_show = show_imgs(torch.tensor(samples).unsqueeze(1))
       
            ax.imshow(imgs_to_show, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(exp, fontsize=20)
        plt.tight_layout()
        if save_to:
            plt.savefig(save_to)
        plt.show()
 
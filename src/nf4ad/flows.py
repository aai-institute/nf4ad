from typing import Any, Dict, Union, Tuple, Optional, Iterable
import logging
import os
from src.veriflow.flows import Flow
from src.explib.config_parser import from_checkpoint
import torch  
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn import Module
import numpy as np
import torchvision.transforms as transforms
from nf4ad.feature_encoder import FeatureEncoder, PretrainedEncoder, PretrainedDecoder, MeanNet, StdNet, feature_encoder_transform
import pyro 
from pyro.infer import SVI
from pyro.optim import Adam 
from src.veriflow.flows import Flow
from pyro import distributions as dist

class FeatureFlow(Flow):
    def __init__(
        self, 
        flow: Flow, 
        pretrained_feature_encoder: str
    ):
        """Implements a flow on the latent space of a feature extractor.
        Both models are trained jointly by the fit method. Supports 
        loading and refinement of pretrained feature extractors.
        
        Args:
            flow (Flow): The flow to be applied on the latent space.
            pretrained_feature_encoder (str): The feature
                extractor checkpoint to be used.  
        """
        super(FeatureFlow, self).__init__(flow.base_distribution, flow.layers, flow.soft_training, flow.training_noise_prior)
        self.flow = flow
        self.feature_encoder = FeatureEncoder() 
        
        checkpoint = torch.load(os.path.abspath(pretrained_feature_encoder))
        self.feature_encoder.load_state_dict(checkpoint["ae_net_dict"])
        
        self.trainable_layers = torch.nn.ModuleList(
            [l for l in flow.layers if isinstance(l, torch.nn.Module)] + 
            [fe_l for _, fe_l in self.feature_encoder.named_modules() if isinstance(fe_l, torch.nn.Module)]
        )
        
    def log_prob(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes the log probability of self.feature_encoder(x).
        
        Args:
            x (Tensor): The batch of samples.
            digit (int): Normal class.
        Returns:
            Tensor: The log probability of the samples.
        """ 
        # TODO: Image dimension (28x28) and digit label (3) are hardcoded. Change this!
        x = x.reshape(x.shape[0], 28, 28).unsqueeze(1)
        z = self.feature_encoder(feature_encoder_transform(x, digit=3))
     
        return self.flow.log_prob(z.reshape(z.shape[0], -1))
    
    def to(self, device: torch.device):
        """Moves the model to the specified device.
        
        Args:
            device (torch.device): The device to move the model to.
        
        Returns:
            FeatureFlow: The model.
        """
        self.device = device
        self.flow.to(device)
        self.feature_encoder.to(device) 
        return self 

    def sample(
        self, sample_shape: Iterable[int] = None, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns n_sample samples from the distribution

        Args:
            n_sample: sample shape.
        """
        if sample_shape is None:
            sample_shape = [1]

        y = self.base_distribution.sample(sample_shape)
        
        for layer in self.layers:
            if context is not None:
                y = layer.forward(y, context=context)
            else:
                y = layer.forward(y)
 
       
        y = self.feature_encoder.reconstruct(y)
      
        return y

class Encoder(torch.nn.Module):
    def __init__(
        self, 
        dim: Optional[int] = None,
        z_dim: Optional[int] = None, 
        hidden_dims: Optional[list[int]] = None,
        hidden_net: Optional[Union[torch.nn.Module, PretrainedEncoder, PretrainedDecoder] ] = None,
        mean_net: Optional[Union[torch.nn.Module, MeanNet]] = None,
        std_net: Optional[torch.nn.Module] = None, 
        nonlinearity = torch.nn.Softplus()
        ):
        
    
        super().__init__()
        if hidden_net:
            # Using an hard coded architecture to use Tim's pretrained weights 
            self.hidden_net = hidden_net
            self.mean_net = mean_net # fc1 from Tim
            # TODO: this is done just for having the code run. The std net should be initialize with a fc layer of all zeros
            if self.std_net is None:
                self.std_net = StdNet()
            else:
                self.std_net = std_net 
                # Tim's network has only one last layer for the mean. We would need to add one (init with all zeros) for std.
        else:
            layers = []
            layers.append(torch.nn.Linear(dim, hidden_dims[0]))
            layers.append(nonlinearity)
            for i in range(len(hidden_dims) - 1):
                layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(nonlinearity)

            self.hidden_net = torch.nn.Sequential(*layers)

            self.mean_net = torch.nn.Linear(hidden_dims[-1], z_dim)
            self.std_net = torch.nn.Linear(hidden_dims[-1], z_dim)
 

            
    def forward(self, x):
        hidden = self.hidden_net(x)
        mean = self.mean_net(hidden)
        relu = torch.nn.ReLU()
        std = relu(self.std_net(hidden)) + 1e-6
        
        return mean, std

class Decoder(torch.nn.Module):
     
    def __init__(
        self, 
        dim: Optional[int] = None,
        z_dim: Optional[int] = None, 
        hidden_dims: Optional[list[int]] = None,
        net: Optional[torch.nn.Module] = None,
        nonlinearity = torch.nn.Softplus(),
        ):
        
        super().__init__()
        # TODO: do we want to use the pretrained weights also for the decoder? Only the mean we need at the beginning right?
        if net:
            # Using an hard coded architecture to use Tim's pretrained weights 
            self.net = net
        else:
            layers = []
            layers.append(torch.nn.Linear(z_dim, hidden_dims[-1]))
            layers.append(nonlinearity)
            for i in reversed(range(1, len(hidden_dims))):
                layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i-1]))
                layers.append(nonlinearity)
                
            layers.append(torch.nn.Linear(hidden_dims[0], dim))
            self.net = torch.nn.Sequential(*layers)
         
            
    def forward(self, x):
        x = self.net(x) 
 
        activation = torch.nn.Sigmoid()
        x = activation(x)

        return x
    
class LatentFlow():
    
    def __init__(self, flow: Flow, encoder: Union[Encoder, str], decoder: Union[Decoder, str], mean_net: Optional[Union[MeanNet, str]] = None):
         
        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            self.encoder = Encoder(hidden_net=torch.load(encoder), mean_net=torch.load(mean_net))
            
        if isinstance(decoder, Decoder):
            self.decoder = decoder
        else:
           self.decoder = Decoder(net=torch.load(decoder))
           
        self.flow = flow
            
    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # (From pyro tutorial) setup hyperparameters for prior p(z)
            # z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            # z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            
            z = pyro.sample("latent", self.flow)
            # decode the latent code z
            loc_img = self.decoder(z)

            # TODO: had to put the workaround "validate_args=False" otherwise I get an error on data range [0, 1]. Check this!
            pyro.sample("obs", dist.Bernoulli(loc_img, validate_args=False).to_event(1), obs=x.reshape(-1, 784))
            
    # (From pyro tutorial) define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            # sample the latent code z
            lambda_dist = lambda loc, scale: dist.Normal(loc, scale) 
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
     # (From pyro tutorial) define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
    
    def to(self, device) -> None:
        
        self.device = device
        
        self.encoder.to(device)
        self.decoder.to(device)
        self.flow.to(device)
        
        return device
        
    def fit(self, data_train: Dataset,
            svi: SVI, 
            batch_size: int = 32,
            shuffle: bool = True,
            gradient_clip: float = True,
            device: torch.device = None,
            epochs: int = 1):
        
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
 
        model = self.to(device)
        
        N = len(data_train)
        epoch_losses = []
        for _ in range(epochs):
            losses = []
            if shuffle:
                perm = np.random.choice(N, N, replace=False)
                data_train_shuffle = data_train[perm][0]

            for idx in range(0, N, batch_size):
                end = min(idx + batch_size, N)
                try:
                    sample = torch.Tensor(data_train_shuffle[idx:end]).to(device)
                except:
                    continue

                # TODO: Using svi step this is not needed anymore. Check it!
                # optimizer.zero_grad()

                loss = svi.step(sample)
                losses.append(loss)
                
                # TODO: check if this is still needed here 
                # if gradient_clip is not None:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # TODO: related to TODO in line 278. Should not be needed anymore
                #optim.step()
                
                # TODO: for the moment LatentFlow doesn't inherit from Flow --> therefore this method doesn't exist. 
                # Check if we need to implement it also for the LatentFlow
                # if not self.is_feasible():
                #     raise RuntimeError("Model is not invertible")

                
            epoch_losses.append(np.mean(losses))

        return epoch_losses
        
    def evaluate(self, dataset: Dataset, svi: SVI, batch_size: int = 32):
        # TODO: check if we need to pass the device or if data and model already on the right device
         
        loss = 0.
        for i in range(0, len(dataset), batch_size):
            j = min([len(dataset), i + batch_size])
            x = dataset[i:j][0]
            loss += svi.evaluate_loss(x)
        
        loss /= len(dataset)
        return loss
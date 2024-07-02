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
from nf4ad.feature_encoder import FeatureEncoder, feature_encoder_transform

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
        
     
        # TODO: check how to change the way trainable layers is created. Here needs to be flow layers + feature encoder layer
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
        # TODO: understand how to pass the digit number to find the right min max values for normalization. need to change something in the base class fit module
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
        hidden_dims: Optional[List[int]] = None,
        hidden_net: Optional[torch.nn.Module] = None,
        mean_net: Optional[torch.nn.Module] = None,
        std_net: Optional[torch.nn.Module] = None
        nonlinearity = torch.nn.Softplus()
        ):
        
        if hidden_net:
            # Using an hard coded architecture to use Tim's pretrained weights 
            self.hidden_net = hidden_net
            self.mean_net = mean_net #fc1 from Tim
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


        super().__init__()
        
            
    def forward(self, x):
        hidden = self.hidden_net(x)
        mean = self.mean_net(hidden)
        relu = torch.nn.relu
        std = relu(self.std_net(hidden))

        return mean, std

class Decoder(torch.nn.module):
    pass

class LatentFlow(Flow):
    
    def __init__(self, encoder: Union[Encoder, str], decoder: Union[Decoder, str]):
        
        if isinstance(encoder, Encoder):
            self.encoder = encoder
        else:
            self. = FeatureEncoder() 
        
            checkpoint = torch.load(os.path.abspath(pretrained_feature_encoder))
            self.feature_encoder.load_state_dict(checkpoint["ae_net_dict"])
            
            
        

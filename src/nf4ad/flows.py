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
from src.nf4ad.feature_encoder import FeatureEncoder, feature_encoder_transform

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


    
    # def fit(
    #         self,
    #         data_train: Dataset,
    #         optim: torch.optim.Optimizer = torch.optim.Adam,
    #         optim_params: Dict[str, Any] = None,
    #         batch_size: int = 32,
    #         shuffle: bool = True,
    #         gradient_clip: float = None,
    #         device: torch.device = None,
    #         epochs: int = 1,
    #     ) -> float:
    #         """
    #         Wrapper function for the fitting procedure. Allows basic configuration of the optimizer and other
    #         fitting parameters.

    #         Args:
    #             data_train: training data.
    #             batch_size: number of samples per optimization step.
    #             optim: optimizer class.
    #             optimizer_params: optimizer parameter dictionary.
    #             jitter: Determines the amount of jitter that is added if the optimization leaves the feasible region.
    #             epochs: number of epochs.

    #         Returns:
    #             Loss curve .
    #         """
    #         if device is None:
    #             if torch.backends.mps.is_available():
    #                 device = torch.device("mps")
    #             elif torch.cuda.is_available():
    #                 device = torch.device("cuda:0")
    #             else:
    #                 device = torch.device("cpu")

    #         model = self.to(device)
    #         print(model)

    #         if optim_params is not None:
    #             optim = optim(model.parameters(), **optim_params) 
    #         else:
    #             optim = optim(model.parameters())

    #         # for param in model.parameters():
    #         #     if param.requires_grad:
    #         #         print(f"MODEL PARAM {param}")
    #         N = len(data_train)

    #         epoch_losses = []
    #         for _ in range(epochs):
    #             losses = []
    #             if shuffle:
    #                 perm = np.random.choice(N, N, replace=False)
    #                 data_train_shuffle = data_train[perm][0]

    #             print(f"DATA TRAIN SHAPE {data_train_shuffle.shape}")
    #             for idx in range(0, N, batch_size):
    #                 end = min(idx + batch_size, N)
    #                 try:
    #                     sample_flatten = torch.Tensor(data_train_shuffle[idx:end]).to(device) # batch of 32x784 (img is an array 28x28 = 784)
    #                 except:
    #                     continue
 
    #                 # TODO: 28 is hardcoded. maybe put in the config mnist shape (or check if it is in the dataset field)
    #                 sample_reshaped = sample_flatten.reshape(sample_flatten.shape[0], 28, 28).unsqueeze(1)
    #                 # FIXME: remove hardcoded values
    #                 #sample_feat = self.feature_encoder(feature_encoder_transform(sample_reshaped, min_max=(-0.7645772083211267, 12.895051191467457)))
    #                 sample_feat = self.feature_encoder(sample_reshaped)
                    
    #                 sample = sample_feat.reshape(sample_feat.shape[0], -1)
                    
    #                 if self.soft_training:
    #                     noise = self.training_noise_prior.sample([sample.shape[0]]) 

    #                     # Repeat noise for all data dimensions
    #                     sigma = noise
    #                     r = torch.Tensor(list(sample.shape[1:])).prod().int()
    #                     sigma = sigma.repeat_interleave(r)
    #                     sigma = sigma.reshape(sample.shape)

    #                     e = torch.normal(torch.zeros_like(sigma), sigma).to(device)
    #                     sample = sample + e
    #                     noise = noise.unsqueeze(-1)
    #                     noise = noise.detach().to(device)
    #                     # scaling of noise for the conditioning recommended by SoftFlow paper
    #                     noise = noise * 2/self.training_noise_prior.high
    #                 else:
    #                     noise = None

    #                 optim.zero_grad()

    #                 loss = -model.log_prob(
    #                     sample_reshaped #, context=noise
    #                 ).mean() - model.flow.log_prior()
    #                 loss.backward()
    #                 losses.append(float(loss.detach()))
    #                 if gradient_clip is not None:
    #                     torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    #                 optim.step()
    #                 if not self.is_feasible():
    #                     raise RuntimeError("Model is not invertible")

    #                 model.flow.transform.clear_cache()
    #             epoch_losses.append(np.mean(losses))

    #         return epoch_losses

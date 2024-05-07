from src.veriflow.flows import Flow
from src.explib.config_parser import from_checkpoint
from torch.nn import Module

class FeatureFlow(Flow):
    def __init__(
        self, 
        flow: Flow, 
        feature_encoder: Union[Module, Tuple[str, str]]
    ):
        """Implements a flow on the latent space of a feature extractor.
        Both models are trained jointly by the fit method. Supports 
        loading and refinement of pretrained feature extractors.
        
        Args:
            flow (Flow): The flow to be applied on the latent space.
            feature_encoder (Union[Module, Tuple[str, str]]): The feature
                extractor to be used. If a tuple, it is assumed to be a
                (model_arch, model_state_dict) pair.
        """
        super(FeatureFlow, self).__init__()
        self.flow = flow
        
        if isinstance(feature_encoder, tuple):
            model_arch, state_dict = feature_encoder
            feature_encoder = from_checkpoint(model_arch, state_dict)
            
        self.feature_encoder = feature_encoder
        
    def log_prob(self, x):
        """Computes the log probability of self.feature_encoder(x).
        
        Args:
            x (Tensor): The batch of samples.
        
        Returns:
            Tensor: The log probability of the samples.
        """
        z = self.feature_encoder(x)
        return self.flow.log_prob(z)
    
    def to(self, device: torch.device):
        """Moves the model to the specified device.
        
        Args:
            device (torch.device): The device to move the model to.
        
        Returns:
            FeatureFlow: The model.
        """
        self.flow.to(device)
        self.feature_encoder.to(device)
        return self 
        

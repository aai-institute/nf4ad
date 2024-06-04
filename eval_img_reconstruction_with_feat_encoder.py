
import os
from typing import Any, Dict
import math
import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from src.explib.visualization import latent_radial_qqplot
from src.nf4ad.flows import FeatureFlow
from visualization import latent_radial_qqplot as latent_radial_qqplot_feat_encoder, show_imgs
from src.nf4ad.feature_encoder import FeatureEncoder, feature_encoder_transform

# TODO: class to evaluate pretrained feature encoder for image reconstruction.
# IDEA: evaluate the trained feat encoder with pretrained decoder for image reconstruction. (Need to load the featureflow model and use only the feat encoder from the ckpt)
class Evaluation():
    """Evaluation."""

    def __init__(
        self,
        config: Dict[str, Any]
    ) -> None:
        """Initialize hyperparameter optimization experiment.

        Args:
            config (Dict[str, Any]): configuration 
        """
        print(config)
        self.name = config["name"]
        self.config = config["experiment"]
        self.device = self.config["device"]
    

    def conduct(self, report_dir: os.PathLike, n_samples = 100):   #TODO check input arg for device
        """Run the evaluation experiment.

        Args:
            report_dir (os.PathLike): report directory
        """
          
        # Load test dataset 
        dataset = self.config["dataset"]
        data_test = dataset.get_test()
       
        # Load feature encoder model
        feature_encoder = FeatureEncoder()
         
        pretrained_feature_encoder = self.config["model_cfg"]["params"]["pretrained_feature_encoder"]
        checkpoint = torch.load(os.path.abspath(pretrained_feature_encoder))
        feature_encoder.load_state_dict(checkpoint["ae_net_dict"])
   
        
        for i in range(len(data_test[:2])):
            x = data_test[i][0] 
            x = x.reshape(28, 28).unsqueeze(0).unsqueeze(0)
            plt.imshow(x[0][0].detach().numpy())
            plt.savefig(f"./reports/original_{i}.png")
            z = feature_encoder(feature_encoder_transform(x, digit=3))
            print(f"Z {z}")
            out = feature_encoder.reconstruct(z)[0][0]
            plt.imshow(out.detach().numpy())
            plt.savefig(f"./reports/reconstructed_{i}.png")            
            
    
    
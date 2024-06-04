
import os
import typing as T
from typing import Any, Dict
import math
import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
# TODO: for the moment couldn't use functions from USFlows/explib. Need to modify them
#from src.explib.visualization import latent_radial_qqplot, plot_digits
from visualization import latent_radial_qqplot, plot_digits, show_imgs
from src.nf4ad.flows import FeatureFlow, Flow
from src.explib.base import Experiment
import pandas as pd 

# TODO: general. Input/output arguments type and documentations for class and method
       
class Evaluation():
    """Evaluation."""

    def __init__(
        self,
        experiments: T.Iterable[Experiment],
        device: torch.device = "cuda"
    ) -> None:
        """Initialize hyperparameter optimization experiment.

        Args:
            # TODO
        """
         
        self.experiments = experiments  
        self.device = device
    

    def conduct(self, report_dir: os.PathLike, n_samples = 100):   #TODO check input arg for device
        """Run the evaluation experiment.

        Args:
            report_dir (os.PathLike): report directory
        """
        
        sepline = "\n" + ("-" * 80) + "\n" + ("-" * 80) + "\n"
        
        # Load test dataset 
        dataset = self.experiments.experiments[0].trial_config["dataset"]
        data_test = dataset.get_test()
        
        models = {}
        losses = {}
        for experiment in self.experiments.experiments:
            
            # TODO: use a logger ? 
            print(f"Evaluating experiment: {experiment.name}")
            
            state_dict = torch.load(os.path.join(report_dir, f"0_{experiment.name}", "best_model.pt"), map_location=torch.device(self.device))
            
            model_hparams = experiment.trial_config["model_cfg"]["params"]
            model = experiment.trial_config["model_cfg"]["type"](**model_hparams)
            model.load_state_dict(state_dict)
            model.to(self.device)
            
            models[experiment.name] = model
            
            # Evaluate best model
            imgs, test_loss = self._test_best_model(model, data_test, n_samples, saveto=os.path.join(report_dir, f"0_{experiment.name}/"), title=f"img_samples_{experiment.name}.png")
            losses[experiment.name] = test_loss
            models[experiment.name] = model
        
        # Reconstruct images
        plot_digits(models, save_to=f"{report_dir}/samples_comparison.png")
        
        # QQplots
        self._qqplot(models, data_test, n_samples, saveto=f"{report_dir}/qqplots.png")
         
        # Test losses
        df = pd.DataFrame(losses, index=[0])
        print(f"{sepline}Test loss{sepline}\n{df}")
        
        ax = df.plot.bar()
        ax.figure.savefig(f"{report_dir}/test_losses.png")
       
       
    def _test_best_model(self, best_model, data, n_samples, im_shape=(28, 28), saveto=None, title=None):
         
        samples = best_model.sample(sample_shape=[n_samples]).cpu().detach().numpy()  
        if isinstance(best_model, FeatureFlow):
            samples = samples.squeeze()
        else:
            samples = samples.reshape(-1, *im_shape)

        samples = np.uint8(np.clip(samples, 0, 1) * 255)
        samples_grid = show_imgs(torch.tensor(samples).unsqueeze(1), saveto, title=title)
        
        # Compute the test loss
        test_loss = 0
        for i in range(0, len(data), 100):
            j = min([len(data), i + 100])
            test_loss += float(
                -best_model.log_prob(data[i:j][0].to(self.device)).sum()
            )
        test_loss /= len(data)
        
        return samples_grid, test_loss
        
    def _qqplot(self, models: dict[str, Flow], data, n_samples, p=1, saveto=None):
        
        curves = latent_radial_qqplot(models, data, p, n_samples, saveto)
        
        return curves
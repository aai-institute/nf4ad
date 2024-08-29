
import os
import typing as T
from typing import Any, Dict
import math
import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
# TODO: for the moment couldn't use functions from USFlows/explib. Need to modify them
#from src.explib.visualization import latent_radial_qqplot # using USFlows functions
from visualization import show_imgs, plot_digits, latent_radial_qqplot, norm_distributions # using nf4ad functions (adapted with the featureflow possibility)
from nf4ad.flows import FeatureFlow
from src.veriflow.flows import Flow
from src.veriflow.distributions import RadialDistribution
from src.explib.config_parser import from_checkpoint
from src.explib.base import Experiment
import pandas as pd 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve, auc
from itertools import cycle
import glob
import numpy as np
import pandas as pd 
import ast 
from torch.utils.data import Dataset
from pickle import load

class Evaluation():
    """Evaluation.
    
    Args:
        report_loc: location of the experiment reports. Expects that 
        each experiemnt is contained in a subdirectory.
        n_samples: number of samples to be used for the plots.
    """

    def __init__(
        self,
        report_loc: str,
        nominal: Dataset,
        anomaly: Dataset,
        name: str = "evaluation",
        n_samples: int = 100,
        device: str = "cuda"
        
    ) -> None:
        
        self.report_loc = report_loc
        self.n_samples = n_samples
        self.device = device
        self.nominal = nominal
        self.anomaly = anomaly
        self.name = name
    

    def conduct(self, report_dir: os.PathLike, storage_path="/.") -> None:  
        """Run the evaluation experiment.

        Args:
            report_dir (os.PathLike): report directory for the evaluation 
        """
  
        # Select device
        device = self.device
        
        
        # Load nominal test dataset 
        nominal_data_test = self.nominal         
        ad_data_test = self.anomaly 
        
        experiments = [
            f for f in os.listdir(self.report_loc) if os.path.isdir(os.path.join(self.report_loc, f))
        ]
               
        models = {} 
        for i, experiment in enumerate(experiments):
            exp_name = experiment
            experiment = os.path.join(self.report_loc, experiment)

            print(f"Evaluating experiment{i}: {exp_name}") 
            
            try:
                ckpt_flow = sorted(glob.glob(
                    os.path.join(experiment,  "*best_model_flow.pt")
                ))[-1]
                ckpt_encoder = sorted(glob.glob(
                    os.path.join(experiment,  "*best_model_encoder.pt")
                ))[-1]
                ckpt_decoder = sorted(glob.glob(
                    os.path.join(experiment,  "*best_model_decoder.pt")
                ))[-1]
                config_pkl = sorted(glob.glob(
                    os.path.join(experiment, "*best_config.pkl")
                ))[-1] 
                
            except:
                print("Checkpoint not found. Experiment skipped.")
                continue
            
            spec = load(open(config_pkl, "rb"))["model_cfg"]
            # pre-instantiate subnetworks
            spec["params"]["encoder"] = spec["params"]["encoder"]["type"](
                **spec["params"]["encoder"]["params"]
            )
            spec["params"]["decoder"] = spec["params"]["decoder"]["type"](
                **spec["params"]["decoder"]["params"]
            )
            spec["params"]["flow"] = spec["params"]["flow"]["type"](
                **spec["params"]["flow"]["params"]
            )
            
            # init model
            model = spec["type"](**spec["params"])

            # Load pretrained weights
            state_dict_enc = torch.load(ckpt_encoder)
            model.encoder.load_state_dict(state_dict_enc)

            state_dict_dec = torch.load(ckpt_decoder)
            model.decoder.load_state_dict(state_dict_dec)

            state_dict_flow = torch.load(ckpt_flow)
            model.flow.load_state_dict(state_dict_flow)
            
            model.to(device)
            models[experiment] = model
      
        # Reconstruct images
        plot_digits(models, save_to=f"{report_dir}/samples_comparison.png")
        
        # QQplots
        self._qqplot(models, nominal_data_test, saveto=f"{report_dir}/qqplots.png")
        
        # Norm distributions
        # TODO: for all models. it is saving only the last one. Change plt layout
        norm_distributions(models, nominal_data_test, n_samples=self.n_samples, saveto=f"{report_dir}/kde.png")
        
        # Test losses
        losses = self._test_losses(models, nominal_data_test)
        df = pd.DataFrame(losses, index=[0]) 
        ax = df.plot.bar()
        ax.figure.savefig(f"{report_dir}/test_losses.png")
        
        # AD metrics
        self._compute_auc_score(models, ad_data_test, nominal_value=nominal_dataset.digit, device=device, saveto=f"{report_dir}/roc_auc.png")
            
       
    def _test_losses(self, models, data):

        losses : dict[str, float] = {}
        for name, model in models.items(): 
            test_loss = 0
            for i in range(0, len(data), 100):
                j = min([len(data), i + 100])
                test_loss += float(
                    -model.log_prob(data[i:j][0]).sum()
                )
            test_loss /= len(data)
            losses[name] = test_loss
        
        return losses
        
    def _qqplot(self, models: dict[str, Flow], data, saveto=None):
        
        curves = latent_radial_qqplot(models, data, self.n_samples, saveto)
        
        return curves
    
    def _compute_auc_score(self, models, data, nominal_value, device, saveto=None):
          
        labels = data.labels 
        if not isinstance(labels, np.ndarray):
            labels = labels.numpy()
        
        labels_1_vs_1 = labels.copy()
        labels_1_vs_1[labels == nominal_value] = 1
        labels_1_vs_1[labels != nominal_value] = 0
        
        label_binarizer = LabelBinarizer().fit(labels_1_vs_1)
        y_onehot_test = label_binarizer.transform(labels_1_vs_1)
         
        # Compute ROC curve and ROC area for each model
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for name, model in models.items():  
            scores = model.log_prob(data[:][0].to(device)).cpu().detach()
            fpr[name], tpr[name], _ = roc_curve(y_onehot_test, scores)
            roc_auc[name] = auc(fpr[name], tpr[name])

        all_fpr = np.unique(np.concatenate([fpr[name] for name in models.keys()]))
 
        mean_tpr = np.zeros_like(all_fpr)
        for name in models.keys():
            mean_tpr += np.interp(all_fpr, fpr[name], tpr[name])


        # Plot all ROC curves
        plt.figure()
        lw = 2
        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for name, color in zip(models.keys(), colors):
            plt.plot(
                fpr[name],
                tpr[name],
                color=color,
                lw=lw,
                label="ROC curve of model {0} (area = {1:0.2f})".format(name, roc_auc[name]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Some extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right")
        
        if saveto:
            plt.savefig(saveto)
 
        return  
        
   
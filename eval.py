
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
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_auc_score, roc_curve, auc
from itertools import cycle
import glob
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
         
        
        # Load nominal test dataset 
        nominal_dataset = self.experiments.experiments[0].trial_config["dataset"]
        nominal_data_test = nominal_dataset.get_test()
        
        dataset_type = type(nominal_dataset)
        ad_dataset = dataset_type()
        ad_data_test = ad_dataset.get_test() # TODO: evaluate only on test portion of Mnist?
        
        models = {} 
        for i, experiment in enumerate(self.experiments.experiments):
            
            print(f"Evaluating experiment: {experiment.name}") 
            
            ckpt = glob.glob(os.path.join(report_dir, f"{i}_{experiment.name}") + "/*best_model.pt")
            print(os.path.join(report_dir, f"{i}_{experiment.name}"))
            if len(ckpt) == 0:
                continue
            
            state_dict = torch.load(ckpt[0], map_location=torch.device(self.device))
            
            model_hparams = experiment.trial_config["model_cfg"]["params"]
            model = experiment.trial_config["model_cfg"]["type"](**model_hparams)
            model.load_state_dict(state_dict)
            model.to(self.device)
            
            models[experiment.name] = model
        
        # Reconstruct images
        plot_digits(models, save_to=f"{report_dir}/samples_comparison.png")
        
        # QQplots
        self._qqplot(models, nominal_data_test, n_samples, saveto=f"{report_dir}/qqplots.png")
         
        # Test losses
        losses = self._test_losses(models, nominal_data_test)
        df = pd.DataFrame(losses, index=[0]) 
        ax = df.plot.bar()
        ax.figure.savefig(f"{report_dir}/test_losses.png")
        
        # AD metrics
        self._compute_auc_score(models, ad_data_test, nominal_value=nominal_dataset.digit, saveto=f"{report_dir}/roc_auc.png")
       
       
    def _test_losses(self, models, data):

        losses : dict[str, float] = {}
        for name, model in models.items(): 
            test_loss = 0
            for i in range(0, len(data), 100):
                j = min([len(data), i + 100])
                test_loss += float(
                    -model.log_prob(data[i:j][0].to(self.device)).sum()
                )
            test_loss /= len(data)
            losses[name] = test_loss
        
        return losses
        
    def _qqplot(self, models: dict[str, Flow], data, n_samples, p=1, saveto=None):
        
        curves = latent_radial_qqplot(models, data, p, n_samples, saveto)
        
        return curves
    
    def _compute_auc_score(self, models, data, nominal_value, saveto=None):
         
        labels = data.labels 
        
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
            scores = model.log_prob(data[:][0].to(self.device)).cpu().detach()
            fpr[name], tpr[name], _ = roc_curve(y_onehot_test, scores)
            roc_auc[name] = auc(fpr[name], tpr[name])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[name] for name in models.keys()]))

        # Then interpolate all ROC curves at this points
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

        # for name, model in models.items():
        #     scores = model.log_prob(data[:][0].to(self.device)).cpu().detach()

        #     display = RocCurveDisplay.from_predictions(
        #         y_onehot_test,
        #         scores,
        #         name=f"name",
        #         color="darkorange",
        #         plot_chance_level=True,
        #     )
        #     o = display.ax_.set(
        #         xlabel="False Positive Rate",
        #         ylabel="True Positive Rate",
        #         title="One-vs-Rest ROC curves:\nNominal value vs non nominal values",
        #     )

        #     plt.savefig("./roc.png")
            
        #     micro_roc_auc_ovo = roc_auc_score(
        #         y_onehot_test,
        #         scores,
        #         multi_class="ovo",
        #         average="micro",
        #     )
                    
        #     macro_roc_auc_ovo = roc_auc_score(
        #         y_onehot_test,
        #         scores,
        #         multi_class="ovo",
        #         average="macro",
        #     )

        #     print(f"Micro-averaged One-vs-One ROC AUC score:\n{micro_roc_auc_ovo:.2f}")
        #     print(f"Macro-averaged One-vs-One ROC AUC score:\n{macro_roc_auc_ovo:.2f}")
        return  
        
   
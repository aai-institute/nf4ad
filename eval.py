
import os
 
from typing import Any, Dict
import math
import torch 
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from src.explib.visualization import latent_radial_qqplot

def show_imgs(imgs, title=None, row_size=4):
     
    # Form a grid of pictures (we use max. 8 columns)
    num_imgs = imgs.shape[0] if isinstance(imgs, torch.Tensor) else len(imgs)
   
    is_int = imgs.dtype==torch.int32 if isinstance(imgs, torch.Tensor) else imgs[0].dtype==torch.int32
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))

    imgs = torchvision.utils.make_grid(imgs, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
 
    # Plot the grid
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()
    
class Evaluation():
    """Hyperparameter optimization experiment."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu", 
    ) -> None:
        """Initialize hyperparameter optimization experiment.

        Args:
            config (Dict[str, Any]): configuration
            device: str: Torch device
        """
        self.name = config["name"]
        self.config = config["experiment"]
        self.device = device
    

    def conduct(self, report_dir: os.PathLike, device: torch.device = torch.device("cpu")):
        """Run the evaluation experiment.

        Args:
            report_dir (os.PathLike): report directory
        """
          
        # Load test dataset 
        dataset = self.config["dataset"]
        data_test = dataset.get_test()
        # imgs = [data_test[i][0].reshape(1, 28, 28) for i in range(8)]
        # show_imgs(imgs)
         
        # Load model
        state_dict = torch.load(os.path.join(report_dir, "best_model.pt"), map_location=torch.device(self.device))
        model_hparams = self.config["model_cfg"]["params"]
        model = self.config["model_cfg"]["type"](**model_hparams)
        model.load_state_dict(state_dict)
        
        # Evaluate best model
        self._test_best_model(model, data_test, device)
  
    
    # TODO: add type of input arguments
    def _test_best_model(self, best_model, data, device, im_shape=(28, 28)):
         
        # Visualize sampled image
        # sample = best_model.sample(sample_shape=[1]).cpu().detach().numpy()
        # np.save("sample.npy", sample)
        
        sample = np.load("sample.npy")
        
        sample = sample.reshape(*im_shape) #sample_shape=[1]
        sample = np.uint8(np.clip(sample, 0, 1) * 255)
         
        plt.imshow(sample, cmap="gray")
        plt.show()
    
        # Compute the test loss
        # test_loss = 0
        # for i in range(0, len(data), 100):
        #     j = min([len(data), i + 100])
        #     test_loss += float(
        #         -best_model.log_prob(data[i:j][0].to(device)).sum()
        #     )
        # test_loss /= len(data)
        
        # print(test_loss)
        
        latent_radial_qqplot({"best_model": best_model}, data, p=1, n_samples=4, save_to=None)
        
      
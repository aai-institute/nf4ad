import os
import sys
sys.path.append("/Users/giorgiapitteri/Desktop/Projects/NF4AD/Veriflow")
from src.explib.config_parser import read_config, from_checkpoint, unfold_raw_config
import matplotlib.pyplot as plt
import torch
import math
import torchvision
import numpy as np
from src.nf4ad.feature_encoder import FeatureEncoder, feature_encoder_transform

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
 
def main(config_path, ckpt_path, config):       
     
    experiment = read_config(os.path.abspath(config))
     
    # dataset = experiment.experiments[0].trial_config["dataset"]

    # data_test = dataset.get_test()
 
    # imgs = [data_test[i][0].reshape(1, 28, 28) for i in range(8)]

    #show_imgs(imgs)
    
    # # Load the training NICE Flow. Need to load also the  feature encoder if used 
    # feature_encoder = FeatureEncoder()
    # # TODO: understand how to get this config part
    # ckpt = "/Users/giorgiapitteri/Downloads/model.tar" 
    # checkpoint = torch.load(ckpt)
    # feature_encoder.load_state_dict(checkpoint["ae_net_dict"])
             
    # # TODO: min max hardcoded for digit 3.
    # imgs_encoder = [feature_encoder_transform(img.cpu().detach().numpy(), min_max=(-0.7645772083211267, 12.895051191467457)).transpose(1, 0) for img in imgs]
    
    # show_imgs(imgs_encoder)   
     
    state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
    print(experiment.experiments[0].trial_config["model_cfg"])
    model_hparams = experiment.experiments[0].trial_config["model_cfg"]["params"]
    model = experiment.experiments[0].trial_config["model_cfg"]["type"](**model_hparams)
    model.load_state_dict(state_dict)

    # best_model = from_checkpoint(
    #     config_path,
    #     ckpt_path
    # )
    # print(type(best_model))
    # sampled_img = best_model.sample(sample_shape=[1]).reshape(28, 28)
     
    # plt.imshow(sampled_img.cpu().detach().numpy())
    # plt.show()
    
    # best_model.log_prob(data_test[i:j][0].to("cpu")).sum()
    # test_loss = 0
    # for i in range(0, len(data_test), 100):
    #     j = min([len(data_test), i + 100])
    #     test_loss += float(
    #         -best_model.log_prob(data_test[i:j][0].to("cpu")).sum()
    #     )
    # test_loss /= len(data_test)
    # print(test_loss)
 
    
    
if __name__ == "__main__":
    dir = "/Users/giorgiapitteri/Downloads/model_mnist_full_laplace/"
    best_model_cfg = "config.pkl"
    best_model_ckpt = "model.pt"
    config = "experiments/mnist/mnist.yaml" #_train_feature_extractor.yaml"
    main(dir + best_model_cfg, dir + best_model_ckpt, config)
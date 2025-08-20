from matplotlib.patches import Circle
import torch
from matplotlib import pyplot as plt
from src.usflows.explib.config_parser import from_checkpoint
from src.usflows.explib.eval import RadialFlowEvaluator
import os
from src.usflows.distributions import Chi
from src.usflows.explib.datasets import DistributionDataset
from src.usflows.distributions import GMM
from torch.nn.functional import softplus

def evaluate_gmm_flow(base_dir):
    subfolders = sorted(os.listdir(base_dir))
    subfolders = [os.path.join(base_dir, d) for d in subfolders]
    subfolders = sorted([d for d in subfolders if os.path.isdir(d)])

    model_dirs = [
        os.path.join(base_dir, subfolder) for subfolder in subfolders if os.path.isdir(os.path.join(base_dir, subfolder))
    ]

    fig, axes = plt.subplots(int(len(model_dirs)/2), 6, figsize=(30, 2.5*len(model_dirs)))

    print(model_dirs)
    for i, model_dir in enumerate(model_dirs):
        print(model_dir)
        # Locate model files
        pkl_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pkl")])
        pt_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pt")])

        if not pkl_files or not pt_files:
            print(f"Skipping {model_dir} (missing files)")
            continue

        pkl_path = os.path.join(model_dir, pkl_files[-1])
        pt_path = os.path.join(model_dir, pt_files[-1])
        try:
            model = from_checkpoint(pkl_path, pt_path)
        except:
            continue
            
        dim = int(model_dir.split("_")[-1][:-1])
        print(f"{dim}D GMM")
        distribution=GMM(
            loc=torch.stack([-torch.ones(dim), torch.ones(dim)]), 
            covariance_matrix=torch.stack([torch.eye(dim)]*2),
            mixture_weights=torch.ones(2)/2
        )
        ds = DistributionDataset(
            distribution=distribution,
            num_samples=10000
        )[:][0]
        
        evaluator = RadialFlowEvaluator(
            model,
            ds,
            p=2.0,
            norm_distribution=Chi(
                df=dim,
                scale=softplus(model.base_distribution.scale_unconstrained),
                validate_args=False
            )
        )

        row = int(i/2)
        col = 3*(i%2)
        
        evaluator.kde_plot_norms(ax=axes[row, col + 1])
        axes[row, col + 1].set_title(f"KDE of Norm Distributions ({dim}D)")
        axes[row, col + 1].legend(loc = 'upper right')
        evaluator.pp_plot_norms(ax=axes[row, col + 2])
        axes[row, col + 2].set_title(f"PP-plot of Norm Distributions ({dim}D)")
        evaluator.logprob_reference_scatter_plot(ax=axes[row, col + 0], ref_distribution=distribution)
        axes[row, col + 0].set_title(f"Log-Probability Comparison ({dim}D)")
        
        scatter_fig, ax = plt.subplots()
        evaluator.nll_norm_scatter_plot(ax=ax, ref_distribution=distribution)
        ax.set_title(f"NLL vs Latent Norm ({dim}D)")
        scatter_fig.savefig(f"gmm_nll_vd_latent_norms_{dim}D.png")

    fig.tight_layout()
    fig.savefig(f"gmm_eval_all.png")
    plt.show()

def gmm_contourplot_2D(model_dir):

    pkl_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pkl")])
    pt_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pt")])

    pkl_path = os.path.join(model_dir, pkl_files[-1])
    pt_path = os.path.join(model_dir, pt_files[-1])

    dim = 2
    distribution=GMM(
        loc=torch.stack([-torch.ones(dim), torch.ones(dim)]), 
        covariance_matrix=torch.stack([torch.eye(dim)]*2),
        mixture_weights=torch.ones(2)/2
    )

    model = from_checkpoint(pkl_path, pt_path)

    with torch.no_grad():
        ds = distribution.sample([1000])
        latents = model.backward(ds) - model.base_distribution.loc


    _, axes = plt.subplots(ncols=2, figsize=(10, 5))
    
    ax = axes[0]
    ax.scatter(ds[:, 0], ds[:, 1])
    ax.set_aspect("equal")
    ax.set_title("Data Distribution")

    ax = axes[1]
    ax.scatter(latents[:, 0], latents[:, 1])
    scale = softplus(model.base_distribution.scale_unconstrained)
    ax.add_patch(Circle((0., 0.), radius=scale, fill=False, edgecolor='blue', linewidth=2))
    ax.add_patch(Circle((0., 0.), radius=2*scale, fill=False, edgecolor='blue', linewidth=2))
    ax.add_patch(Circle((0., 0.), radius=3*scale, fill=False, edgecolor='blue', linewidth=2))
    ax.set_aspect('equal')
    ax.set_title("Distribution of Centered Data Latents")

    plt.show()
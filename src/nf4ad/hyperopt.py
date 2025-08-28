from glob import glob
import io
import json
import logging
import math
import os
import shutil
import typing as T
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from pickle import dump
from PIL import Image
import torch
from torch.utils.tensorboard import SummaryWriter
import plotly.express as px
import ray
from ray import tune
from ray.air import RunConfig, session
from sklearn.metrics import roc_auc_score, average_precision_score


from src.usflows.explib.config_parser import from_checkpoint, create_objects_from_classes
from src.usflows.explib.hyperopt import HyperoptExperiment
from nf4ad.vaeflow import VAEFlow

class VAEFlowHyperoptExperiment(HyperoptExperiment):
    """Hyperopt experiment specialized for VAEFlow models.

    Overrides _trial to compute per-epoch validation metrics:
      - reconstruction NLL
      - latent NLL (from flow prior)
      - anomaly score = recon NLL + latent NLL
      - reconstruction MSE
    and reports them via Ray after each epoch; saves best checkpoint.
    """

    @classmethod
    def _trial(cls, config: T.Dict[str, T.Any], device: torch.device = None) -> Dict[str, float]:
        config = create_objects_from_classes(config)
        writer = SummaryWriter()
        torch.autograd.set_detect_anomaly(True)
        if device is None:
            if config.get("device") is not None:
                device = config["device"]
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        dataset = config["dataset"]
        data_train = dataset.get_train()
        data_val = dataset.get_val()

        # Instantiate VAEFlow (expect model_cfg to point to VAEFlow or compatible constructor)
        model_hparams = config["model_cfg"].get("params", {})
        model_cls = config["model_cfg"]["type"]
        # model_cls may already be the class/object after create_objects_from_classes
        model = model_cls(**model_hparams)
        model.to(device)

        best_score = float("inf")
        strikes = 0

        for epoch in range(config["epochs"]):
            # Train for one epoch (fit defaults to 1 epoch in our use)
            train_loss = model.fit(
                data_train,
                config["optim_cfg"]["optimizer"],
                config["optim_cfg"]["params"],
                batch_size=config["batch_size"],
                device=device,
            )[-1]

            # Validation: compute per-sample recon NLL, latent NLL and anomaly scores
            total_recon_nll = 0.0
            total_latent_nll = 0.0
            total_anom = 0.0
            total_mse = 0.0
            # accumulate model loss (KL or loss returned by model.loss_function)
            total_model_loss = 0.0
            n_samples = 0

            all_scores = []
            all_labels = []

            for i in range(0, len(data_val), config["batch_size"]):
                j = min([len(data_val), i + config["batch_size"]])
                # fetch items robustly (supports dataset[i:j] returning lists or indexing)
                items = [data_val[k] for k in range(i, j)]
                # items may be (x, y) or x
                if isinstance(items[0], (list, tuple)):
                    xs = torch.stack([it[0] for it in items]).to(device)
                    ys = [it[1] for it in items]
                else:
                    xs = torch.stack(items).to(device)
                    ys = None

                with torch.no_grad():
                    x_recon, mu, logvar = model.forward(xs)
                    z = model.reparameterize(mu, logvar)

                    # reconstruction NLL (per-sample)
                    batch_b = xs.shape[0]
                    D = xs[0].numel()
                    sigma2 = (model.sigma_min ** 2)
                    sq_err_per_sample = ((xs - x_recon) ** 2).view(batch_b, -1).sum(dim=1)
                    recon_nll_per_sample = 0.5 * sq_err_per_sample / sigma2 + 0.5 * D * math.log(2.0 * math.pi * sigma2)

                    # latent NLL via flow prior (use VAEFlow helper to reshape before passing to flow)
                    log_p_z = model.prior_log_prob(z)
                    if log_p_z.dim() == 0:
                        log_p_z = log_p_z.repeat(batch_b)
                    latent_nll_per_sample = -log_p_z

                    anom_per_sample = recon_nll_per_sample + latent_nll_per_sample
                    mse_per_sample = sq_err_per_sample / float(D)

                    # New: compute model's loss for this batch robustly
                    try:
                        model_loss = model.loss_function(xs, x_recon, mu, logvar, z=z)
                    except TypeError:
                        # fallback if model.loss_function expects different ordering
                        model_loss = model.loss_function(x_recon, xs, mu, logvar, z=z)

                    # Normalize model_loss to a batch sum (robust to scalar mean or per-sample tensor)
                    if isinstance(model_loss, torch.Tensor):
                        if model_loss.dim() == 0:
                            batch_loss_sum = float(model_loss.item()) * batch_b
                        elif model_loss.dim() == 1 and model_loss.shape[0] == batch_b:
                            batch_loss_sum = float(model_loss.sum().cpu())
                        else:
                            batch_loss_sum = float(model_loss.sum().cpu())
                    else:
                        # numeric scalar (assume mean)
                        batch_loss_sum = float(model_loss) * batch_b

                total_recon_nll += float(recon_nll_per_sample.sum().cpu())
                total_latent_nll += float(latent_nll_per_sample.sum().cpu())
                total_anom += float(anom_per_sample.sum().cpu())
                total_mse += float(mse_per_sample.sum().cpu())
                # accumulate model loss
                total_model_loss += batch_loss_sum
                n_samples += batch_b

                all_scores.extend(anom_per_sample.detach().cpu().numpy().tolist())
                if ys is not None:
                    # ensure labels are numeric 0/1
                    all_labels.extend([int(y) for y in ys])

            # averages
            val_recon_nll = total_recon_nll / n_samples
            val_latent_nll = total_latent_nll / n_samples
            val_anom = total_anom / n_samples
            val_mse = total_mse / n_samples
            # New: average validation loss (KL / model loss)
            val_loss = total_model_loss / n_samples

            # Compute ROC AUC and PR AUC if labels available and have both classes
            val_rocauc = float("nan")
            val_prauc = float("nan")
            if len(all_labels) > 0:
                labels_arr = np.array(all_labels)
                scores_arr = np.array(all_scores)
                # need at least one positive and one negative sample
                if labels_arr.min() != labels_arr.max():
                    try:
                        val_rocauc = roc_auc_score(labels_arr, scores_arr)
                    except Exception:
                        val_rocauc = float("nan")
                    try:
                        val_prauc = average_precision_score(labels_arr, scores_arr)
                    except Exception:
                        val_prauc = float("nan")

            # report to Ray (include val_loss)
            session.report(
                {
                    "train_loss": train_loss,
                    "val_recon_nll": val_recon_nll,
                    "val_latent_nll": val_latent_nll,
                    "val_anomaly": val_anom,
                    "val_mse": val_mse,
                    "val_rocauc": val_rocauc,
                    "val_prauc": val_prauc,
                    "val_loss": val_loss,
                },
                checkpoint=None,
            )

            # Save best model on anomaly score improvement (lower is better)
            if val_loss < best_score:
                strikes = 0
                best_score = val_loss

                # Save checkpoint similar to parent implementation
                try:
                    wdr = os.getcwd()
                    wdr_split = wdr.split("/")
                    expdir = [d for d in wdr_split if d.startswith("_trial_")][0]
                    trialdir = wdr_split[-1]
                    torch.save(model.state_dict(), f"{config['storage_path']}/{expdir}/{trialdir}/checkpoint.pt")
                except Exception:
                    # best-effort fallback: try to find checkpoint in temp
                    try:
                        chkpt_path = glob(f"{os.getcwd()}/session_latest/**/checkpoint.pt", recursive=True)[0]
                        shutil.copyfile(chkpt_path, f"{config['storage_path']}/checkpoint_tmp.pt")
                    except Exception:
                        pass
                
            else:
                strikes += 1
                if strikes > config.get("patience", 0):
                    break

        writer.close()
        return {
            "val_loss_best": best_score,
            "val_anomaly": val_anom,
            "val_recon_nll": val_recon_nll,
            "val_latent_nll": val_latent_nll,
            "val_mse": val_mse,
            "val_loss": val_loss,
        }
    
    def conduct(self, report_dir: os.PathLike, storage_path: os.PathLike = None):
        """Run hyperparameter optimization experiment.

        Args:
            report_dir (os.PathLike): report directory
            storage_path (os.PathLike, optional): Ray logging path. Defaults to None.
        """
        if self.skip:
            return
        
                
        #ray.init()
        
        if storage_path is None:
            storage_path = os.path.expanduser("~/ray_results")

        runcfg = RunConfig(storage_path=storage_path)
        runcfg.local_dir = f"{storage_path}/local/"
        tuner_config = {"run_config": runcfg}
        
        self.temp_dir = os.path.join(storage_path, "temp")
        ray.init(_temp_dir=f"{storage_path}/temp/")
        self.trial_config["storage_path"] = storage_path

        try:
            exptime = str(datetime.now())
            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(VAEFlowHyperoptExperiment._trial),
                    resources={"cpu": self.cpus_per_trial, "gpu": self.gpus_per_trial},
                ),
                tune_config=tune.TuneConfig(
                    scheduler=self.scheduler,
                    #search_alg=search_alg,
                    num_samples=self.num_hyperopt_samples,
                    **(self.tuner_params),
                ),
                param_space=self.trial_config,
                **(tuner_config),
            )
            results = tuner.fit()

            # TODO: hacky way to determine the last experiment
            exppath = (
                storage_path
                + [
                    "/" + f
                    for f in sorted(os.listdir(storage_path))
                    if f.startswith("_trial")
                ][-1]
            )
            report_file = os.path.join(
                report_dir, f"report_{self.name}_" + exptime + ".csv"
            )
            results = self._build_report(exppath, report_file=report_file, config_prefix="param_")
            best_result = results.iloc[results["val_loss_best"].argmin()].copy()

            self._test_best_model(best_result, exppath, report_dir, device=self.device, exp_id=exptime)
        finally:
            ray.shutdown()

    def _test_best_model(self, best_result: pd.Series, expdir: str, report_dir: str, device: torch.device = "cpu", exp_id: str = "foo") -> pd.Series:
        """Load best trial model and evaluate test metrics (recon NLL, latent NLL, anomaly, MSE, ROC AUC, PR AUC)."""
        trial_id = best_result.trial_id if hasattr(best_result, "trial_id") else best_result["trial_id"]
        id = f"exp_{exp_id}_{trial_id}"
        # Locate trial folder and copy checkpoint + params
        for d in os.listdir(expdir):
            if trial_id in d:
                try:
                    shutil.copyfile(
                        os.path.join(expdir, d, "checkpoint.pt"),
                        os.path.join(report_dir, f"{self.name}_{id}_best_model.pt"),
                    )
                except Exception:
                    # fallback to temp session checkpoint
                    chkpt_path = glob(f"{self.temp_dir}/session_latest/**/checkpoint.pt", recursive=True)[0]
                    shutil.copyfile(chkpt_path, os.path.join(report_dir, f"{self.name}_{id}_best_model.pt"))
                # copy params.pkl if present
                try:
                    shutil.copyfile(
                        os.path.join(expdir, d, "params.pkl"),
                        os.path.join(report_dir, f"{self.name}_{id}_best_config.pkl"),
                    )
                except Exception:
                    # ignore missing params.pkl; from_checkpoint likely needs it but we'll try best-effort
                    pass
                break
        """
        # Instantiate best model from saved config/checkpoint if possible
        best_model_path = os.path.join(report_dir, f"{self.name}_{id}_best_model.pt")
        best_cfg_path = os.path.join(report_dir, f"{self.name}_{id}_best_config.pkl")
        try:
            # from_checkpoint expects (config_path, model_path)
            best_model = from_checkpoint(best_cfg_path, best_model_path).to(device)
        except Exception:
            # Fallback: try loading state_dict into a fresh model from current trial_config
            cfg = create_objects_from_classes(self.trial_config)
            model_hparams = cfg["model_cfg"].get("params", {})
            model_cls = cfg["model_cfg"]["type"]
            best_model = model_cls(**model_hparams).to(device)
            try:
                state = torch.load(best_model_path, map_location=device)
                best_model.load_state_dict(state)
            except Exception:
                # if loading fails, raise
                raise RuntimeError("Failed to instantiate best model from checkpoint or trial config.")

        # Prepare test dataset (use experiment trial_config to construct dataset)
        cfg = create_objects_from_classes(self.trial_config)
        data_test = cfg["dataset"].get_test()

        # Evaluate metrics on test set
        best_model.eval()
        total_recon_nll = 0.0
        total_latent_nll = 0.0
        total_anom = 0.0
        total_mse = 0.0
        n_samples = 0

        all_scores = []
        all_labels = []

        bs = 100
        for i in range(0, len(data_test), bs):
            j = min([len(data_test), i + bs])
            items = [data_test[k] for k in range(i, j)]
            if isinstance(items[0], (list, tuple)):
                xs = torch.stack([it[0] for it in items]).to(device)
                ys = [it[1] for it in items]
            else:
                xs = torch.stack(items).to(device)
                ys = None

            with torch.no_grad():
                x_recon, mu, logvar = best_model.forward(xs)
                z = best_model.reparameterize(mu, logvar)

                batch_b = xs.shape[0]
                D = xs[0].numel()
                sigma2 = (best_model.sigma_min ** 2)
                sq_err_per_sample = ((xs - x_recon) ** 2).view(batch_b, -1).sum(dim=1)
                recon_nll_per_sample = 0.5 * sq_err_per_sample / sigma2 + 0.5 * D * math.log(2.0 * math.pi * sigma2)

                log_p_z = best_model.prior_log_prob(z)
                if log_p_z.dim() == 0:
                    log_p_z = log_p_z.repeat(batch_b)
                latent_nll_per_sample = -log_p_z

                anom_per_sample = recon_nll_per_sample + latent_nll_per_sample
                mse_per_sample = sq_err_per_sample / float(D)

            total_recon_nll += float(recon_nll_per_sample.sum().cpu())
            total_latent_nll += float(latent_nll_per_sample.sum().cpu())
            total_anom += float(anom_per_sample.sum().cpu())
            total_mse += float(mse_per_sample.sum().cpu())
            n_samples += batch_b

            all_scores.extend(anom_per_sample.detach().cpu().numpy().tolist())
            if ys is not None:
                all_labels.extend([int(y) for y in ys])

        # compute averages
        test_recon_nll = total_recon_nll / n_samples
        test_latent_nll = total_latent_nll / n_samples
        test_anom = total_anom / n_samples
        test_mse = total_mse / n_samples

        # compute ROC / PR AUC if labels available
        test_rocauc = float("nan")
        test_prauc = float("nan")
        if len(all_labels) > 0:
            labels_arr = np.array(all_labels)
            scores_arr = np.array(all_scores)
            if labels_arr.min() != labels_arr.max():
                try:
                    test_rocauc = roc_auc_score(labels_arr, scores_arr)
                except Exception:
                    test_rocauc = float("nan")
                try:
                    test_prauc = average_precision_score(labels_arr, scores_arr)
                except Exception:
                    test_prauc = float("nan")

        # attach metrics to best_result and save
        best_result["test_recon_nll"] = test_recon_nll
        best_result["test_latent_nll"] = test_latent_nll
        best_result["test_anomaly"] = test_anom
        best_result["test_mse"] = test_mse
        best_result["test_rocauc"] = test_rocauc
        best_result["test_prauc"] = test_prauc

        os.makedirs(report_dir, exist_ok=True)
        best_result.to_csv(os.path.join(report_dir, f"{self.name}_best_result.csv"))
        return best_result
        """
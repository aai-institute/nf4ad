from asyncio import sleep
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Iterable, Tuple, List
from .flows import Flow
import math
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)


class VectorEncoder(nn.Module):
    """MLP-based encoder for vector/tabular data.
    
    Designed for ADBench classical datasets with solid baseline architecture.
    Uses a progressive dimensionality reduction with batch normalization and dropout.
    
    Args:
        input_dim: Dimension of input vectors
        latent_dim: Dimension of latent space
        hidden_dims: List of hidden layer dimensions (default: [512, 256, 128])
        dropout: Dropout probability (default: 0.2)
        use_batchnorm: Whether to use batch normalization (default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Project to latent space (mu and logvar)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            logvar: Log variance of latent distribution (batch_size, latent_dim)
        """
        # Flatten if needed (handle both (B, D) and (B, 1, D) shapes)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


class VectorDecoder(nn.Module):
    """MLP-based decoder for vector/tabular data.
    
    Mirrors the encoder architecture with progressive dimensionality expansion.
    Suitable for ADBench classical datasets.
    
    Args:
        latent_dim: Dimension of latent space
        output_dim: Dimension of output vectors
        hidden_dims: List of hidden layer dimensions (default: [128, 256, 512])
        dropout: Dropout probability (default: 0.2)
        use_batchnorm: Whether to use batch normalization (default: True)
        output_activation: Optional output activation ('sigmoid', 'tanh', or None)
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        output_activation: Optional[str] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output layer
        self.fc_out = nn.Linear(prev_dim, output_dim)
        
        # Optional output activation
        self.output_activation = None
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        elif output_activation is not None:
            raise ValueError(f"Unknown output_activation: {output_activation}")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
        
        Returns:
            x_recon: Reconstructed tensor of shape (batch_size, output_dim)
        """
        x = self.decoder(z)
        x = self.fc_out(x)
        
        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x


class PreTrainedEncoder(nn.Module):
    """Wrapper for pre-trained encoders from torchvision"""
    
    def __init__(self, arch: str = 'resnet18', latent_dim: int = 64, 
                 pretrained: bool = True, trainable: bool = False):
        super().__init__()

        # Load pre-trained model and remove classifier head, infer feature dim
        arch_l = arch.lower()

        if not hasattr(models, arch):
            raise ValueError(f"Unknown model variant: {arch}")

        self.encoder = getattr(models, arch)(pretrained=pretrained)

        # Remove or replace the final classification layer based on model type
        if arch_l.startswith(('resnet', 'wide_resnet')):
            self.encoder.fc = nn.Identity()
        elif arch_l.startswith(('vgg', 'mobilenet', 'squeezenet')):
            if hasattr(self.encoder, 'classifier'):
                self.encoder.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported encoder architecture: {arch}. "
                             "Supported: resnet*, vgg*, mobilenet*, squeezenet*, wide_resnet*.")

        # Infer feature dimension using a dummy forward pass
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            out = self.encoder(dummy_input)
            feature_dim = out.view(out.shape[0], -1).shape[1]

        # Project to latent space
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
         
        # Freeze encoder if not trainable
        if not trainable:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar

class SimpleDecoder(nn.Module):
    """Flexible decoder mapping latent vectors to images of arbitrary size.

    The decoder computes a small starting spatial resolution and a number of
    ConvTranspose2d upsampling stages required to reach the target output size.
    A final conv + sigmoid and interpolation ensure the exact requested output_size.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_channels: int = 3,
        output_size: Tuple[int, int] = (224, 224),
        base_channels: int = 512,
        min_channels: int = 32,
        start_min: int = 4,
    ):
        super().__init__()
        self.output_size = output_size
        out_h, out_w = output_size

        # determine number of upsampling layers to reach target size from a small start
        max_target = max(out_h, out_w)
        # ensure at least one upsample stage
        n_ups = max(1, int(math.ceil(math.log2(max_target / float(start_min))))) if max_target > start_min else 1

        # compute starting spatial resolution
        start_h = int(math.ceil(out_h / (2 ** n_ups)))
        start_w = int(math.ceil(out_w / (2 ** n_ups)))
        start_h = max(1, start_h)
        start_w = max(1, start_w)

        # channels schedule: halve channels each upsample (but keep above min_channels)
        channels = []
        ch = base_channels
        for _ in range(n_ups):
            channels.append(ch)
            ch = max(min_channels, ch // 2)

        # fully connected projection to starting feature map
        self.fc = nn.Linear(latent_dim, channels[0] * start_h * start_w)

        # build upsampling stack
        ups = []
        in_ch = channels[0]
        for idx in range(n_ups):
            out_ch = channels[idx + 1] if (idx + 1) < len(channels) else max(min_channels, in_ch // 2)
            # ConvTranspose2d upsamples by factor 2 (kernel=4,stride=2,pad=1)
            ups.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            ups.append(nn.ReLU(inplace=True))
            in_ch = out_ch

        self.ups = nn.Sequential(*ups)

        # final conv to get desired output channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # store starting spatial dims for reshape in forward
        self._start_h = start_h
        self._start_w = start_w

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        batch = z.shape[0]
        x = x.view(batch, -1, self._start_h, self._start_w)
        x = self.ups(x)
        x = self.final_conv(x)
        # Ensure exact output size (in case of mismatches due to ceil)
        if (x.shape[-2], x.shape[-1]) != self.output_size:
            x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return x

class VAEFlow(nn.Module):
    """VAE with Flow-based prior"""
    
    def __init__(self, flow_prior: Flow, 
                 encoder: Optional[nn.Module] = None,
                 decoder: Optional[nn.Module] = None,
                 latent_dim: int = 64,
                 encoder_arch: str = 'resnet18',
                 encoder_trainable: bool = False,
                 sigma_min: float = 0.1,
                 prior_shape: Optional[Tuple[int, ...]] = None,  # NEW
                 ):
        super().__init__()
        
        self.flow_prior = flow_prior
        self.latent_dim = latent_dim
        self.sigma_min = float(sigma_min)
        self.prior_shape = prior_shape  # NEW
        
        # Initialize encoder and decoder
        if encoder is None:
            self.encoder = PreTrainedEncoder(
                arch=encoder_arch, 
                latent_dim=latent_dim,
                trainable=encoder_trainable
            )
        else:
            self.encoder = encoder
            
        if decoder is None:
            self.decoder = SimpleDecoder(latent_dim=latent_dim)
        else:
            self.decoder = decoder

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    # NEW helper: reshape (batch, latent_dim) -> (batch, *prior_shape) for the flow
    def _reshape_for_flow(self, z: torch.Tensor) -> torch.Tensor:
        if self.prior_shape is None:
            return z
        # verify total size matches
        expected = int(np.prod(self.prior_shape))
        if z.shape[1] != expected:
            raise ValueError(f"latent_dim ({z.shape[1]}) does not match product(prior_shape) ({expected})")
        return z.view(z.shape[0], *self.prior_shape)

    # NEW helper: call flow prior log_prob with correctly shaped tensor
    def prior_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) under the flow prior. Accepts z as (batch, latent_dim)."""
        z_for_flow = self._reshape_for_flow(z)
        log_p = self.flow_prior.log_prob(z_for_flow)
        return log_p

    def loss_function(self, x: torch.Tensor, x_recon: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor, 
                     z: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Negative log-likelihood under X|Z ~ N(decoder(z), sigma_min^2 I)
        plus KL between approximate posterior q(z|x)=N(mu,diag(exp(logvar))) and flow prior p(z).
        Returns scalar loss (sum over batch).
        """
        # Reconstruction negative log-likelihood per-sample
        batch_size = x.shape[0]
        D = x[0].numel()  # dimensionality per sample
        sigma2 = (self.sigma_min ** 2)

        # per-sample squared error sums
        sq_err_per_sample = ((x - x_recon) ** 2).view(batch_size, -1).sum(dim=1)
        recon_nll_per_sample = 0.5 * sq_err_per_sample / sigma2 + 0.5 * D * math.log(2.0 * math.pi * sigma2)
        recon_loss = recon_nll_per_sample.sum()

        # KL divergence between q(z|x) and flow prior p(z)
        std = torch.exp(0.5 * logvar)
        q_dist = torch.distributions.Normal(mu, std)
        # per-sample log q(z)
        log_q_z_per_sample = q_dist.log_prob(z).view(batch_size, -1).sum(dim=1)
        # per-sample log p(z) from flow prior (use helper to reshape)
        log_p_z_per_sample = self.prior_log_prob(z)
        if log_p_z_per_sample.dim() == 0:
            # if flow returns scalar sum for whole batch, convert to per-sample (unlikely)
            log_p_z_per_sample = log_p_z_per_sample.repeat(batch_size)
        kl_per_sample = log_q_z_per_sample - log_p_z_per_sample
        kl_loss = kl_per_sample.sum()

        return recon_loss + beta * kl_loss

    def sample(self, n_samples: int) -> torch.Tensor:
        with torch.no_grad():
            # flow returns samples in its native shape; flatten if needed for decoder
            z_flow = self.flow_prior.sample([n_samples])
            if self.prior_shape is not None:
                z = z_flow.view(n_samples, -1)
            else:
                z = z_flow
            samples = self.decoder(z)
        return samples

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return mu

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly score as per-sample negative log-likelihood: recon NLL + (- log p(z))"""
        x_recon, mu, logvar = self.forward(x)
        z = self.reparameterize(mu, logvar)

        batch_size = x.shape[0]
        D = x[0].numel()
        sigma2 = (self.sigma_min ** 2)

        # reconstruction negative log-likelihood per-sample
        sq_err_per_sample = ((x - x_recon) ** 2).view(batch_size, -1).sum(dim=1)
        recon_nll_per_sample = 0.5 * sq_err_per_sample / sigma2 + 0.5 * D * math.log(2.0 * math.pi * sigma2)

        # latent negative log-likelihood per-sample (using flow prior via helper)
        log_p_z_per_sample = self.prior_log_prob(z)
        latent_nll_per_sample = -log_p_z_per_sample

        return recon_nll_per_sample + latent_nll_per_sample
    
    def fit(
        self,
        data_train,
        optim_cls=torch.optim.Adam,
        optim_params=None,
        batch_size=32,
        shuffle=True,
        gradient_clip=None,
        device=None,
        epochs=1,
        beta=1.0,
    ):
        """
        Fit the VAEFlow model.

        Args:
            data_train: training data (torch.utils.data.Dataset or Tensor).
            optim_cls: optimizer class.
            optim_params: optimizer parameter dictionary.
            batch_size: number of samples per optimization step.
            shuffle: shuffle data each epoch.
            gradient_clip: max norm for gradient clipping.
            device: torch.device.
            epochs: number of epochs.
            beta: KL weight.

        Returns:
            List of mean epoch losses.
        """

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

        self.to(device)
        self.train()

        if optim_params is None:
            optim_params = {"lr": 1e-3}
        optimizer = optim_cls(self.parameters(), **optim_params)

        if isinstance(data_train, torch.utils.data.Dataset):
            loader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle)
        else:
            # Assume Tensor or numpy array
            data = torch.as_tensor(data_train)
            loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        epoch_losses = []
        for _ in range(epochs):
            losses = []
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                optimizer.zero_grad()
                x_recon, mu, logvar = self.forward(batch)
                z = self.reparameterize(mu, logvar)
                loss = self.loss_function(batch, x_recon, mu, logvar, z, beta=beta)
                loss.backward()
                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            epoch_losses.append(np.mean(losses))
        return epoch_losses

    def elbo(
        self,
        x: torch.Tensor,
        nsamples: int = 1,
        beta: float = 1.0,
        reduction: str = "mean",
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Estimate the (beta-)ELBO for inputs x via Monte-Carlo with nsamples draws from q(z|x).

        Args:
            x: Input tensor (batch, ...).
            nsamples: Number of Monte-Carlo samples to estimate the expectation.
            beta: Weight for the KL term (beta-VAE style). ELBO = E_q[log p(x|z)] - beta * E_q[log q(z|x) - log p(z)].
            reduction: "mean", "sum" or "none" (per-sample).
            device: Optional device to perform computations on.

        Returns:
            Tensor: reduced ELBO (scalar if reduction="mean" or "sum"), or per-sample ELBO (if reduction="none").
        """
        if device is None:
            device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device("cpu")

        x = x.to(device)
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x)
            batch = x.shape[0]
            std = torch.exp(0.5 * logvar)

            # accumulate ELBO estimates for each MC sample
            elbo_samples = []
            for _ in range(max(1, int(nsamples))):
                eps = torch.randn_like(std, device=std.device)
                z = mu + eps * std

                # decode
                x_recon = self.decoder(z)

                # reconstruction negative log-likelihood per-sample
                D = x[0].numel()
                sigma2 = (self.sigma_min ** 2)
                sq_err_per_sample = ((x - x_recon) ** 2).view(batch, -1).sum(dim=1)
                recon_nll_per_sample = 0.5 * sq_err_per_sample / sigma2 + 0.5 * D * math.log(2.0 * math.pi * sigma2)
                recon_logp_per_sample = -recon_nll_per_sample  # log p(x|z)

                # q(z|x) log prob per-sample
                std_z = std
                q_dist = torch.distributions.Normal(mu, std_z)
                log_q_z_per_sample = q_dist.log_prob(z).view(batch, -1).sum(dim=1)

                # p(z) log prob via flow prior helper (robust to scalar/per-sample)
                log_p_z = self.prior_log_prob(z)
                if isinstance(log_p_z, torch.Tensor):
                    if log_p_z.dim() == 0:
                        log_p_z = log_p_z.repeat(batch)
                else:
                    log_p_z = torch.tensor(float(log_p_z), device=device).repeat(batch)

                # KL per-sample = log_q - log_p
                kl_per_sample = log_q_z_per_sample - log_p_z

                # ELBO per-sample (with beta on KL)
                elbo_per_sample = recon_logp_per_sample - beta * kl_per_sample
                elbo_samples.append(elbo_per_sample)

            # stack and average over MC samples
            elbo_stack = torch.stack(elbo_samples, dim=0)  # (nsamples, batch)
            elbo_est = elbo_stack.mean(dim=0)  # (batch,)

            if reduction == "mean":
                return elbo_est.mean().detach()
            elif reduction == "sum":
                return elbo_est.sum().detach()
            elif reduction == "none":
                return elbo_est.detach()
            else:
                raise ValueError(f"Unknown reduction '{reduction}', expected 'mean', 'sum' or 'none'.")

class VAEFlowEvaluator:
    """
    Evaluator for VAEFlow models.

    Args:
        model (VAEFlow): trained model
        dataset: dataset providing __len__ and __getitem__; items may be x or (x,y)
        device: torch device (defaults to model device)
        batch_size: evaluation batch size
        score_type: one of {"anomaly","recon","latent","neg_elbo"}
        nsamples: number of MC samples for ELBO estimation (used when score_type == "neg_elbo")
        beta: beta weight for elbo/kl (passed to model.elbo when used)
    """

    def __init__(
        self,
        model: VAEFlow,
        dataset,
        device: Optional[torch.device] = None,
        batch_size: int = 64,
        score_type: str = "anomaly",
        nsamples: int = 1,
        beta: float = 1.0,
    ):
        self.model = model
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.score_type = score_type
        self.nsamples = int(max(1, nsamples))
        self.beta = float(beta)
        self.model.eval()

    def _iter_batches(self):
        n = len(self.dataset)
        for i in range(0, n, self.batch_size):
            j = min(n, i + self.batch_size)
            items = [self.dataset[k] for k in range(i, j)]
            if isinstance(items[0], (list, tuple)):
                xs = torch.stack([it[0] for it in items]).to(self.device)
                ys = [it[1] for it in items]
            else:
                xs = torch.stack(items).to(self.device)
                ys = None
            yield xs, ys

    def scores_and_labels(self):
        """
        Compute per-sample anomaly scores and labels.

        Returns:
            scores: numpy array (n_samples,)
            labels: numpy array (n_samples,) or None if dataset has no labels
        """
        scores = []
        labels = [] if any(isinstance(self.dataset[k], (list, tuple)) for k in range(min(10, len(self.dataset)))) else None

        with torch.no_grad():
            for xs, ys in self._iter_batches():
                # forward
                x_recon, mu, logvar = self.model.forward(xs)
                z = self.model.reparameterize(mu, logvar)

                batch_b = xs.shape[0]
                D = xs[0].numel()
                sigma2 = (self.model.sigma_min ** 2)

                # reconstruction NLL per sample
                sq_err_per_sample = ((xs - x_recon) ** 2).view(batch_b, -1).sum(dim=1)
                recon_nll_per_sample = 0.5 * sq_err_per_sample / sigma2 + 0.5 * D * math.log(2.0 * math.pi * sigma2)

                # latent NLL per sample (negative log p(z))
                log_p_z = self.model.prior_log_prob(z)
                if isinstance(log_p_z, torch.Tensor):
                    if log_p_z.dim() == 0:
                        log_p_z = log_p_z.repeat(batch_b)
                else:
                    log_p_z = torch.tensor(float(log_p_z), device=self.device).repeat(batch_b)
                latent_nll_per_sample = -log_p_z

                if self.score_type == "anomaly":
                    s_batch = (recon_nll_per_sample + latent_nll_per_sample).detach().cpu().numpy()
                elif self.score_type == "recon":
                    s_batch = recon_nll_per_sample.detach().cpu().numpy()
                elif self.score_type == "latent":
                    s_batch = latent_nll_per_sample.detach().cpu().numpy()
                elif self.score_type == "neg_elbo":
                    # model.elbo returns ELBO; we want negative ELBO as anomaly score
                    elbo_vals = self.model.elbo(xs, nsamples=self.nsamples, beta=self.beta, reduction="none", device=self.device)
                    # ensure tensor on cpu
                    s_batch = (-elbo_vals).detach().cpu().numpy()
                else:
                    raise ValueError(f"Unknown score_type '{self.score_type}'")

                scores.extend(s_batch.tolist())

                if ys is not None:
                    labels.extend([int(y) for y in ys])

        scores = np.array(scores)
        labels = np.array(labels) if labels is not None and len(labels) > 0 else None
        return scores, labels

    def compute_roc_pr(self, return_curve: bool = False):
        """
        Compute ROC AUC and PR AUC.

        Returns:
            roc_auc, pr_auc, (optional) dict with curve arrays {fpr,tpr,thresholds,precision,recall,pr_thresholds}
        """
        scores, labels = self.scores_and_labels()
        if labels is None:
            return float("nan"), float("nan"), None if return_curve else (float("nan"), float("nan"))

        # need at least one positive and one negative
        if labels.min() == labels.max():
            return float("nan"), float("nan"), None if return_curve else (float("nan"), float("nan"))

        try:
            roc_auc = float(roc_auc_score(labels, scores))
        except Exception:
            roc_auc = float("nan")
        try:
            pr_auc = float(average_precision_score(labels, scores))
        except Exception:
            pr_auc = float("nan")

        if not return_curve:
            return roc_auc, pr_auc

        # compute precision-recall curve arrays
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        # for ROC curve we can compute fpr/tpr thresholds via sklearn if needed (omitted here to stay concise)
        return roc_auc, pr_auc, {
            "precision": precision,
            "recall": recall,
            "pr_thresholds": pr_thresholds,
        }

    def precision_recall_at_threshold(self, threshold: float):
        """
        Compute precision and recall at a given score threshold.

        Args:
            threshold: decision threshold on scores; samples with score >= threshold are predicted positive.

        Returns:
            precision, recall, f1
        """
        scores, labels = self.scores_and_labels()
        if labels is None:
            return float("nan"), float("nan"), float("nan")
        preds = (scores >= float(threshold)).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f = f1_score(labels, preds, zero_division=0)
        return float(p), float(r), float(f)

    def best_f1_threshold(self):
        """
        Find threshold that maximizes F1 on the dataset.

        Returns:
            best_threshold, best_f1, precision_array, recall_array, thresholds_array
        """
        scores, labels = self.scores_and_labels()
        if labels is None:
            return None, float("nan"), None, None, None

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        # precision_recall_curve returns thresholds of length n-1; compute F1 for each threshold
        # align thresholds with precision/recall: precision[i], recall[i] correspond to threshold thresholds[i-1]
        # we'll compute F1 across precision/recall (skip last point where threshold is undefined)
        f1s = (2 * precision * recall) / (precision + recall + 1e-12)
        # find best index
        best_idx = int(np.nanargmax(f1s))
        # derive threshold: if best_idx == len(thresholds) -> threshold slightly below min, handle gracefully
        if best_idx == 0:
            best_threshold = thresholds[0] if len(thresholds) > 0 else None
        else:
            thr_idx = best_idx - 1
            best_threshold = thresholds[thr_idx] if thr_idx < len(thresholds) else thresholds[-1] if len(thresholds) > 0 else None

        return best_threshold, float(np.nanmax(f1s)), precision, recall, thresholds

    def summary(self):
        """
        Convenience: compute and return main metrics dictionary.
        """
        roc_auc, pr_auc, _ = self.compute_roc_pr(return_curve=True)
        best_thr, best_f1, _, _, _ = self.best_f1_threshold()
        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "best_f1": best_f1,
            "best_threshold": best_thr,
            "score_type": self.score_type,
        }

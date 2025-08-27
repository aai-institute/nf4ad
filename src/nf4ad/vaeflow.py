from asyncio import sleep
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Iterable, Tuple
from .flows import Flow
import math
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F


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

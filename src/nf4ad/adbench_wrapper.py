"""
Wrapper classes to integrate VAEFlow with ADBench framework.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .vaeflow import VAEFlow, SimpleDecoder
from .flows import Flow


class ADBenchVAEFlow:
    """
    Wrapper for VAEFlow to work with ADBench's interface.
    
    ADBench expects:
    - fit(X_train, y_train=None)
    - predict_score(X_test) -> anomaly scores
    """
    
    def __init__(
        self,
        flow_prior: Flow,
        latent_dim: int = 64,
        encoder_arch: str = 'resnet18',
        encoder_trainable: bool = False,
        input_channels: int = 1,
        input_size: tuple = (28, 28),
        sigma_min: float = 0.1,
        prior_shape: Optional[tuple] = None,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: Optional[str] = None,
        gradient_clip: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        Args:
            flow_prior: Flow model for the latent prior
            latent_dim: Dimensionality of latent space
            encoder_arch: Architecture for encoder (if using pretrained)
            encoder_trainable: Whether to train encoder weights
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            input_size: Spatial size of input images (H, W)
            sigma_min: Minimum sigma for reconstruction likelihood
            prior_shape: Shape for flow prior (if needed)
            batch_size: Training batch size
            epochs: Number of training epochs
            lr: Learning rate
            beta: KL weight
            device: Device to use ('cuda', 'cpu', or 'mps')
            gradient_clip: Gradient clipping norm
            verbose: Print training progress
        """
        self.latent_dim = latent_dim
        self.encoder_arch = encoder_arch
        self.encoder_trainable = encoder_trainable
        self.input_channels = input_channels
        self.input_size = input_size
        self.sigma_min = sigma_min
        self.prior_shape = prior_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.gradient_clip = gradient_clip
        self.verbose = verbose
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Create custom encoder/decoder for tabular/small image data
        if input_channels == 1 or input_size[0] < 224:
            encoder = self._create_simple_encoder()
            decoder = SimpleDecoder(
                latent_dim=latent_dim,
                output_channels=input_channels,
                output_size=input_size,
            )
        else:
            encoder = None  # Use default pretrained
            decoder = SimpleDecoder(
                latent_dim=latent_dim,
                output_channels=input_channels,
                output_size=input_size,
            )
        
        # Initialize VAEFlow model
        self.model = VAEFlow(
            flow_prior=flow_prior,
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            encoder_arch=encoder_arch,
            encoder_trainable=encoder_trainable,
            sigma_min=sigma_min,
            prior_shape=prior_shape,
        )
        self.model.to(self.device)
        
    def _create_simple_encoder(self) -> nn.Module:
        """Create a simple CNN encoder for small images or tabular data."""
        class SimpleEncoder(nn.Module):
            def __init__(self, input_channels, input_size, latent_dim):
                super().__init__()
                h, w = input_size
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                )
                # Calculate flattened size after convolutions.
                # We use a dummy forward pass to determine the output shape of the conv layers,
                # since the output size depends on the input size and convolution parameters.
                with torch.no_grad():
                    dummy = torch.zeros(1, input_channels, h, w)
                    out = self.conv_layers(dummy)
                    self.conv_output_shape = out.shape  # Store intermediate output shape for clarity
                    self.flat_size = out.view(1, -1).shape[1]  # Flatten to get feature size for FC layers
                self.fc_mu = nn.Linear(self.flat_size, latent_dim)
                self.fc_logvar = nn.Linear(self.flat_size, latent_dim)
                
            def forward(self, x):
                features = self.conv_layers(x)
                features = features.view(features.size(0), -1)
                mu = self.fc_mu(features)
                logvar = self.fc_logvar(features)
                return mu, logvar
        
        return SimpleEncoder(self.input_channels, self.input_size, self.latent_dim)
    
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """
        Fit the model on training data.
        
        Args:
            X_train: Training data, shape (n_samples, n_features) or (n_samples, C, H, W)
            y_train: Labels (ignored, only normal data expected)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_train)
        
        # Reshape if needed (flatten -> image)
        if X_tensor.dim() == 2:
            n_samples = X_tensor.shape[0]
            X_tensor = X_tensor.view(n_samples, self.input_channels, *self.input_size)
        
        if self.verbose:
            print(f"Training VAEFlow on {len(X_tensor)} samples...")
            print(f"Input shape: {X_tensor.shape}")
            print(f"Device: {self.device}")
        
        # Train model
        losses = self.model.fit(
            data_train=X_tensor,
            optim_cls=torch.optim.Adam,
            optim_params={"lr": self.lr},
            batch_size=self.batch_size,
            shuffle=True,
            gradient_clip=self.gradient_clip,
            device=self.device,
            epochs=self.epochs,
            beta=self.beta,
        )
        
        if self.verbose:
            print(f"Training completed. Final loss: {losses[-1]:.4f}")
        
        return self
    
    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for test data.
        
        Args:
            X_test: Test data, shape (n_samples, n_features) or (n_samples, C, H, W)
            
        Returns:
            Anomaly scores, shape (n_samples,)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test)
        
        # Reshape if needed
        if X_tensor.dim() == 2:
            n_samples = X_tensor.shape[0]
            X_tensor = X_tensor.view(n_samples, self.input_channels, *self.input_size)
        
        X_tensor = X_tensor.to(self.device)
        
        # Compute anomaly scores
        self.model.eval()
        with torch.no_grad():
            scores = self.model.anomaly_score(X_tensor)
        
        return scores.cpu().numpy()
    
    def predict(self, X_test: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X_test: Test data
            threshold: Decision threshold (if None, use median of scores)
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        scores = self.predict_score(X_test)
        if threshold is None:
            threshold = np.median(scores)
        return (scores > threshold).astype(int)


class ADBenchVAEFlowTabular(ADBenchVAEFlow):
    """
    Specialized wrapper for tabular data from ADBench.
    
    Automatically handles reshaping tabular data to/from image format.
    """
    
    def __init__(
        self,
        flow_prior: Flow,
        n_features: int,
        latent_dim: int = 64,
        sigma_min: float = 0.1,
        prior_shape: Optional[tuple] = None,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: Optional[str] = None,
        gradient_clip: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        Args:
            n_features: Number of input features (will be reshaped to square image)
        """
        # Compute square image size
        img_size = int(np.ceil(np.sqrt(n_features)))
        self.n_features = n_features
        self.img_size = img_size
        
        super().__init__(
            flow_prior=flow_prior,
            latent_dim=latent_dim,
            encoder_trainable=True,
            input_channels=1,
            input_size=(img_size, img_size),
            sigma_min=sigma_min,
            prior_shape=prior_shape,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            beta=beta,
            device=device,
            gradient_clip=gradient_clip,
            verbose=verbose,
        )
    
    def _tabular_to_image(self, X: np.ndarray) -> np.ndarray:
        """Convert tabular data to image format with padding."""
        n_samples = X.shape[0]
        img_arr = np.zeros((n_samples, 1, self.img_size, self.img_size), dtype=np.float32)
        for i in range(n_samples):
            flat = X[i, :self.n_features]
            img_arr[i, 0].flat[:self.n_features] = flat
        return img_arr
    
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None):
        """Fit on tabular data."""
        X_img = self._tabular_to_image(X_train)
        return super().fit(X_img, y_train)
    
    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for tabular data."""
        X_img = self._tabular_to_image(X_test)
        return super().predict_score(X_img)

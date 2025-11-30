"""
Wrapper classes to integrate VAEFlow with ADBench framework.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
from .vaeflow import VAEFlow, VectorEncoder, VectorDecoder
from .flows import Flow


class ADBenchVAEFlow:
    """
    Wrapper for VAEFlow to work with ADBench's interface for tabular/vector data.
    
    This class is designed to work exclusively with vector (tabular) data from
    ADBench classical datasets. It uses MLP-based encoder and decoder architectures
    suitable for anomaly detection benchmarks.
    
    ADBench expects:
    - fit(X_train, y_train=None)
    - predict_score(X_test) -> anomaly scores
    """
    
    def __init__(
        self,
        flow_prior: Flow,
        input_dim: int,
        latent_dim: int = 64,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        output_activation: Optional[str] = None,
        sigma_min: float = 0.1,
        prior_shape: Optional[tuple] = None,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: Optional[str] = None,
        gradient_clip: Optional[float] = None,
        verbose: bool = True,
        patience: Optional[int] = None,
        min_delta: float = 1e-4,
    ):
        """
        Args:
            flow_prior: Flow model for the latent prior
            input_dim: Dimensionality of input vectors
            latent_dim: Dimensionality of latent space
            encoder_hidden_dims: Hidden layer dimensions for encoder (default: [512, 256, 128])
            decoder_hidden_dims: Hidden layer dimensions for decoder (default: [128, 256, 512])
            dropout: Dropout probability for encoder/decoder
            use_batchnorm: Whether to use batch normalization
            output_activation: Output activation ('sigmoid', 'tanh', or None)
            sigma_min: Minimum sigma for reconstruction likelihood
            prior_shape: Shape for flow prior (if needed)
            batch_size: Training batch size
            epochs: Number of training epochs
            lr: Learning rate
            beta: KL weight
            device: Device to use ('cuda', 'cpu', or 'mps')
            gradient_clip: Gradient clipping norm
            verbose: Print training progress
            patience: Number of epochs with no improvement for early stopping (None to disable)
            min_delta: Minimum change in loss to qualify as an improvement
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sigma_min = sigma_min
        self.prior_shape = prior_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.beta = beta
        self.gradient_clip = gradient_clip
        self.verbose = verbose
        self.patience = patience
        self.min_delta = min_delta
        
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
        
        # Create encoder and decoder for vector data
        encoder = VectorEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        
        decoder = VectorDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            output_activation=output_activation,
        )
        
        # Initialize VAEFlow model
        self.model = VAEFlow(
            flow_prior=flow_prior,
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            sigma_min=sigma_min,
            prior_shape=prior_shape,
        )
        self.model.to(self.device)
        
        # Store training history
        self.training_losses_: Optional[List[float]] = None
    
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'ADBenchVAEFlow':
        """
        Fit the model on training data with early stopping.
        
        Args:
            X_train: Training data, shape (n_samples, n_features)
            y_train: Labels (ignored, only normal data expected)
            
        Returns:
            self with training_losses_ attribute set
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_train)
        
        # Ensure 2D shape (batch_size, features)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        elif X_tensor.dim() > 2:
            X_tensor = X_tensor.view(X_tensor.size(0), -1)
        
        if self.verbose:
            print(f"Training VAEFlow on {len(X_tensor)} samples...")
            print(f"Input shape: {X_tensor.shape}")
            print(f"Device: {self.device}")
        
        # Create data loader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.epochs):
            losses = []
            for (batch,) in loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                x_recon, mu, logvar = self.model(batch)
                z = self.model.reparameterize(mu, logvar)
                
                # Compute loss
                loss = self.model.loss_function(batch, x_recon, mu, logvar, z, beta=self.beta)
                
                loss.backward()
                
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            
            epoch_loss = np.mean(losses)
            epoch_losses.append(epoch_loss)
            
            # Early stopping check
            if self.patience is not None:
                if epoch_loss < best_loss - self.min_delta:
                    best_loss = epoch_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}/{self.epochs}")
                    break
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")
        
        # Restore best model if early stopping was triggered
        if best_state is not None and self.patience is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            if self.verbose:
                print(f"Restored best model with loss: {best_loss:.4f}")
        
        if self.verbose:
            print(f"Training completed. Final loss: {epoch_losses[-1]:.4f}")
        
        self.training_losses_ = epoch_losses
        return self
    
    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for test data.
        
        Args:
            X_test: Test data, shape (n_samples, n_features)
            
        Returns:
            Anomaly scores, shape (n_samples,)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test)
        
        # Ensure 2D shape
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        elif X_tensor.dim() > 2:
            X_tensor = X_tensor.view(X_tensor.size(0), -1)
        
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


class ADBenchFlow:
    """
    Wrapper for Flow models (USFlow/NonUSFlow) to work with ADBench's interface.
    
    Uses the flow model's negative log-likelihood as anomaly score.
    
    ADBench expects:
    - fit(X_train, y_train=None)
    - predict_score(X_test) -> anomaly scores
    """
    
    def __init__(
        self,
        flow_model: Flow,
        batch_size: int = 32,
        epochs: int = 50,
        lr: float = 1e-3,
        device: Optional[str] = None,
        gradient_clip: Optional[float] = None,
        verbose: bool = True,
        patience: Optional[int] = None,
        min_delta: float = 1e-4,
    ):
        """
        Args:
            flow_model: Flow model (USFlow or NonUSFlow)
            batch_size: Training batch size
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to use ('cuda', 'cpu', or 'mps')
            gradient_clip: Gradient clipping norm
            verbose: Print training progress
            patience: Number of epochs with no improvement for early stopping (None to disable)
            min_delta: Minimum change in loss to qualify as an improvement
        """
        self.flow_model = flow_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.gradient_clip = gradient_clip
        self.verbose = verbose
        self.patience = patience
        self.min_delta = min_delta
        
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
        
        self.flow_model.to(self.device)
        
        # Store training history
        self.training_losses_: Optional[List[float]] = None
    
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'ADBenchFlow':
        """
        Fit the flow model on training data with early stopping.
        
        Args:
            X_train: Training data, shape (n_samples, n_features)
            y_train: Labels (ignored, only normal data expected)
            
        Returns:
            self with training_losses_ attribute set
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_train)
        
        if self.verbose:
            print(f"Training Flow model on {len(X_tensor)} samples...")
            print(f"Input shape: {X_tensor.shape}")
            print(f"Device: {self.device}")
        
        # Create data loader
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.lr)
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        # Training loop
        self.flow_model.train()
        epoch_losses = []
        
        for epoch in range(self.epochs):
            losses = []
            for (batch,) in loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Negative log-likelihood
                log_prob = self.flow_model.log_prob(batch)
                loss = -log_prob.mean()  # Maximize log-likelihood = minimize negative log-likelihood
                
                loss.backward()
                
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.gradient_clip)
                
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            
            epoch_loss = np.mean(losses)
            epoch_losses.append(epoch_loss)
            
            # Early stopping check
            if self.patience is not None:
                if epoch_loss < best_loss - self.min_delta:
                    best_loss = epoch_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.flow_model.state_dict().items()}
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}/{self.epochs}")
                    break
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}")
        
        # Restore best model if early stopping was triggered
        if best_state is not None and self.patience is not None:
            self.flow_model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            if self.verbose:
                print(f"Restored best model with loss: {best_loss:.4f}")
        
        if self.verbose:
            print(f"Training completed. Final loss: {epoch_losses[-1]:.4f}")
        
        self.training_losses_ = epoch_losses
        return self
    
    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for test data.
        
        Anomaly score = negative log probability under the flow model.
        
        Args:
            X_test: Test data, shape (n_samples, n_features)
            
        Returns:
            Anomaly scores, shape (n_samples,)
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Compute negative log probability as anomaly score
        self.flow_model.eval()
        with torch.no_grad():
            log_prob = self.flow_model.log_prob(X_tensor)
            # Handle both per-sample and batch-sum log_prob
            if log_prob.dim() == 0:
                # Scalar (sum over batch) - shouldn't happen in eval mode
                scores = -log_prob.item() / len(X_tensor) * torch.ones(len(X_tensor))
            else:
                # Per-sample log probabilities
                scores = -log_prob
        
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

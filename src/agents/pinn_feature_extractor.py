"""
PINN Feature Extractor for DRL
==============================
Integrates PINN Heston parameters into DRL observation space.
Handles concatenation, normalization, and scaling of PINN features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PINNFeatureExtractor(nn.Module):
    """
    PyTorch module for integrating PINN features with technical indicators.
    
    Input: Concatenated [technical_features, heston_params]
    Output: Normalized, scaled observation for DRL agent
    
    Combines:
    - Technical indicators (RSI, MACD, SMA, ATR, etc.) [~10 features]
    - PINN Heston parameters (nu, theta, kappa, xi, rho) [5 features]
    """
    
    def __init__(
        self,
        num_technical_features: int = 10,
        num_pinn_features: int = 5,
        use_pinn_features: bool = True,
        pinn_feature_weights: Optional[Dict[str, float]] = None,
        normalize_method: str = 'z_score',
        clip_range: Tuple[float, float] = (-3.0, 3.0),
        verbose: bool = True
    ):
        """
        Initialize PINN feature extractor.
        
        Args:
            num_technical_features: Number of technical indicators
            num_pinn_features: Number of Heston parameters (always 5)
            use_pinn_features: If False, ignore PINN parameters
            pinn_feature_weights: Dict mapping param_name -> weight
                Ex: {'nu': 1.0, 'theta': 1.0, ..., 'rho': 0.8}
            normalize_method: 'z_score' or 'min_max'
            clip_range: Bounds for clipping normalized values
            verbose: Print logging info
        """
        super().__init__()
        
        self.num_technical_features = num_technical_features
        self.num_pinn_features = num_pinn_features
        self.use_pinn_features = use_pinn_features
        self.normalize_method = normalize_method
        self.clip_range = clip_range
        self.verbose = verbose
        
        # Set up feature weights
        if pinn_feature_weights is None:
            pinn_feature_weights = {k: 1.0 for k in ['nu', 'theta', 'kappa', 'xi', 'rho']}
        
        # Convert dict to array in order [nu, theta, kappa, xi, rho]
        weight_order = ['nu', 'theta', 'kappa', 'xi', 'rho']
        weights = np.array([pinn_feature_weights.get(k, 1.0) for k in weight_order])
        
        # Register as buffer (not trainable)
        self.register_buffer('pinn_scale', torch.from_numpy(weights).float())
        
        # Output dimension
        self.output_dim = num_technical_features + (num_pinn_features if use_pinn_features else 0)
        
        if verbose:
            logger.info(
                f"✅ PINNFeatureExtractor initialized:\n"
                f"   Tech features: {num_technical_features}\n"
                f"   PINN features: {num_pinn_features if use_pinn_features else 0}\n"
                f"   Output dim: {self.output_dim}\n"
                f"   PINN weights: {dict(zip(weight_order, weights))}"
            )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Extract and normalize features.
        
        Args:
            observation: Tensor of shape [batch_size, num_features]
                Where num_features = tech_features + pinn_features
        
        Returns:
            extracted_features: [batch_size, output_dim] normalized
        """
        batch_size = observation.shape[0]
        
        # Split technical and PINN features
        tech_features = observation[:, :self.num_technical_features]
        
        if self.use_pinn_features and observation.shape[1] > self.num_technical_features:
            pinn_features = observation[
                :,
                self.num_technical_features:self.num_technical_features + self.num_pinn_features
            ]
        else:
            pinn_features = None
        
        # Normalize technical features (Z-score)
        tech_normalized = self._normalize_features(tech_features, label='technical')
        
        # Scale and normalize PINN features
        if pinn_features is not None:
            pinn_scaled = self._scale_pinn_features(pinn_features)
            pinn_normalized = self._normalize_features(pinn_scaled, label='PINN')
            
            # Concatenate
            extracted = torch.cat([tech_normalized, pinn_normalized], dim=1)
        else:
            extracted = tech_normalized
        
        # Clip extreme values
        extracted = torch.clamp(extracted, self.clip_range[0], self.clip_range[1])
        
        return extracted
    
    def _normalize_features(
        self,
        features: torch.Tensor,
        label: str = ''
    ) -> torch.Tensor:
        """
        Normalize features to zero mean, unit variance.
        
        Args:
            features: [batch_size, num_features]
            label: Label for logging
        
        Returns:
            Normalized features [batch_size, num_features]
        """
        if self.normalize_method == 'z_score':
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            normalized = (features - mean) / (std + 1e-8)
        elif self.normalize_method == 'min_max':
            min_val = features.min(dim=0, keepdim=True).values
            max_val = features.max(dim=0, keepdim=True).values
            normalized = (features - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")
        
        return normalized
    
    def _scale_pinn_features(self, pinn_features: torch.Tensor) -> torch.Tensor:
        """
        Scale PINN features by learned weights.
        
        Args:
            pinn_features: [batch_size, 5] in order [nu, theta, kappa, xi, rho]
        
        Returns:
            Scaled features [batch_size, 5]
        """
        # Apply scaling
        pinn_scale = self.pinn_scale.to(pinn_features.device).to(pinn_features.dtype)
        scaled = pinn_features * pinn_scale.unsqueeze(0)
        
        return scaled
    
    def set_pinn_feature_weights(self, weights_dict: Dict[str, float]):
        """
        Update PINN feature weights dynamically.
        
        Args:
            weights_dict: Dict mapping param name to weight
                Ex: {'nu': 1.0, 'theta': 1.0, ..., 'rho': 0.8}
        """
        weight_order = ['nu', 'theta', 'kappa', 'xi', 'rho']
        new_weights = np.array([weights_dict.get(k, 1.0) for k in weight_order])
        
        self.pinn_scale.copy_(torch.from_numpy(new_weights).float())
        
        if self.verbose:
            logger.info(f"✅ Updated PINN feature weights: {weights_dict}")
    
    def get_pinn_feature_weights(self) -> Dict[str, float]:
        """Get current PINN feature weights."""
        weight_order = ['nu', 'theta', 'kappa', 'xi', 'rho']
        weights = self.pinn_scale.cpu().numpy()
        
        return dict(zip(weight_order, weights))


def create_observation_with_pinn(
    technical_obs: np.ndarray,
    pinn_features: Optional[np.ndarray] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Helper function to create combined observation vector.
    
    Args:
        technical_obs: [batch, num_tech_features] or [num_tech_features]
        pinn_features: [batch, 5] or [5] (nu, theta, kappa, xi, rho)
        normalize: Apply Z-score normalization
    
    Returns:
        combined_obs: [batch, num_features] with PINN features appended
    """
    
    # Handle 1D inputs
    if technical_obs.ndim == 1:
        technical_obs = technical_obs.reshape(1, -1)
    
    if pinn_features is not None:
        if pinn_features.ndim == 1:
            pinn_features = pinn_features.reshape(1, -1)
        
        # Ensure same batch size
        if technical_obs.shape[0] != pinn_features.shape[0]:
            # If PINN has different batch size, broadcast
            if pinn_features.shape[0] == 1:
                pinn_features = np.broadcast_to(
                    pinn_features,
                    (technical_obs.shape[0], pinn_features.shape[1])
                )
            else:
                raise ValueError(
                    f"Batch size mismatch: technical {technical_obs.shape[0]} "
                    f"vs PINN {pinn_features.shape[0]}"
                )
        
        # Concatenate
        combined = np.concatenate([technical_obs, pinn_features], axis=1)
    else:
        combined = technical_obs
    
    # Normalize
    if normalize:
        mean = combined.mean(axis=0, keepdims=True)
        std = combined.std(axis=0, keepdims=True)
        combined = (combined - mean) / (std + 1e-8)
    
    return combined.astype(np.float32)


def extract_pinn_features_from_obs(
    observation: np.ndarray,
    num_technical_features: int = 10,
    num_pinn_features: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split observation into technical and PINN components.
    
    Args:
        observation: [batch, num_features] concatenated observation
        num_technical_features: Number of technical indicators
        num_pinn_features: Number of PINN parameters
    
    Returns:
        (technical_obs, pinn_obs)
    """
    tech_obs = observation[:, :num_technical_features]
    pinn_obs = observation[:, num_technical_features:num_technical_features + num_pinn_features]
    
    return tech_obs, pinn_obs

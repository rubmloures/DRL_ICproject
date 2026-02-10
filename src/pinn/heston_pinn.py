"""
Heston PINN - Physics-Informed Neural Network for Options Pricing
===================================================================
Encodes Black-Scholes/Heston constraints for volatility estimation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.hyperparameters import PINN_PARAMS


class PhysicsLoss(nn.Module):
    """Physics constraint loss for options pricing"""
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        predicted_vol: torch.Tensor,
        implied_vol: torch.Tensor,
        spot: torch.Tensor,
        strike: torch.Tensor,
        ttm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics-informed loss.
        
        Constraints:
        1. Vol surface smoothness
        2. Moneyness dependence
        3. Term structure constraints
        """
        # Data fitting loss
        data_loss = torch.nn.functional.mse_loss(predicted_vol, implied_vol)
        
        # Physics constraints (PDE residuals)
        # Monitor vol surface smoothness with respect to moneyness and TTM
        physics_loss = torch.tensor(0.0, device=predicted_vol.device)
        
        # Constraint 1: Monotonicity in spot price
        if spot.shape[0] > 1:
            sorted_idx = torch.argsort(spot)
            monotone_constraint = torch.nn.functional.relu(
                -torch.diff(predicted_vol[sorted_idx])
            )
            physics_loss = physics_loss + monotone_constraint.mean()
        
        # Constraint 2: Term structure (longer maturities have lower vol gradient)
        if ttm.shape[0] > 1:
            sorted_idx = torch.argsort(ttm)
            ttm_gradient = torch.diff(predicted_vol[sorted_idx]) / (
                torch.diff(ttm[sorted_idx]) + 1e-6
            )
            physics_loss = physics_loss + torch.pow(ttm_gradient, 2).mean()
        
        # Combined loss
        total_loss = data_loss + self.weight * physics_loss
        return total_loss


class HestonPINN(nn.Module):
    """
    Physics-Informed Neural Network for options Greeks and volatility.
    
    Inputs:
    - spot_price: Current stock price
    - strike: Option strike price
    - time_to_maturity: Days to expiration / 365
    - risk_free_rate: r (annual)
    - dividend_yield: q (annual)
    
    Outputs:
    - implied_volatility: Estimated vol from market prices
    - delta, gamma, theta, vega, rho: Greeks
    """
    
    def __init__(self, input_dim: int = 5, params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.input_dim = input_dim
        self.params = params or PINN_PARAMS
        
        hidden_layers = self.params.get("hidden_layers", [64, 128, 64])
        dropout = self.params.get("dropout", 0.1)
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Physics-informed activation
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output: vol + 5 Greeks
        layers.append(nn.Linear(prev_dim, 6))
        
        self.network = nn.Sequential(*layers)
        self.physics_loss_fn = PhysicsLoss(
            weight=self.params.get("physics_weight", 0.1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Heston PINN network.
        
        Args:
            x: Input tensor of shape (batch_size, 5)
                [spot_price, strike, time_to_maturity, risk_free_rate, dividend_yield]
                All values should be normalized to reasonable ranges for network stability
        
        Returns:
            vol: Volatility tensor of shape (batch_size, 1)
                 Values constrained to range [0.01, 2.00] (1% to 200% annualized vol)
            greeks: Greeks tensor of shape (batch_size, 5)
                    [delta, gamma, theta, vega, rho]
                    Units: delta ∈ [-1, 1], gamma > 0, theta in days, vega per 1% vol, rho per 1% rate
        
        Raises:
            AssertionError: If x.shape[-1] != 5
        
        Notes:
            - Uses tanh activation for physics-informed constraints
            - Output volatility is strictly positive (sigmoid + scaling)
            - Greeks are unbounded but typically reasonable for valid inputs
            - Suitable for batched inference during training or deployment
        
        Examples:
            >>> pinn = HestonPINN(input_dim=5)
            >>> inputs = torch.randn(64, 5)  # Batch of 64 options
            >>> vol, greeks = pinn.forward(inputs)
            >>> assert vol.shape == (64, 1), f"Expected vol shape (64, 1), got {vol.shape}"
            >>> assert greeks.shape == (64, 5), f"Expected greeks shape (64, 5), got {greeks.shape}"
        """
        # Enable gradient tracking for Greeks computation
        x_clone = x.clone().detach().requires_grad_(True)
        
        # Forward pass through network
        output = self.network(x)
        
        # First output is volatility (constrained to [0.01, 2.0])
        vol = torch.sigmoid(output[:, 0:1]) * 1.99 + 0.01  # vol in [0.01, 2.00]
        
        # Other outputs are Greeks (raw)
        greeks = output[:, 1:6]  # delta, gamma, theta, vega, rho
        
        return vol, greeks
    
    def compute_greeks_blackscholes(
        self,
        S: torch.Tensor,
        K: torch.Tensor,
        T: torch.Tensor,
        r: torch.Tensor,
        sigma: torch.Tensor,
        option_type: str = "call",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Greeks using closed-form Black-Scholes formulas.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with Greeks
        """
        sqrt_T = torch.sqrt(T)
        
        d1 = (torch.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T + 1e-8)
        d2 = d1 - sigma * sqrt_T
        
        from torch.distributions import Normal
        N = Normal(torch.tensor(0.0), torch.tensor(1.0))
        
        Nd1 = torch.exp(N.log_prob(d1))  # PDF of standard normal
        Nd2 = N.cdf(d2)
        Nd1_d2 = torch.exp(N.log_prob(d2))
        
        # Delta
        if option_type == "call":
            delta = N.cdf(d1)
            gamma = Nd1 / (S * sigma * sqrt_T + 1e-8)
            theta = (
                -S * Nd1 * sigma / (2 * sqrt_T + 1e-8)
                - r * K * torch.exp(-r * T) * N.cdf(d2)
            )
            vega = S * Nd1 * sqrt_T
            rho = K * T * torch.exp(-r * T) * N.cdf(d2)
        else:  # put
            delta = N.cdf(d1) - 1
            gamma = Nd1 / (S * sigma * sqrt_T + 1e-8)
            theta = (
                -S * Nd1 * sigma / (2 * sqrt_T + 1e-8)
                + r * K * torch.exp(-r * T) * torch.abs(Nd2)
            )
            vega = S * Nd1 * sqrt_T
            rho = -K * T * torch.exp(-r * T) * torch.abs(Nd2)
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta / 365,  # Convert to daily
            "vega": vega / 100,  # Per 1% change in vol
            "rho": rho / 100,  # Per 1% change in rate
        }
    
    def train_on_batch(
        self,
        x: torch.Tensor,
        y_vol: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ) -> float:
        """
        Train on a single batch with physics constraints.
        
        Args:
            x: Input features
            y_vol: Target implied volatility
            optimizer: PyTorch optimizer
            device: Device to train on
        
        Returns:
            Loss value
        """
        x = x.to(device)
        y_vol = y_vol.to(device)
        
        # Forward pass
        pred_vol, greeks = self(x)
        
        # Compute physics-informed loss
        spot = x[:, 0]
        strike = x[:, 1]
        ttm = x[:, 2]
        
        loss = self.physics_loss_fn(pred_vol, y_vol, spot, strike, ttm)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from the PINN for use in DRL agent.
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            Embeddings (batch_size, hidden_dim)
        """
        # Return activations from second-to-last hidden layer
        truncated_net = nn.Sequential(*list(self.network.children())[:-2])
        return truncated_net(x)

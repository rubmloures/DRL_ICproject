"""
DRL Stock Trading Agent with PINN Integration - Hyperparameters
================================================================
Neural network and training hyperparameters.
"""

from typing import Dict, Any

# =============================================================================
# PPO Agent Hyperparameters
# =============================================================================
PPO_PARAMS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# =============================================================================
# DDPG Agent Hyperparameters
# =============================================================================
DDPG_PARAMS: Dict[str, Any] = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "learning_starts": 1_000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.001,
    "action_noise": 0.2,
    "verbose": 1,
}

# =============================================================================
# A2C Agent Hyperparameters
# =============================================================================
A2C_PARAMS: Dict[str, Any] = {
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 0.98,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "verbose": 1,
}

# =============================================================================
# PINN Architecture
# =============================================================================
PINN_PARAMS: Dict[str, Any] = {
    "hidden_layers": [64, 128, 64],
    "activation": "tanh",
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "physics_weight": 0.1,  # Weight for physics loss term
    "epochs": 100,
    "batch_size": 256,
}

# =============================================================================
# Feature Extractor (for PPO)
# =============================================================================
FEATURE_EXTRACTOR_PARAMS: Dict[str, Any] = {
    "net_arch": [256, 256],
    "activation_fn": "relu",
}

# =============================================================================
# PINN Feature Integration
# =============================================================================
PINN_FEATURE_WEIGHTS: Dict[str, float] = {
    'nu': 1.0,          # Instantaneous variance
    'theta': 1.0,       # Long-term variance
    'kappa': 1.0,       # Mean reversion speed
    'xi': 1.0,          # Volatility of volatility
    'rho': 0.8,         # Spot-volatility correlation (bounded [-1,1], use lower weight)
}

PINN_FEATURES_NORMALIZATION = {
    'method': 'z_score',        # 'z_score' or 'min_max'
    'independent': False,        # If False, use global stats from training
    'clip_range': (-3.0, 3.0),  # Clip extreme values
}

# =============================================================================
# A/B Testing Configuration
# =============================================================================
AB_TESTING_CONFIG: Dict[str, Any] = {
    'enabled': True,                      # Enable A/B testing (with/without PINN)
    'model_a_pinn': False,               # Model A: without PINN features
    'model_b_pinn': True,                # Model B: with PINN features
    'same_seeds': True,                  # Use same random seeds for fair comparison
    'statistical_test': 'ttest',         # 'ttest' or 'mann_whitney'
    'significance_level': 0.05,          # Alpha for statistical significance
    'expected_improvement': 0.10,        # Expected improvement (10% Sharpe lift)
}

# =============================================================================
# Composite Reward Configuration (Dynamic Weighting with PINN)
# =============================================================================
COMPOSITE_REWARD_CONFIG: Dict[str, Any] = {
    'enabled': False,                    # Enable composite reward function (requires PINN)
    'base_weights': {
        'excess_return': 1.0,           # Weight for excess return (portfolio - Selic)
        'downside_risk': 1.0,           # Weight for downside risk penalty (Sortino)
        'alpha_return': 1.0,            # Weight for alpha (beating Ibovespa)
        'transaction_cost': 1.0,        # Weight for transaction cost penalty
    },
    'selic_daily_rate': 0.00025,        # Daily risk-free rate (~9%/252 default)
    'transaction_cost_pct': 0.0015,     # Transaction cost % (corretagem + spread)
    'emolumentos_pct': 0.0005,          # B3 emolumentos %
    'window_size': 20,                  # Rolling window for vol calculations
    
    # Regime Detection Thresholds
    'regime_thresholds': {
        'stable_vol': 0.12,             # Max nu (mean vol) for stable regime
        'high_vol_of_vol': 0.4,         # Min xi (vol-of-vol) for turbulence
        'negative_correlation': -0.5,   # Min rho for crisis detection
        'percentile_high_vol': 0.75,    # Percentile for historical vol comparison
    },
    
    # Dynamic weight multipliers per regime
    'regime_multipliers': {
        'stable_trending': {
            'excess_return': 1.0,
            'downside_risk': 0.5,       # Tolerate vol in uptrends
            'alpha_return': 1.5,        # Aggressive alpha chasing
            'transaction_cost': 0.8,    # Allow more trading
        },
        'normal_ranging': {
            'excess_return': 1.0,
            'downside_risk': 1.0,       # Normal risk aversion
            'alpha_return': 1.0,        # Balanced
            'transaction_cost': 1.0,    # Standard
        },
        'elevated_volatility': {
            'excess_return': 0.8,
            'downside_risk': 1.5,       # Higher penalty
            'alpha_return': 0.8,        # Reduce aggressive moves
            'transaction_cost': 1.2,    # Reduce trading
        },
        'turbulent_shock': {
            'excess_return': 0.3,       # Minimal return focus
            'downside_risk': 3.0,       # VERY high risk aversion
            'alpha_return': 0.1,        # Forget about alpha
            'transaction_cost': 2.0,    # Preserve capital priority
        },
    },
}

# =============================================================================
# Training Configuration
# =============================================================================
TRAINING_CONFIG: Dict[str, Any] = {
    "total_timesteps": 100_000,
    "eval_freq": 10_000,
    "n_eval_episodes": 5,
    "save_freq": 25_000,
    "log_interval": 10,
}

# =============================================================================
# Environment Parameters
# =============================================================================
ENV_PARAMS: Dict[str, Any] = {
    "lookback_window": 30,      # Days of history to include
    "reward_scaling": 1e-4,     # Scale rewards for stability
    "max_position": 1.0,        # Max position size (fraction of portfolio)
    "normalize_obs": True,
    "normalize_reward": True,
}

# =============================================================================
# Training Configuration (Timeouts, Checkpointing)
# =============================================================================
TRAINING_CONFIG: Dict[str, Any] = {
    "timeout_seconds": 600,         # Training timeout (0 = no timeout)
    "checkpoint_interval": 10_000,  # Save checkpoint every N timesteps
    "keep_checkpoints": 3,          # Keep last N checkpoints
    "early_stopping_patience": 10,  # Episodes without improvement before stopping
}

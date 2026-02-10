"""
Core Constants and Shared Configuration
========================================
Centralized constants used across the entire project.
"""

from enum import Enum
from typing import Dict, Any

# =============================================================================
# Market Constants (B3 Brasil)
# =============================================================================

# Transaction costs (realistic for B3 normal accounts)
DEFAULT_TRANSACTION_COST = 0.0003  # 0.03%
DEFAULT_SLIPPAGE = 0.0001  # 0.01%
DEFAULT_MIN_TRANSACTION = 100.0  # R$ 100 minimum

# Position constraints
DEFAULT_MAX_POSITION_SIZE = 0.3  # Max 30% per stock
DEFAULT_LEVERAGE = 1.0  # No leverage for retail

# =============================================================================
# Time Configurations
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
SELIC_TO_DAILY_RATE = 1.0 / TRADING_DAYS_PER_YEAR

# =============================================================================
# Data Columns Mapping
# =============================================================================

OHLCV_COLUMNS = [
    'acao_open',
    'acao_high', 
    'acao_low',
    'acao_close_ajustado',
    'acao_vol_fin',
]

GREEK_COLUMNS = [
    'delta',
    'gamma',
    'theta',
    'vega',
    'rho',
]

OPTIONS_COLUMNS = [
    'spot_price',
    'strike',
    'premium',
    'days_to_maturity',
    'moneyness',
] + GREEK_COLUMNS

TECHNICAL_INDICATORS = [
    'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_signal', 'MACD_hist',
    'RSI_14',
    'BB_upper', 'BB_middle', 'BB_lower',
    'ATR_14',
    'volume_sma_20',
    'returns', 'volatility_20'
]

# =============================================================================
# Moneyness and Option Type Encoding
# =============================================================================

class Moneyness(Enum):
    """Moneyness categories."""
    ITM = 0  # In The Money
    ATM = 1  # At The Money
    OTM = 2  # Out of The Money

class OptionType(Enum):
    """Option types."""
    CALL = 0
    PUT = 1

# Convert to dict for mapping
MONEYNESS_MAP = {e.name: e.value for e in Moneyness}
MONEYNESS_REVERSE = {e.value: e.name for e in Moneyness}

OPTION_TYPE_MAP = {e.name: e.value for e in OptionType}
OPTION_TYPE_REVERSE = {e.value: e.name for e in OptionType}

# =============================================================================
# DRL Algorithm Parameters (Default)
# =============================================================================

DEFAULT_PPO_PARAMS: Dict[str, Any] = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

DEFAULT_DDPG_PARAMS: Dict[str, Any] = {
    "learning_rate": 1e-3,
    "buffer_size": 100_000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": (1, "episode"),
    "gradient_steps": -1,
}

DEFAULT_A2C_PARAMS: Dict[str, Any] = {
    "learning_rate": 7e-4,
    "n_steps": 5,
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

# =============================================================================
# PINN Parameters (Default)
# =============================================================================

DEFAULT_PINN_PARAMS: Dict[str, Any] = {
    "hidden_layers": [64, 128, 64],
    "activation": "tanh",
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "physics_weight": 0.1,
    "epochs": 100,
    "batch_size": 256,
}

# =============================================================================
# Environment Parameters (Default)
# =============================================================================

DEFAULT_ENV_PARAMS: Dict[str, Any] = {
    "lookback_window": 30,
    "reward_scaling": 1e-4,
    "max_position": DEFAULT_MAX_POSITION_SIZE,
    "normalize_obs": True,
    "normalize_reward": True,
    "transaction_cost": DEFAULT_TRANSACTION_COST,
    "slippage": DEFAULT_SLIPPAGE,
}

# =============================================================================
# Hyperparameter Tuning (Optuna)
# =============================================================================

DEFAULT_OPTUNA_PARAMS: Dict[str, Any] = {
    "n_trials": 50,
    "timeout": None,
    "n_jobs": 1,  # Set to -1 for parallel execution
    "sampler": "tpe",  # Tree-structured Parzen Estimator
    "pruner": "median",
    "direction": "maximize",  # Maximize Sharpe ratio
}

# =============================================================================
# Logging and Debugging
# =============================================================================

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"

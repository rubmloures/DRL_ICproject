"""
DRL Stock Trading Agent - Configuration
========================================
Global parameters for the trading system.
Supports rolling window backtesting and multi-agent ensemble strategies.
"""

from pathlib import Path
from typing import List, Dict, Literal

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAINED_MODELS = PROJECT_ROOT / "trained_models"
RESULTS = PROJECT_ROOT / "results"

# =============================================================================
# Assets Configuration
# =============================================================================
ASSETS: List[str] = [
    "PETR4",
    "VALE3", 
    "BOVA11",
    "BBAS3",
    "MGLU3",
    "CSNA3",
    "B3SA3",
    "ABEV3",
]

# Primary assets for initial training
PRIMARY_ASSETS: List[str] = ["PETR4", "VALE3"]

# =============================================================================
# Time Configuration (Generic - Works with any market)
# =============================================================================
# Global date range for all data
GLOBAL_START = "2019-01-01"
GLOBAL_END = "2025-12-31"

# Temporal splits: Train (≤2023), Test (2024), Validation (2025+)
TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31"      # ← Train/Fine-tune until end of 2023
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"        # ← Test entire 2024
VAL_START = "2025-01-01"
VAL_END = "2025-12-31"         # ← Validation in 2025

# Current strategy uses rolling window (see ROLLING_WINDOW_CONFIG)
# For fixed splits, use TRAIN_START/END and TEST_START/END above

# =============================================================================
# Rolling Window Configuration (Janela Deslizante)
# =============================================================================
# Strategy: 14 weeks training, 4 weeks testing, 2 weeks overlap
# With K-fold validation inside each train window
ROLLING_WINDOW_CONFIG = {
    'train_weeks': 14,              # ~14 * 5 = 70 trading days
    'test_weeks': 4,                # ~4 * 5 = 20 trading days
    'overlap_weeks': 2,             # Overlap between consecutive windows
    'enabled': True,                # Enable rolling window strategy
    'with_validation_fold': True,   # Enable K-fold validation inside train
    'k_fold': 3,                    # Number of folds for cross-validation
}

# Alternative: If False, use fixed train/test split above

# =============================================================================
# Trading Parameters
# =============================================================================
INITIAL_CAPITAL = 100_000.0     # Starting capital (currency-agnostic)
TRANSACTION_COST = 0.0005       # 0.05% per trade (generic market)
SLIPPAGE = 0.0001               # 0.01% slippage

# =============================================================================
# Feature Configuration
# =============================================================================

# PINN Input Features (from options data)
PINN_FEATURES: List[str] = [
    "spot_price",
    "strike", 
    "premium",
    "days_to_maturity",
    "moneyness",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
]

# Market View Features (from stock data)
MARKET_FEATURES: List[str] = [
    "acao_open",
    "acao_high",
    "acao_low",
    "acao_close_ajustado",
    "acao_vol_fin",
]

# Technical Indicators to Calculate
TECHNICAL_INDICATORS: List[str] = [
    "MACD",
    "RSI",
    "BBands",
    "ATR",
    "SMA_20",
    "SMA_50",
    "EMA_12",
    "EMA_26",
]

# =============================================================================
# Data Parsing Configuration (Generic - Auto-detects format)
# =============================================================================
# Supported separators: ';' (European), ',' (US), '\t' (Tab)
# Supported decimals: ',' (European), '.' (US)
# Auto-detection enabled in DataLoader
#
# Example formats:
#   European: data;ticker;acao_close_ajustado
#             11/09/2025;PETR4;"1.234,56"
#   US:       data,ticker,price
#             2025-09-11,PETR4,1234.56

CSV_SEPARATOR = ";"             # Can be auto-detected
DECIMAL_SEPARATOR = ","         # Can be auto-detected

# =============================================================================
# Ensemble Configuration
# =============================================================================
ENSEMBLE_CONFIG = {
    'enabled': True,
    'algorithms': ['PPO', 'DDPG', 'A2C'],  # Algorithms to ensemble
    'voting_strategy': 'weighted',           # 'mean', 'weighted', 'majority', 'best'
    'initial_weights': {                     # Equal weights initially
        'PPO': 0.333,
        'DDPG': 0.333,
        'A2C': 0.334,
    },
}

# =============================================================================
# PINN Configuration (Physics-Informed Neural Networks - Heston Model)
# =============================================================================
PINN_ENABLED = True
PINN_CHECKPOINT_PATH = str(PROJECT_ROOT / "repo_contex" / "PINN" / "resultados" / "modelo_final" / "best_hybrid_model.pth")
PINN_DATA_STATS_PATH = str(PROJECT_ROOT / "repo_contex" / "PINN" / "resultados" / "modelo_final" / "data_stats.json")
PINN_WINDOW_SIZE = 30                   # 30-day sliding window for LSTM
PINN_BATCH_SIZE_INFERENCE = 256         # Batch size for PINN inference
PINN_DEVICE = "cuda"                    # 'cuda' or 'cpu'
PINN_DTYPE = "float32"                  # 'float32' or 'float64'
PINN_INPUT_VALIDATION = True            # Validate inputs against training ranges
PINN_ERROR_HANDLING = "fill_zeros"      # 'fill_zeros' or 'raise' if inference fails
PINN_CACHE_SIZE = 1000                  # LRU cache for inference results

PINN_CONFIG = {
    'enabled': PINN_ENABLED,
    'use_options_data': True,                           # Use options for Greeks
    'greek_features': ['delta', 'gamma', 'vega', 'theta', 'rho'],
    'model_type': 'DeepHestonHybrid',                   # LSTM + PINN for Heston
    'heston_params': ['nu', 'theta', 'kappa', 'xi', 'rho'],  # Stochastic Vol params
    'window_size': PINN_WINDOW_SIZE,
    'batch_size': PINN_BATCH_SIZE_INFERENCE,
}

# =============================================================================
# Moneyness Categories
# =============================================================================
MONEYNESS_MAP: Dict[str, int] = {
    "ITM": 0,  # In The Money
    "ATM": 1,  # At The Money
    "OTM": 2,  # Out of The Money
}

# Option Type
OPTION_TYPE_MAP: Dict[str, int] = {
    "CALL": 0,
    "PUT": 1,
}

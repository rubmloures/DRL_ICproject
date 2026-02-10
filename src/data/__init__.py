"""
Data Layer - ETL and Preprocessing
===================================
Handles data ingestion, cleaning, and feature engineering following FinRL architecture.

Modules:
- data_loader.py: Raw data loading (CSV parsing, multi-format support)
- data_processor.py: Feature engineering (technical indicators, normalization, options data)
- rolling_window.py: Rolling window cross-validation strategy for backtesting
  (14 weeks training + 4 weeks testing, configurable overlap)

Supports:
- Multiple CSV formats (European: ;, American: ,)
- Multiple decimal formats (European: comma, American: dot)
- Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, ADX)
- Options data with Greeks (via PINN)
- Multi-asset portfolios
"""

from .data_loader import DataLoader
from .data_processor import DataProcessor
from .rolling_window import RollingWindowStrategy, ExpandingWindowStrategy

__all__ = [
    "DataLoader", 
    "DataProcessor",
    "RollingWindowStrategy",
    "ExpandingWindowStrategy",
]

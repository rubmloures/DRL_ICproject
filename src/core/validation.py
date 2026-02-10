"""
Input Validation & Configuration
=================================

Uses Pydantic for robust input validation across the system.
Catches errors BEFORE expensive RL training starts.

Usage:
    from src.core.validation import EnvironmentConfig, validate_dataframe
    
    config = EnvironmentConfig(
        df=my_df,
        stock_dim=3,
        initial_amount=100_000,
    )
    validate_dataframe(config.df, required_columns=['time', 'acao_close_ajustado'])
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


class EnvironmentConfig(BaseModel):
    """
    Validation for trading environment configuration.
    
    Examples:
        >>> config = EnvironmentConfig(
        ...     df=data_df,
        ...     stock_dim=3,
        ...     initial_amount=100_000,
        ...     buy_cost_pct=0.0005
        ... )
    """
    
    stock_dim: int
    initial_amount: float = 100_000.0
    buy_cost_pct: float = 0.0005
    sell_cost_pct: float = 0.0005
    reward_scaling: float = 1e-4
    hmax: int = 100
    
    @field_validator('stock_dim')
    @classmethod
    def stock_dim_positive(cls, v):
        if v <= 0:
            raise ValueError('stock_dim must be > 0')
        if v > 100:
            raise ValueError('stock_dim must be <= 100 (too large for single agent)')
        return v
    
    @field_validator('initial_amount')
    @classmethod
    def initial_amount_positive(cls, v):
        if v <= 0:
            raise ValueError('initial_amount must be > 0')
        if v > 1e10:
            raise ValueError('initial_amount unreasonably large (> 10B), check input')
        return v
    
    @field_validator('buy_cost_pct', 'sell_cost_pct')
    @classmethod
    def cost_in_range(cls, v):
        if not (0 <= v <= 0.1):  # 0% to 10% max
            raise ValueError(f'cost_pct must be in [0, 0.1], got {v}')
        return v
    
    @field_validator('hmax')
    @classmethod
    def hmax_positive(cls, v):
        if v <= 0:
            raise ValueError('hmax (max shares per trade) must be > 0')
        return v


class DataFrameValidator(BaseModel):
    """
    Validation for input DataFrame.
    
    Examples:
        >>> validator = DataFrameValidator(df=data_df)
        >>> validator.validate_columns(['time', 'acao_close_ajustado'])
    """
    
    df: Any  # Accept Any because pydantic doesn't handle pd.DataFrame well
    
    class Config:
        arbitrary_types_allowed = True
    
    @field_validator('df')
    @classmethod
    def df_is_dataframe(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise TypeError(f'df must be pd.DataFrame, got {type(v)}')
        return v
    
    @field_validator('df')
    @classmethod
    def df_not_empty(cls, v):
        if v.empty:
            raise ValueError('df must not be empty')
        return v
    
    def validate_columns(self, required_columns: List[str]) -> bool:
        """
        Validate that required columns exist.
        
        Args:
            required_columns: List of column names that must exist
        
        Returns:
            True if validation passes
        
        Raises:
            ValueError: If columns are missing
        """
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            available = list(self.df.columns)
            raise ValueError(
                f"DataFrame missing required columns: {missing}\n"
                f"Available columns: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
        return True
    
    def validate_dtypes(self, type_map: Dict[str, str]) -> bool:
        """
        Validate column data types.
        
        Args:
            type_map: Dict mapping column names to expected dtype strings
                     (e.g., {'time': 'datetime64', 'close': 'float64'})
        
        Returns:
            True if validation passes
        
        Raises:
            TypeError: If column types don't match
        """
        for col, expected_type in type_map.items():
            if col not in self.df.columns:
                continue
            
            if expected_type == 'float' or expected_type == 'float64':
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    raise TypeError(
                        f"Column {col} must be numeric, got {self.df[col].dtype}"
                    )
            elif expected_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    raise TypeError(
                        f"Column {col} must be datetime, got {self.df[col].dtype}"
                    )
            elif expected_type == 'int':
                if not pd.api.types.is_integer_dtype(self.df[col]):
                    raise TypeError(
                        f"Column {col} must be integer, got {self.df[col].dtype}"
                    )
        
        return True
    
    def validate_no_nulls(self, columns: Optional[List[str]] = None) -> bool:
        """
        Validate that critical columns have no NaN values.
        
        Args:
            columns: List of columns to check (default: all numeric columns)
        
        Returns:
            True if no nulls found
        
        Raises:
            ValueError: If nulls are found
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                if null_count > 0:
                    raise ValueError(
                        f"Column {col} contains {null_count} NaN values "
                        f"({null_count/len(self.df)*100:.1f}% of data)"
                    )
        
        return True


def validate_input_safety(
    df: pd.DataFrame,
    stock_dim: int,
    initial_amount: float,
    required_columns: Optional[List[str]] = None
) -> bool:
    """
    Quick validation function to check all inputs before training.
    
    Args:
        df: DataFrame to validate
        stock_dim: Number of stocks
        initial_amount: Initial capital
        required_columns: Required DataFrame columns
    
    Returns:
        True if all validations pass
    
    Raises:
        Various validation errors if inputs are invalid
    
    Examples:
        >>> validate_input_safety(
        ...     df=data_df,
        ...     stock_dim=3,
        ...     initial_amount=100_000,
        ...     required_columns=['time', 'acao_close_ajustado']
        ... )
    """
    # Validate environment config
    env_config = EnvironmentConfig(
        stock_dim=stock_dim,
        initial_amount=initial_amount
    )
    logger.debug("✓ EnvironmentConfig validation passed")
    
    # Validate DataFrame
    df_validator = DataFrameValidator(df=df)
    logger.debug("✓ DataFrame is valid pandas.DataFrame")
    
    # Check required columns
    if required_columns:
        df_validator.validate_columns(required_columns)
        logger.debug(f"✓ All required columns present: {required_columns}")
    
    # Check for critical NaNs
    df_validator.validate_no_nulls()
    logger.debug("✓ No NaN values in numeric columns")
    
    logger.info("✓ All input validations passed!")
    return True

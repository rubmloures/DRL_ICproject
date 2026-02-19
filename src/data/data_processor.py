"""
Data Processor - Feature Engineering and Preprocessing
=======================================================
Follows FinRL architecture: clean data, add technical indicators, normalize.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas_ta not installed. Limited technical indicators available.")


class DataProcessor:
    """
    Preprocesses raw market data for training.
    
    Responsibilities:
    - Clean data (remove duplicates, handle missing values)
    - Calculate technical indicators (MACD, RSI, Bollinger, ATR)
    - Normalize features
    - Aggregate options data
    - Prepare train/test splits
    
    Follows the Data Pipeline pattern from FinRL.
    """
    
    def __init__(self):
        """Initialize data processor with empty scalers."""
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_stats: Dict[str, Dict] = {}
        self.logger = logging.getLogger(__name__)
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data: remove duplicates, handle missing values.
        
        Args:
            df: Raw dataframe
        
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        # Remove complete duplicates
        df = df.drop_duplicates()
        logger.debug(f"Removed duplicates: {len(df.drop_duplicates())} rows remain")
        
        # Identify date column
        date_cols = [c for c in ['time', 'date', 'data'] if c in df.columns]
        if date_cols:
            date_col = date_cols[0]
            df = df.sort_values(by=date_col)
        
        # Forward fill missing values (for market continuity)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                df[col] = df[col].ffill().bfill()
                logger.debug(f"Filled {missing_count} missing values in {col}")
        
        return df
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        include_indicators: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data.
        
        Uses pandas_ta library if available, otherwise falls back to manual calculation.
        
        Supports multiple column naming schemes:
        - Original: acao_open, acao_high, acao_low, acao_close_ajustado, acao_vol_fin
        - Aggregated: spot_price_mean/min/max, close (from aggregation)
        - Merged: open, high, low, close, volume (post-merge)
        
        Args:
            df: DataFrame with OHLCV columns (any naming scheme)
            include_indicators: List of indicators to add (default: all)
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # Default indicators
        if include_indicators is None:
            include_indicators = [
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'MACD', 'RSI', 'BBands', 'ATR',
                'volume_sma_20'
            ]
        
        # ========================================================================
        # PHASE 1: Identify OHLCV columns (try multiple naming schemes)
        # ========================================================================
        
        # Scheme 1: Original B3 naming
        ohlcv_map_original = {
            'acao_open': 'open',
            'acao_high': 'high',
            'acao_low': 'low',
            'acao_close_ajustado': 'close',
            'acao_vol_fin': 'volume',
        }
        
        # Scheme 2: After options aggregation (spot_price_* columns)
        ohlcv_map_aggregated = {
            'spot_price_max': 'high',      # High: max of spot prices
            'spot_price_min': 'low',       # Low: min of spot prices
            'spot_price_mean': 'close',    # Close: mean of spot prices
        }
        
        # Scheme 3: Already standardized (open, high, low, close, volume)
        if 'open' in df.columns and 'close' in df.columns:
            # Columns are already standardized - no renaming needed
            available_renames = {}
            # Ensure all required columns exist with sensible defaults
            if 'high' not in df.columns:
                df['high'] = df.get('spot_price_max', df.get('close', df.iloc[:, 0]))
            if 'low' not in df.columns:
                df['low'] = df.get('spot_price_min', df.get('close', df.iloc[:, 0]))
            if 'volume' not in df.columns:
                df['volume'] = df.get('acao_vol_fin_mean', 1.0)
            # Mark as valid - columns are already standard
            available_renames = {'open': 'open', 'close': 'close'}  # Sentinel: indicates columns are ready
        else:
            # Try scheme 1 (original B3 naming)
            available_renames = {k: v for k, v in ohlcv_map_original.items() if k in df.columns}
            
            # If not found, try scheme 2 (aggregated spot_price_*)
            if not available_renames:
                available_renames = {k: v for k, v in ohlcv_map_aggregated.items() if k in df.columns}
                
                # Ensure we have all required columns with fallbacks
                if 'spot_price_mean' in df.columns or 'close' in df.columns:
                    close_col = 'close' if 'close' in df.columns else 'spot_price_mean'
                    high_col = 'spot_price_max' if 'spot_price_max' in df.columns else close_col
                    low_col = 'spot_price_min' if 'spot_price_min' in df.columns else close_col
                    vol_col = 'acao_vol_fin_mean' if 'acao_vol_fin_mean' in df.columns else 'volume'
                    
                    available_renames = {
                        close_col: 'close',
                        high_col: 'high',
                        low_col: 'low',
                        vol_col: 'volume'
                    }
                    # Remove duplicates (fallback might cause duplicates)
                    available_renames = {k: v for k, v in available_renames.items() if k in df.columns}
        
        if not available_renames:
            logger.warning(
                f"No OHLCV columns found. Available: {list(df.columns)[:10]}... "
                "Skipping technical indicators."
            )
            return df
        
        # Handle already-standardized columns (Scheme 3)
        if available_renames.get('open') == 'open' and available_renames.get('close') == 'close':
            # Columns are already standard - use df directly
            logger.info(f"✓ OHLCV columns already standardized: open, high, low, close, volume")
            df_ta = df
        else:
            # Need to rename
            logger.info(f"Detected OHLCV columns: {list(available_renames.keys())}")
            df_ta = df.rename(columns=available_renames)
        
        if not HAS_PANDAS_TA:
            logger.warning("pandas_ta not available. Computing basic indicators only.")
            return self._add_basic_indicators(df_ta)
        
        # ===== SMA =====
        if 'SMA_20' in include_indicators and 'close' in df_ta.columns:
            df['SMA_20'] = ta.sma(df_ta['close'], length=20)
        if 'SMA_50' in include_indicators and 'close' in df_ta.columns:
            df['SMA_50'] = ta.sma(df_ta['close'], length=50)
        
        # ===== EMA =====
        if 'EMA_12' in include_indicators and 'close' in df_ta.columns:
            df['EMA_12'] = ta.ema(df_ta['close'], length=12)
        if 'EMA_26' in include_indicators and 'close' in df_ta.columns:
            df['EMA_26'] = ta.ema(df_ta['close'], length=26)
        
        # ===== MACD =====
        if 'MACD' in include_indicators and 'close' in df_ta.columns:
            macd = ta.macd(df_ta['close'], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)
        
        # ===== RSI =====
        if 'RSI' in include_indicators and 'close' in df_ta.columns:
            rsi = ta.rsi(df_ta['close'], length=14)
            if rsi is not None:
                df['RSI_14'] = rsi
        
        # ===== Bollinger Bands =====
        if 'BBands' in include_indicators and 'close' in df_ta.columns:
            bbands = ta.bbands(df_ta['close'], length=20, std=2)
            if bbands is not None:
                df = pd.concat([df, bbands], axis=1)
        
        # ===== ATR =====
        if 'ATR' in include_indicators:
            if all(c in df_ta.columns for c in ['high', 'low', 'close']):
                atr = ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
                if atr is not None:
                    df['ATR_14'] = atr
        
        # ===== Volume Indicators =====
        if 'volume_sma_20' in include_indicators and 'volume' in df_ta.columns:
            df['volume_sma_20'] = ta.sma(df_ta['volume'], length=20)
        
        # ===== Basic Return and Volatility =====
        if 'close' in df_ta.columns:
            df['returns'] = df_ta['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['log_returns'] = np.log(df_ta['close'] / df_ta['close'].shift(1))
        
        # Fill NaN and Inf values
        df = df.bfill().fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Added {len([c for c in df.columns if 'SMA_' in c or 'RSI_' in c or 'MACD' in c or 'ATR_' in c])} technical indicators")
        
        return df
    
    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback: add basic indicators without pandas_ta."""
        if 'close' in df.columns:
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['EMA_12'] = df['close'].ewm(span=12).mean()
            df['EMA_26'] = df['close'].ewm(span=26).mean()
            
            # Manual MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']
            
            # Manual RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Returns and volatility
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def fit_scaler(
        self,
        df: pd.DataFrame,
        columns: List[str],
        scaler_name: str = "default",
        scaler_type: str = "standard",
    ) -> None:
        """
        Fit a scaler on specified columns (on training data).
        
        Args:
            df: Training data
            columns: Columns to scale
            scaler_name: Identifier for this scaler
            scaler_type: 'standard' (mean=0, std=1) or 'minmax' (0-1)
        """
        available_cols = [c for c in columns if c in df.columns]
        
        if not available_cols:
            logger.warning(f"No columns found to fit scaler: {columns}")
            return
        
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        scaler.fit(df[available_cols].values)
        self.scalers[scaler_name] = scaler
        self.feature_stats[scaler_name] = {
            'columns': available_cols,
            'mean': dict(zip(available_cols, scaler.mean_)) if scaler_type == "standard" else None,
            'std': dict(zip(available_cols, scaler.scale_)) if scaler_type == "standard" else None,
        }
        
        logger.debug(f"Fitted scaler '{scaler_name}' on {len(available_cols)} columns")
    
    def transform(
        self,
        df: pd.DataFrame,
        scaler_name: str = "default",
        add_suffix: bool = True,
    ) -> pd.DataFrame:
        """
        Apply fitted scaler to data.
        
        Args:
            df: DataFrame to transform
            scaler_name: Which scaler to use
            add_suffix: Add '_scaled' suffix to transformed columns
        
        Returns:
            Dataframe with scaled columns
        """
        if scaler_name not in self.scalers:
            raise ValueError(
                f"Scaler '{scaler_name}' not fitted. Available: {list(self.scalers.keys())}"
            )
        
        df = df.copy()
        scaler = self.scalers[scaler_name]
        columns = self.feature_stats[scaler_name]['columns']
        
        available_cols = [c for c in columns if c in df.columns]
        transformed = scaler.transform(df[available_cols].values)
        
        for i, col in enumerate(available_cols):
            new_col = f"{col}_scaled" if add_suffix else col
            df[new_col] = transformed[:, i]
        
        logger.debug(f"Transformed {len(available_cols)} columns using '{scaler_name}'")
        return df
    
    def normalize_greeks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Greeks for PINN input using robust scaling.
        
        Args:
            df: DataFrame with Greek columns (delta, gamma, theta, vega, rho)
        
        Returns:
            DataFrame with normalized Greeks
        """
        df = df.copy()
        greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        
        for col in greek_cols:
            if col not in df.columns:
                continue
            
            # Delta is already in [-1, 1], so keep as is
            if col == 'delta':
                df[f'{col}_norm'] = df[col]
            else:
                # Robust scaling: (x - median) / IQR
                median = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                
                if iqr > 1e-6:
                    df[f'{col}_norm'] = (df[col] - median) / iqr
                else:
                    df[f'{col}_norm'] = 0
                
                # Clip to reasonable range to avoid outliers
                df[f'{col}_norm'] = df[f'{col}_norm'].clip(-5, 5)
        
        logger.debug(f"Normalized {len([c for c in greek_cols if f'{c}_norm' in df.columns])} Greeks")
        return df
    
    def aggregate_daily_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate options data to daily summaries (for PINN features).
        
        Creates mean/std of Greeks across all options per day.
        CRITICAL: Preserves spot_price_mean/min/max for technical indicators.
        
        Args:
            df: Options data with Greeks and time column
        
        Returns:
            Daily aggregated DataFrame with OHLCV-like structure
        """
        df = df.copy()
        
        # Identify date column
        date_cols = [c for c in ['time', 'date', 'data'] if c in df.columns]
        if not date_cols:
            logger.warning("No date column found for daily aggregation")
            return df
        
        date_col = date_cols[0]
        df['date'] = pd.to_datetime(df[date_col]).dt.date
        
        greek_cols = ['delta', 'gamma', 'theta', 'vega', 'rho']
        other_cols = ['volatility', 'days_to_maturity', 'premium', 'spot_price']
        
        agg_dict = {}
        
        for col in greek_cols:
            if col in df.columns:
                agg_dict[col] = ['mean', 'std']
        
        for col in other_cols:
            if col in df.columns:
                agg_dict[col] = ['mean', 'std', 'min', 'max']
        
        # Aggregate by date and asset
        groupby_cols = ['date'] + ([c for c in ['ativo', 'asset', 'symbol'] if c in df.columns][:1])
        daily = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        
        # Flatten multi-level columns
        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = ['_'.join(col).strip('_') for col in daily.columns.values]
        
        daily['date'] = pd.to_datetime(daily['date'])
        
        # ========================================================================
        # CRITICAL: Ensure we have close price for technical indicators
        # ========================================================================
        # After aggregation, we have: spot_price_mean, spot_price_std, spot_price_min, spot_price_max
        # add_technical_indicators() looks for 'close', 'open', 'high', 'low'
        # Solution: Create these columns as aliases
        if 'spot_price_mean' in daily.columns:
            daily['close'] = daily['spot_price_mean']  # Close = mean spot price
        
        if 'spot_price_max' in daily.columns:
            daily['high'] = daily['spot_price_max']    # High = max spot price
        
        if 'spot_price_min' in daily.columns:
            daily['low'] = daily['spot_price_min']     # Low = min spot price
        
        # Open: use close (no separate open for options data)
        if 'close' in daily.columns and 'open' not in daily.columns:
            daily['open'] = daily['close']
        
        # Volume: use acao_vol_fin_mean if available, else 1.0
        if 'acao_vol_fin_mean' in daily.columns:
            daily['volume'] = daily['acao_vol_fin_mean']
        elif 'volume' not in daily.columns:
            daily['volume'] = 1.0
        
        logger.info(f"Aggregated to {len(daily)} daily records")
        logger.info(f"  OHLCV columns created: open, high, low, close, volume")
        return daily
    
    @staticmethod
    def split_data(
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        date_col: str = 'time',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/test by date.
        
        Args:
            df: Full dataset with date column
            train_ratio: Proportion for training (0.8 = 80% train, 20% test)
            date_col: Name of date column
        
        Returns:
            Tuple of (train_df, test_df)
        """
        if date_col not in df.columns:
            # Find date column automatically
            date_cols = [c for c in ['time', 'date', 'data'] if c in df.columns]
            if not date_cols:
                raise ValueError("No date column found for splitting")
            date_col = date_cols[0]
        
        df = df.sort_values(by=date_col)
        split_idx = int(len(df) * train_ratio)
        
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        logger.info(f"Split data: {len(train)} train, {len(test)} test")
        return train, test

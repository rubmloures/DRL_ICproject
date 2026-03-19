"""
PINN Data Preprocessor
=====================
Processes raw CSV files containing OHLCV + Options data together.
Handles tax rates, dividend yields, and prepares data for PINN inference.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PINNDataPreprocessor:
    """
    Preprocessor for CSV files with OHLCV + Options data.
    
    Expected CSV structure:
    - symbol, time, spot_price, strike, premium, days_to_maturity, moneyness
    - delta, gamma, theta, vega, rho, volatility
    - acao_open, acao_high, acao_low, acao_close_ajustado, acao_vol_fin
    - ewma_vol, vol_parkinson
    """
    
    def __init__(self, data_stats_path: Optional[str] = None, verbose: bool = True):
        """
        Args:
            data_stats_path: Path to data_stats.json from PINN training
            verbose: Print logging information
        """
        self.verbose = verbose
        self.data_stats = None
        
        if data_stats_path:
            self._load_data_stats(data_stats_path)
    
    def _load_data_stats(self, data_stats_path: str):
        """Load normalization statistics from PINN training."""
        try:
            import json
            with open(data_stats_path, 'r') as f:
                self.data_stats = json.load(f)
            if self.verbose:
                logger.info(f"Loaded data stats from {data_stats_path}")
        except Exception as e:
            logger.warning(f"Could not load data stats: {e}")
    
    def load_raw_csv(
        self,
        csv_path: str,
        date_format: str = 'datetime',
        decimal: str = ',',
        sep: str = ';'
    ) -> pd.DataFrame:
        """
        Load raw CSV with OHLCV + Options data together.
        
        Args:
            csv_path: Path to CSV file
            date_format: Date format handling
            decimal: Decimal separator (',' for European)
            sep: Field separator (';' for European)
        
        Returns:
            Clean DataFrame ready for PINN processing
        """
        try:
            df = pd.read_csv(csv_path, decimal=decimal, sep=sep, dtype_backend='numpy_nullable')
            
            # Rename date column variants
            if 'time' in df.columns:
                df.rename(columns={'time': 'date'}, inplace=True)
            elif 'data' in df.columns:
                df.rename(columns={'data': 'date'}, inplace=True)
            
            # Parse dates
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            
            # Handle any date parsing errors
            if df['date'].isna().any():
                logger.warning(f"Found {df['date'].isna().sum()} unparseable dates in {csv_path}")
                df = df.dropna(subset=['date'])
            
            # Rename acao columns (stock OHLCV)
            rename_map = {
                'acao_open': 'open',
                'acao_high': 'high',
                'acao_low': 'low',
                'acao_close_ajustado': 'close',
                'acao_vol_fin': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)
            
            if self.verbose:
                logger.info(f"Loaded {len(df)} rows from {csv_path}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            raise
    
    def merge_market_rates(
        self,
        df: pd.DataFrame,
        taxa_selic_path: str,
        dividend_yields_path: str
    ) -> pd.DataFrame:
        """
        Merge risk-free rate (taxa Selic) and dividend yields.
        Set r (risk-free rate) and q (dividend yield) for PINN input [S, K, T, r, q].
        
        Args:
            df: DataFrame with options data
            taxa_selic_path: Path to taxa_selic.csv
            dividend_yields_path: Path to dividend_yields.csv
        
        Returns:
            DataFrame with r_rate and dividend_yield columns added
        """
        
        # Load taxa Selic
        try:
            selic_df = pd.read_csv(taxa_selic_path, decimal=',', sep=',')
            selic_df.columns = ['data', 'valor']
            selic_df['data'] = pd.to_datetime(selic_df['data'], format='%d/%m/%Y', errors='coerce')
            selic_df['valor'] = selic_df['valor'].str.replace(',', '.').astype(float) / 100.0
            selic_df = selic_df.dropna()
            selic_df.rename(columns={'data': 'date', 'valor': 'r_rate'}, inplace=True)
            selic_df = selic_df.sort_values('date')
            
            if self.verbose:
                logger.info(f"Loaded {len(selic_df)} taxa_selic records")
        except Exception as e:
            logger.error(f"Error loading taxa_selic: {e}")
            raise
        
        # Load dividend yields
        try:
            div_df = pd.read_csv(dividend_yields_path)
            div_df['data_only'] = pd.to_datetime(div_df['data_only'], utc=True)
            div_df['date'] = div_df['data_only'].dt.date
            div_df['date'] = pd.to_datetime(div_df['date'])
            div_df.rename(columns={'ativo': 'symbol'}, inplace=True)
            
            # Fill missing dividend yields with 0
            div_df['Dividend_Yield'] = pd.to_numeric(div_df['Dividend_Yield'], errors='coerce').fillna(0.0)
            
            if self.verbose:
                logger.info(f"Loaded {len(div_df)} dividend yield records")
        except Exception as e:
            logger.error(f"Error loading dividend_yields: {e}")
            raise
        
        # Merge taxa Selic (one rate per date for all assets)
        df = df.merge(selic_df[['date', 'r_rate']], on='date', how='left')
        
        # Merge dividend yields (one rate per asset/date)
        df = df.merge(
            div_df[['date', 'symbol', 'Dividend_Yield']],
            on=['date', 'symbol'],
            how='left'
        )
        
        # Forward fill missing rates (handle weekends/holidays)
        df['r_rate'] = df.groupby(level=0, group_keys=False)['r_rate'].fillna(method='ffill')
        df['Dividend_Yield'] = df.groupby(level=0, group_keys=False)['Dividend_Yield'].fillna(0.0)
        
        # Backward fill for dates before first rate
        df['r_rate'] = df['r_rate'].bfill()
        
        if self.verbose:
            logger.info(f"Merged taxa_selic and dividend_yields")
        
        return df
    
    def calculate_lstm_features(
        self,
        df: pd.DataFrame,
        window_size: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple]]:
        """
        Calculate time-series features for LSTM input.
        Generate sliding windows for PINN inference.
        
        Args:
            df: DataFrame with OHLCV data
            window_size: Size of sliding window (days)
        
        Returns:
            x_seq: [num_windows, window_size, 6] - LSTM sequences
            x_phy: [num_windows, 5] - Physical inputs [S, K, T, r, q]
            metadata: List of (date, symbol, strike) tuples for tracking
        """
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # Calculate features (Aligned with src/pinn/data_loader.py)
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0.0)
        df['rolling_vol_20'] = df['log_ret'].rolling(window=20).std().fillna(0.0)
        df['ewma_vol'] = df['log_ret'].ewm(span=20, adjust=False).std().fillna(0.0)
        
        # Log Volume Financeiro (ACAO_VOL_FIN)
        # Handle different potential column names for volume
        vol_col = next((c for c in ['volume', 'acao_vol_fin'] if c in df.columns), None)
        if vol_col:
            df['log_vol_fin'] = np.log(pd.to_numeric(df[vol_col], errors='coerce') + 1.0).fillna(0.0)
        else:
            df['log_vol_fin'] = 0.0
            
        # Volatilidade de Parkinson
        if 'high' in df.columns and 'low' in df.columns:
            h, l = pd.to_numeric(df['high'], errors='coerce'), pd.to_numeric(df['low'], errors='coerce')
            df['vol_parkinson'] = np.sqrt((1/(4*np.log(2))) * (np.log(h/(l+1e-8))**2)).fillna(0.0)
        else:
            df['vol_parkinson'] = 0.0

        # Log-retornos do IBOV (Contexto de mercado)
        # Try to load IBOV context if available in the same directory or via PATHS
        df['log_ret_ibov'] = 0.0
        try:
            from config.config import DATA_RAW
            ibov_path = DATA_RAW / "BOVA11.csv"
            if ibov_path.exists():
                    # Simple load to get returns
                    ibov_df = pd.read_csv(ibov_path, sep=';', decimal=',', skiprows=1)
                    ibov_date_col = next((c for c in ['time', 'date', 'data'] if c in ibov_df.columns), None)
                    ibov_close_col = next((c for c in ['acao_close_ajustado', 'close'] if c in ibov_df.columns), None)
                    if ibov_date_col and ibov_close_col:
                        ibov_df[ibov_date_col] = pd.to_datetime(ibov_df[ibov_date_col], errors='coerce')
                        
                        # --- FIX: Aggregate IBOV to daily basis to prevent merge explosion ---
                        ibov_df['date_norm'] = ibov_df[ibov_date_col].dt.normalize()
                        # Take the last close of each day
                        ibov_daily = ibov_df.sort_values(ibov_date_col).groupby('date_norm').agg({
                            ibov_close_col: 'last'
                        }).reset_index()
                        
                        # Calculate daily log returns for IBOV
                        ibov_daily['log_ret_ibov_real'] = np.log(
                            ibov_daily[ibov_close_col] / ibov_daily[ibov_close_col].shift(1)
                        ).fillna(0.0)
                        
                        # Merge on normalized date
                        df['temp_date'] = df['date'].dt.normalize()
                        df = pd.merge(df, ibov_daily[['date_norm', 'log_ret_ibov_real']], 
                                     left_on='temp_date', right_on='date_norm', how='left')
                        df['log_ret_ibov'] = df['log_ret_ibov_real'].fillna(0.0)
                        # Clean up temp columns
                        df = df.drop(columns=['temp_date', 'date_norm', 'log_ret_ibov_real'])
        except Exception as e:
            logger.debug(f"Could not load IBOV context for PINN: {e}")
        
        x_seq_list = []
        x_phy_list = []
        metadata_list = []
        
        # Generate sliding windows
        for i in range(len(df) - window_size + 1):
            window = df.iloc[i:i+window_size]
            last_row = window.iloc[-1]
            
            # Build x_seq [window_size, 6]
            seq = np.column_stack([
                window['log_ret'].values,
                window['rolling_vol_20'].values,
                window['ewma_vol'].values,
                window['vol_parkinson'].values,
                window['log_vol_fin'].values,
                window['log_ret_ibov'].values
            ]).astype(np.float32)
            
            # Build x_phy [5] = [S, K, T, r, q]
            phy = np.array([
                float(last_row['spot_price']),
                float(last_row['strike']),
                float(last_row['days_to_maturity'] / 365.0),
                float(last_row.get('r_rate', 0.105)),  # Default: 10.5%
                float(last_row.get('Dividend_Yield', 0.0))
            ], dtype=np.float32)
            
            x_seq_list.append(seq)
            x_phy_list.append(phy)
            metadata_list.append((
                last_row['date'],
                last_row.get('symbol', 'UNKNOWN'),
                last_row['strike']
            ))
        
        return (
            np.array(x_seq_list),  # [num_windows, 30, 6]
            np.array(x_phy_list),  # [num_windows, 5]
            metadata_list
        )
    
    def filter_moneyness(
        self,
        df: pd.DataFrame,
        min_moneyness: float = 0.7,
        max_moneyness: float = 1.3,
        option_type: str = 'CALL'
    ) -> pd.DataFrame:
        """
        Filter options by moneyness range and type.
        
        Args:
            df: DataFrame with options data
            min_moneyness: Min moneyness to keep
            max_moneyness: Max moneyness to keep
            option_type: 'CALL' or 'PUT'
        
        Returns:
            Filtered DataFrame
        """
        if 'moneyness' not in df.columns:
            df['moneyness'] = df['spot_price'] / df['strike']
        
        filtered = df[
            (df['moneyness'] >= min_moneyness) &
            (df['moneyness'] <= max_moneyness)
        ]
        
        if 'type' in filtered.columns:
            filtered = filtered[filtered['type'] == option_type]
        
        if self.verbose:
            logger.info(
                f"Filtered {len(df)} -> {len(filtered)} options "
                f"(moneyness [{min_moneyness}, {max_moneyness}], type={option_type})"
            )
        
        return filtered
    
    def normalize_features(
        self,
        x_seq: np.ndarray,
        x_phy: np.ndarray,
        method: str = 'z_score'
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Normalize inputs using statistics from PINN training.
        
        Args:
            x_seq: [batch, window_size, features]
            x_phy: [batch, 5]
            method: 'z_score' or 'min_max'
        
        Returns:
            x_seq_norm, x_phy_norm, stats_dict
        """
        
        if method == 'z_score':
            # Compute statistics
            x_seq_mean = x_seq.mean(axis=(0, 1), keepdims=True)
            x_seq_std = x_seq.std(axis=(0, 1), keepdims=True)
            x_phy_mean = x_phy.mean(axis=0, keepdims=True)
            x_phy_std = x_phy.std(axis=0, keepdims=True)
            
            # Normalize
            x_seq_norm = (x_seq - x_seq_mean) / (x_seq_std + 1e-8)
            x_phy_norm = (x_phy - x_phy_mean) / (x_phy_std + 1e-8)
            
            stats = {
                'x_seq_mean': x_seq_mean,
                'x_seq_std': x_seq_std,
                'x_phy_mean': x_phy_mean,
                'x_phy_std': x_phy_std,
            }
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        if self.verbose:
            logger.info(f"Normalized features using {method}")
        
        return x_seq_norm.astype(np.float32), x_phy_norm.astype(np.float32), stats

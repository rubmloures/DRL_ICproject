"""
Refactored Data Loader
======================
Follows FinRL pattern: simple data loading with proper error handling.
Handles Brazilian CSV format (semicolon-separated, comma decimals).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from functools import lru_cache
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LRUCacheDict:
    """LRU Cache with maximum size limit to prevent memory leaks."""
    
    def __init__(self, size: int = 50):
        """
        Initialize LRU cache.
        
        Args:
            size: Maximum number of items to cache (default: 50)
        """
        self.cache = OrderedDict()
        self.size = size
        self.hits = 0
        self.misses = 0
    
    def __contains__(self, key):
        """Check if key exists in cache."""
        return key in self.cache
    
    def __getitem__(self, key):
        """Get item from cache (moves to end for LRU)."""
        if key not in self.cache:
            raise KeyError(f"Key {key} not in cache")
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]
    
    def __setitem__(self, key, value):
        """Set item in cache, evicting oldest if necessary."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.size:
            removed_key = self.cache.popitem(last=False)[0]
            logger.debug(f"Evicted {removed_key} from cache (size limit {self.size} exceeded)")
    
    def pop(self, key, default=None):
        """Remove key from cache."""
        return self.cache.pop(key, default)
    
    def clear(self):
        """Clear all items from cache."""
        self.cache.clear()
        logger.debug("Cache cleared")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            'size': len(self.cache),
            'max_size': self.size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%"
        }
            
class DataLoader:
    """
    Loads raw market data from CSV files.
    
    Responsibilities:
    - Parse Brazilian CSV format (semicolon separator, comma decimals)
    - Handle missing data gracefully
    - Convert date formats
    - Encode categorical variables
    
    Does NOT handle:
    - Technical indicator calculation (handled by DataProcessor)
    - Normalization (handled by DataProcessor)
    - Train/test splitting (handled by environment)
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize DataLoader with LRU cache.
        
        Args:
            data_path: Root directory containing CSV files
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "raw"
        
        self.data_path = Path(data_path)
        self._cache = LRUCacheDict(size=50)  # LRU cache with 50 item limit
        
        if not self.data_path.exists():
            logger.warning(f"Data path does not exist: {self.data_path}")
    
    def _parse_brazilian_number(self, value) -> float:
        """Convert Brazilian number format (comma decimal) to float."""
        if pd.isna(value) or value == "":
            return np.nan
        
        if isinstance(value, (int, float)):
            return float(value)
        
        # Convert string: "1.234,56" -> 1234.56
        value_str = str(value).strip()
        value_str = value_str.replace('.', '')  # Remove thousands separator
        value_str = value_str.replace(',', '.')  # Convert decimal separator
        
        try:
            return float(value_str)
        except ValueError:
            return np.nan
    
    def load_asset(
        self,
        asset: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Load data for a specific asset from CSV.
        
        Args:
            asset: Asset ticker (e.g., 'PETR4', 'VALE3')
            start_date: Filter data from this date (format: 'YYYY-MM-DD')
            end_date: Filter data until this date (format: 'YYYY-MM-DD')
            use_cache: Use cached data if available
        
        Returns:
            DataFrame with parsed data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV cannot be parsed
        """
        cache_key = f"{asset}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached data for {asset}")
            return self._cache[cache_key].copy()
        
        # Find CSV file
        csv_path = self.data_path / f"{asset}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {csv_path}\n"
                f"Available files in {self.data_path}: "
                f"{list(self.data_path.glob('*.csv'))}"
            )
        
        logger.info(f"Loading {asset} from {csv_path}")
        
        try:
            # Read CSV with Brazilian format
            # Skip first line if it contains "sep=" (format indicator)
            df = pd.read_csv(
                csv_path,
                sep=";",
                skipinitialspace=True,
                low_memory=False,
                skiprows=1,  # Skip "sep=;" header line
            )
            
            logger.debug(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Parse numeric columns with Brazilian decimal format
            numeric_cols = [
                'spot_price', 'strike', 'premium', 'days_to_maturity',
                'delta', 'gamma', 'theta', 'vega', 'rho', 'volatility',
                'poe', 'bs',
                'acao_open', 'acao_high', 'acao_low',
                'acao_close_ajustado', 'acao_vol_fin',
                'ewma_vol', 'vol_parkinson'
            ]
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].apply(self._parse_brazilian_number)
            
            # Parse datetime
            date_cols = ['time', 'data', 'date']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    break
            
            # Encode categorical columns
            if 'moneyness' in df.columns:
                moneyness_map = {'ITM': 0, 'ATM': 1, 'OTM': 2}
                df['moneyness_encoded'] = df['moneyness'].map(moneyness_map).fillna(-1).astype(int)
            
            if 'type' in df.columns:
                type_map = {'CALL': 0, 'PUT': 1}
                df['type_encoded'] = df['type'].map(type_map).fillna(-1).astype(int)
            
            # Add asset identifier
            df['asset'] = asset
            
            # Filter by date range (optional)
            if start_date or end_date:
                date_col = next((c for c in ['time', 'date', 'data'] if c in df.columns), None)
                if date_col:
                    if start_date:
                        start_dt = pd.to_datetime(start_date)
                        df = df[df[date_col] >= start_dt]
                    if end_date:
                        end_dt = pd.to_datetime(end_date)
                        df = df[df[date_col] <= end_dt]
            
            # Cache result
            self._cache[cache_key] = df.copy()
            
            logger.info(f"Successfully loaded {asset}: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading {asset}: {str(e)}")
            raise ValueError(f"Cannot parse {csv_path}: {str(e)}")
    
    def load_multiple_assets(
        self,
        assets: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data for multiple assets and merge.
        
        Args:
            assets: List of asset tickers
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            Merged DataFrame with data for all assets
        """
        dfs = []
        
        for asset in assets:
            try:
                df = self.load_asset(asset, start_date, end_date)
                dfs.append(df)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {asset}: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"No data loaded for assets: {assets}")
        
        # Merge data (keep all dates from all assets)
        merged = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Merged data: {len(merged)} total records from {len(dfs)} assets")
        return merged
    
    def load_risk_free_rate(
        self,
        filepath: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Load risk-free rate (SELIC) data.
        
        Args:
            filepath: Path to SELIC CSV (default: data/raw/taxa_selic.csv)
        
        Returns:
            DataFrame with columns [date, rate]
        """
        if filepath is None:
            filepath = self.data_path / "taxa_selic.csv"
        
        if not filepath.exists():
            logger.warning(f"Risk-free rate file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath, sep=";")
            
            # Find date and value columns
            date_col = next((c for c in ['data', 'date', 'time'] if c in df.columns), None)
            value_col = next((c for c in ['valor', 'value', df.columns[1]] if c in df.columns), None)
            
            if not date_col or not value_col:
                raise ValueError(f"Cannot find date and value columns in {filepath}")
            
            df['date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            df['rate'] = df[value_col].apply(self._parse_brazilian_number)
            
            # Normalize to daily rate (typically SELIC is yearly)
            if df['rate'].mean() > 0.1:  # If mean > 10%, likely yearly
                df['rate'] = df['rate'] / 100.0 / 252.0
            
            result = df[['date', 'rate']].dropna().drop_duplicates('date')
            logger.info(f"Loaded risk-free rate: {len(result)} dates")
            return result
            
        except Exception as e:
            logger.error(f"Error loading risk-free rate: {e}")
            return None
    
    def load_dividend_yields(
        self,
        filepath: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Load dividend yield data.
        
        Args:
            filepath: Path to dividend CSV (default: data/raw/dividend_yields.csv)
        
        Returns:
            DataFrame with columns [date, asset, yield]
        """
        if filepath is None:
            filepath = self.data_path / "dividend_yields.csv"
        
        if not filepath.exists():
            logger.warning(f"Dividend file not found: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath, sep=",")  # Usually comma-separated
            
            # Standardize columns
            if 'data_only' in df.columns:
                df['date'] = pd.to_datetime(df['data_only'], errors='coerce')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            else:
                df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
            
            # Get yield column
            yield_col = next((c for c in ['Dividend_Yield', 'yield', 'div_yield'] if c in df.columns), None)
            if not yield_col:
                yield_col = df.columns[-1]
            
            df['yield'] = df[yield_col].apply(self._parse_brazilian_number)
            df['asset'] = df.get('ativo', df.get('asset', 'GENERIC'))
            
            result = df[['date', 'asset', 'yield']].dropna()
            logger.info(f"Loaded dividend yields: {len(result)} records")
            return result
            
        except Exception as e:
            logger.error(f"Error loading dividend yields: {e}")
            return None
    
    def clear_cache(self, asset: Optional[str] = None):
        """Clear cached data."""
        if asset:
            self._cache.pop(asset, None)
            logger.debug(f"Cleared cache for {asset}")
        else:
            self._cache.clear()
            logger.debug("Cleared all cache")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache hit rate, size, etc.
        """
        return self._cache.get_stats()

"""
Temporal Data Splitter for Time Series Cross-Validation
========================================================
Implements temporal splitting for financial data to avoid data leakage:
- Train (≤2023)
- Test (2024)  
- Validation (2025+)
- K-fold cross-validation within train windows
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class TemporalDataSplitter:
    """
    Splits financial time series data respecting temporal order.
    Prevents data leakage by ensuring train dates < test dates < val dates.
    """
    
    def __init__(self, date_column: str = 'date'):
        """
        Args:
            date_column: Name of the date column
        """
        self.date_column = date_column
    
    def split_temporal(
        self,
        df: pd.DataFrame,
        train_until: str = '2023-12-31',
        test_start: str = '2024-01-01',
        test_until: str = '2024-12-31',
        val_start: str = '2025-01-01'
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/test/val based on temporal boundaries.
        
        Args:
            df: Input DataFrame with date column
            train_until: Last date for training data (YYYY-MM-DD)
            test_start: First date for test data (YYYY-MM-DD)
            test_until: Last date for test data (YYYY-MM-DD)
            val_start: First date for validation data (YYYY-MM-DD)
        
        Returns:
            Dict with 'train', 'test', 'val' DataFrames
        
        Example:
            >>> splitter = TemporalDataSplitter()
            >>> splits = splitter.split_temporal(df)
            >>> train_df = splits['train']  # Until 2023-12-31
            >>> test_df = splits['test']     # 2024 only
            >>> val_df = splits['val']       # 2025+
        """
        # Convert to datetime
        df = df.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        train_until_dt = pd.to_datetime(train_until)
        test_start_dt = pd.to_datetime(test_start)
        test_until_dt = pd.to_datetime(test_until)
        val_start_dt = pd.to_datetime(val_start)
        
        # Split
        train_df = df[df[self.date_column] <= train_until_dt]
        test_df = df[
            (df[self.date_column] >= test_start_dt) & 
            (df[self.date_column] <= test_until_dt)
        ]
        val_df = df[df[self.date_column] >= val_start_dt]
        
        return {
            'train': train_df.reset_index(drop=True),
            'test': test_df.reset_index(drop=True),
            'val': val_df.reset_index(drop=True)
        }
    
    def kfold_split_train_window(
        self,
        train_df: pd.DataFrame,
        k: int = 3,
        shuffle: bool = False
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        K-fold cross-validation within training data.
        Maintains temporal order (no shuffling) for time series.
        
        Args:
            train_df: Training DataFrame (temporal order preserved)
            k: Number of folds
            shuffle: If False (default), maintains temporal order for time series
        
        Returns:
            List of (train_fold, val_fold) tuples
            
        Example:
            >>> splitter = TemporalDataSplitter()
            >>> folds = splitter.kfold_split_train_window(train_df, k=3)
            >>> for fold_idx, (train_fold, val_fold) in enumerate(folds):
            ...     print(f"Fold {fold_idx}: train {len(train_fold)}, val {len(val_fold)}")
        """
        n = len(train_df)
        fold_size = n // k
        
        folds = []
        
        for i in range(k):
            val_start_idx = i * fold_size
            val_end_idx = (i + 1) * fold_size if i < k - 1 else n
            
            val_fold = train_df.iloc[val_start_idx:val_end_idx].copy()
            
            # Train fold includes all data before and after val fold
            train_fold_parts = [
                train_df.iloc[:val_start_idx],
                train_df.iloc[val_end_idx:]
            ]
            train_fold = pd.concat(train_fold_parts, ignore_index=True)
            
            folds.append((train_fold, val_fold))
        
        return folds
    
    def stratified_kfold_by_asset(
        self,
        df: pd.DataFrame,
        asset_column: str = 'symbol',
        k: int = 3
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        K-fold split ensuring each fold has all assets.
        Useful for multi-asset backtesting.
        
        Args:
            df: DataFrame with asset column
            asset_column: Name of asset/symbol column
            k: Number of folds
        
        Returns:
            List of (train_fold, val_fold) tuples where each fold contains all assets
        """
        folds = []
        
        for fold_idx in range(k):
            val_fold_parts = []
            train_fold_parts = []
            
            # For each asset, split its data into k parts
            for asset in df[asset_column].unique():
                asset_df = df[df[asset_column] == asset].copy()
                
                n = len(asset_df)
                fold_size = n // k
                
                val_start_idx = fold_idx * fold_size
                val_end_idx = (fold_idx + 1) * fold_size if fold_idx < k - 1 else n
                
                val_fold_parts.append(asset_df.iloc[val_start_idx:val_end_idx])
                
                train_fold_parts.append(asset_df.iloc[:val_start_idx])
                train_fold_parts.append(asset_df.iloc[val_end_idx:])
            
            val_fold = pd.concat(val_fold_parts, ignore_index=True)
            train_fold = pd.concat(train_fold_parts, ignore_index=True)
            
            folds.append((train_fold, val_fold))
        
        return folds
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        train_size_days: int = 252,  # 1 year of trading days
        test_size_days: int = 63,    # 1 quarter of trading days
        step_size_days: Optional[int] = None
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-forward validation (expanding window or fixed window with step).
        
        Args:
            df: Input DataFrame sorted by date
            train_size_days: Number of trading days for training
            test_size_days: Number of trading days for testing
            step_size_days: Step size between walk-forward steps (if None, uses test_size_days)
        
        Returns:
            List of (train, test) DataFrames
        """
        if step_size_days is None:
            step_size_days = test_size_days
        
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        splits = []
        idx = 0
        
        while idx + train_size_days + test_size_days <= len(df):
            train_end_idx = idx + train_size_days
            test_end_idx = idx + train_size_days + test_size_days
            
            train = df.iloc[idx:train_end_idx]
            test = df.iloc[train_end_idx:test_end_idx]
            
            splits.append((train, test))
            
            idx += step_size_days
        
        return splits
    
    def report_split_info(
        self,
        splits: Dict[str, pd.DataFrame],
        date_column: str = 'date'
    ) -> str:
        """
        Generate informative report about data splits.
        
        Args:
            splits: Dict with 'train', 'test', 'val' DataFrames
            date_column: Name of date column
        
        Returns:
            Formatted string report
        """
        report = "\n" + "=" * 70 + "\n"
        report += "TEMPORAL DATA SPLIT REPORT\n"
        report += "=" * 70 + "\n\n"
        
        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                report += f"{split_name.upper()}: [EMPTY]\n"
            else:
                min_date = split_df[date_column].min()
                max_date = split_df[date_column].max()
                report += f"{split_name.upper()}:\n"
                report += f"  Rows: {len(split_df)}\n"
                report += f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}\n"
                report += f"  Days: {(max_date - min_date).days + 1}\n\n"
        
        report += "=" * 70 + "\n"
        return report

"""
Rolling Window Strategy for Backtesting
========================================

Implements sliding window cross-validation for time-series data.
Default: 14 weeks training, 4 weeks testing, 2 weeks overlap.

Example:
    >>> df = load_data()
    >>> strategy = RollingWindowStrategy(df, train_weeks=14, test_weeks=4)
    >>> for train_df, test_df, window_idx, dates in strategy.generate_rolling_windows():
    ...     train_agent(train_df)
    ...     evaluate_agent(test_df)
"""

import logging
from typing import Tuple, Dict, List, Generator
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RollingWindowStrategy:
    """
    Implements rolling window (walk-forward) strategy for backtesting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'data' column for dates
    train_weeks : int
        Number of weeks for training (default: 14)
    test_weeks : int
        Number of weeks for testing (default: 4)
    overlap_weeks : int
        Overlap between windows in weeks (default: 2)
    
    Attributes
    ----------
    train_days : int
        ~5 trading days per week
    test_days : int
        ~5 trading days per week
    shift_days : int
        Days to shift window (train_weeks - overlap_weeks) * 5
    
    Example
    -------
    >>> strategy = RollingWindowStrategy(df, train_weeks=14, test_weeks=4)
    >>> windows = list(strategy.generate_rolling_windows())
    >>> print(f"Generated {len(windows)} windows")
    >>> metrics = strategy.get_metrics_across_windows(results)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        train_weeks: int = 14,
        test_weeks: int = 4,
        overlap_weeks: int = 0,
        with_validation_fold: bool = False,
        k_fold: int = 3
    ):
        """Initialize rolling window strategy."""
        self.df = df.copy()
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.overlap_weeks = overlap_weeks
        self.with_validation_fold = with_validation_fold
        self.k_fold = k_fold
        
        # Convert weeks to trading days (~5 days per week)
        self.train_days = train_weeks * 5
        self.test_days = test_weeks * 5
        self.shift_days = (train_weeks - overlap_weeks) * 5
        
        # Ensure date column exists
        if 'data' not in self.df.columns:
            logger.warning("'data' column not found. Using index as dates.")
            self.df['data'] = pd.to_datetime(self.df.index)
        
        logger.info(f"RollingWindowStrategy initialized:")
        logger.info(f"  Training window: {self.train_weeks} weeks ({self.train_days} days)")
        logger.info(f"  Testing window: {self.test_weeks} weeks ({self.test_days} days)")
        logger.info(f"  Overlap: {self.overlap_weeks} weeks ({self.overlap_weeks * 5} days)")
        logger.info(f"  Window shift: {self.shift_days} days")
        logger.info(f"  K-fold validation: {with_validation_fold} (k={k_fold if with_validation_fold else 'N/A'})")
        logger.info(f"  Total data points: {len(self.df)}")
    
    def generate_rolling_windows(self) -> Generator[Tuple, None, None]:
        """
        Generate rolling train/test window pairs.
        
        Yields
        ------
        train_df : pd.DataFrame
            Training data for this window
        test_df : pd.DataFrame
            Testing data for this window
        window_idx : int
            Window index (0, 1, 2, ...)
        date_range : dict
            Dictionary with 'train_start', 'train_end', 'test_start', 'test_end'
        
        Example
        -------
        >>> for train_df, test_df, idx, dates in strategy.generate_rolling_windows():
        ...     print(f"Window {idx}: {dates['train_start']} to {dates['test_end']}")
        """
        # Calculate number of possible windows
        total_length = len(self.df)
        min_required = self.train_days + self.test_days
        
        if total_length < min_required:
            logger.warning(f"Data too short ({total_length} days) for at least one window "
                          f"({min_required} days required)")
            return
        
        n_windows = ((total_length - min_required) // self.shift_days) + 1
        logger.info(f"Generating {n_windows} rolling windows...")
        
        for window_idx in range(n_windows):
            # Calculate indices for this window
            start_idx = window_idx * self.shift_days
            train_end_idx = start_idx + self.train_days
            test_end_idx = train_end_idx + self.test_days
            
            # Check bounds
            if test_end_idx > len(self.df):
                logger.debug(f"Window {window_idx} exceeds data length, stopping")
                break
            
            # Extract data
            train_df = self.df.iloc[start_idx:train_end_idx].copy()
            test_df = self.df.iloc[train_end_idx:test_end_idx].copy()
            
            # Get date range
            if isinstance(self.df['data'].iloc[0], pd.Timestamp):
                train_start = train_df['data'].iloc[0]
                train_end = train_df['data'].iloc[-1]
                test_start = test_df['data'].iloc[0]
                test_end = test_df['data'].iloc[-1]
            else:
                train_start = train_df['data'].iloc[0]
                train_end = train_df['data'].iloc[-1]
                test_start = test_df['data'].iloc[0]
                test_end = test_df['data'].iloc[-1]
            
            date_range = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            }
            
            logger.debug(f"Window {window_idx}: Train[{start_idx}:{train_end_idx}] "
                        f"Test[{train_end_idx}:{test_end_idx}]")
            
            yield train_df, test_df, window_idx, date_range
    
    def generate_rolling_windows_with_kfold(self) -> Generator[Tuple, None, None]:
        """
        Generate rolling windows with K-fold validation inside train window.
        
        Yields
        ------
        train_fold_df : pd.DataFrame
            Training data for this fold
        val_fold_df : pd.DataFrame
            Validation data for this fold
        test_df : pd.DataFrame
            Test data for this window
        window_idx : int
            Rolling window index
        fold_idx : int
            K-fold index
        date_range : dict
            Dictionary with date ranges
        
        Example
        -------
        >>> for train_fold, val_fold, test_df, win_idx, fold_idx, dates in \
        ...         strategy.generate_rolling_windows_with_kfold():
        ...     print(f"Window {win_idx}, Fold {fold_idx}")
        """
        if not self.with_validation_fold:
            logger.warning("K-fold validation disabled. Use generate_rolling_windows() instead.")
            for train_df, test_df, window_idx, date_range in self.generate_rolling_windows():
                yield train_df, None, test_df, window_idx, 0, date_range
            return
        
        for train_df, test_df, window_idx, date_range in self.generate_rolling_windows():
            # Generate K-fold splits within training data
            kfold_splits = self._kfold_split_train(train_df, k=self.k_fold)
            
            for fold_idx, (train_fold, val_fold) in enumerate(kfold_splits):
                # Dates for this fold (same as overall window)
                fold_date_range = {
                    'window_idx': window_idx,
                    'fold_idx': fold_idx,
                    'train_start': train_fold['data'].iloc[0] if len(train_fold) > 0 else None,
                    'train_end': train_fold['data'].iloc[-1] if len(train_fold) > 0 else None,
                    'val_start': val_fold['data'].iloc[0] if len(val_fold) > 0 else None,
                    'val_end': val_fold['data'].iloc[-1] if len(val_fold) > 0 else None,
                    'test_start': test_df['data'].iloc[0] if len(test_df) > 0 else None,
                    'test_end': test_df['data'].iloc[-1] if len(test_df) > 0 else None,
                }
                
                logger.debug(f"Window {window_idx}, Fold {fold_idx}: "
                            f"Train {train_fold.shape[0]}, Val {val_fold.shape[0]}, Test {test_df.shape[0]}")
                
                yield train_fold, val_fold, test_df, window_idx, fold_idx, fold_date_range
    
    @staticmethod
    def _kfold_split_train(
        train_df: pd.DataFrame,
        k: int = 3,
        shuffle: bool = False
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split training data into k folds for cross-validation.
        Maintains temporal order for time series (no shuffling).
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data to split
        k : int
            Number of folds
        shuffle : bool
            If False (default), maintains temporal order
        
        Returns
        -------
        list of (train_fold, val_fold) tuples
        """
        n = len(train_df)
        fold_size = n // k
        
        folds = []
        
        for i in range(k):
            val_start_idx = i * fold_size
            val_end_idx = (i + 1) * fold_size if i < k - 1 else n
            
            val_fold = train_df.iloc[val_start_idx:val_end_idx].copy()
            
            # Train fold = all data except validation fold
            train_fold_parts = [
                train_df.iloc[:val_start_idx],
                train_df.iloc[val_end_idx:]
            ]
            train_fold = pd.concat(train_fold_parts, ignore_index=True).reset_index(drop=True)
            
            folds.append((train_fold, val_fold))
        
        return folds
    
    @staticmethod
    def get_metrics_across_windows(results: List[Dict]) -> Dict:
        """
        Aggregate metrics across all windows.
        
        Parameters
        ----------
        results : list of dict
            Results from each window with keys: 'window_idx', 'sharpe', 'sortino',
            'returns', 'max_drawdown', etc.
        
        Returns
        -------
        dict
            Aggregated metrics across all windows
        
        Example
        -------
        >>> results = [
        ...     {'window_idx': 0, 'sharpe': 0.5, 'returns': 0.01},
        ...     {'window_idx': 1, 'sharpe': 0.7, 'returns': 0.02},
        ... ]
        >>> metrics = RollingWindowStrategy.get_metrics_across_windows(results)
        >>> print(f"Avg Sharpe: {metrics['avg_sharpe']:.4f}")
        """
        if not results:
            logger.warning("No results to aggregate")
            return {}
        
        logger.info(f"Aggregating metrics from {len(results)} windows...")
        
        # Extract metrics by window
        metrics_by_window = {metric: [] for metric in results[0].keys() if metric != 'window_idx'}
        
        for result in results:
            for metric, value in result.items():
                if metric != 'window_idx' and isinstance(value, (int, float)):
                    metrics_by_window[metric].append(value)
        
        # Compute statistics for each metric
        aggregated = {
            'total_windows': len(results),
            'window_indices': [r.get('window_idx', i) for i, r in enumerate(results)],
        }
        
        for metric, values in metrics_by_window.items():
            if values:
                aggregated[f'avg_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)
                aggregated[f'median_{metric}'] = np.median(values)
        
        logger.info("Aggregation complete:")
        for key, value in aggregated.items():
            if not key.startswith('window_indices'):
                logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        return aggregated
    
    @staticmethod
    def print_window_summary(date_range: Dict) -> None:
        """
        Print a summary of a rolling window.
        
        Parameters
        ----------
        date_range : dict
            Dictionary with 'train_start', 'train_end', 'test_start', 'test_end'
        """
        print("\n" + "="*70)
        print("Rolling Window Summary")
        print("="*70)
        print(f"Training:  {date_range['train_start']} to {date_range['train_end']}")
        print(f"Testing:   {date_range['test_start']} to {date_range['test_end']}")
        print("="*70)


class ExpandingWindowStrategy:
    """
    Alternative: Expanding window strategy (no training window size limit).
    Data grows over time: [start, time_t], [start, time_t+1], etc.
    
    Useful for online learning scenarios.
    """
    
    def __init__(self, df: pd.DataFrame, test_weeks: int = 4):
        """Initialize expanding window strategy."""
        self.df = df.copy()
        self.test_weeks = test_weeks
        self.test_days = test_weeks * 5
        
        logger.info(f"ExpandingWindowStrategy initialized:")
        logger.info(f"  Testing window: {test_weeks} weeks ({self.test_days} days)")
    
    def generate_expanding_windows(self) -> Generator[Tuple, None, None]:
        """
        Generate expanding train/test window pairs.
        Training data grows, test data is fixed.
        """
        min_train_days = 20  # Minimum training data
        
        for end_idx in range(min_train_days, len(self.df) - self.test_days):
            train_df = self.df.iloc[:end_idx].copy()
            test_df = self.df.iloc[end_idx:end_idx + self.test_days].copy()
            
            date_range = {
                'train_start': train_df['data'].iloc[0],
                'train_end': train_df['data'].iloc[-1],
                'test_start': test_df['data'].iloc[0],
                'test_end': test_df['data'].iloc[-1],
            }
            
            window_idx = end_idx - min_train_days
            yield train_df, test_df, window_idx, date_range

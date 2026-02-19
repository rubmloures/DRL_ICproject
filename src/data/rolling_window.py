"""
Purged Rolling Window Strategy for Backtesting
================================================

Implements sliding window cross-validation for time-series data
with **Purged (Embargo) Gap** to prevent Data Leakage.

The Problem (Data Leakage):
    Standard rolling window trains until day T and tests on T+1.
    Serial correlation in financial data means the model "leaks"
    information from training into testing if they are adjacent.

The Solution (Purged Cross-Validation):
    A gap (embargo) of N days is inserted between train and test.
    This breaks the serial correlation chain.

    Standard (Leaky):
    ┌──────────── TRAIN ────────────┐┌──── TEST ────┐
    │ day 1  ...  day 70            ││ day 71 ... 90 │
    └───────────────────────────────┘└───────────────┘
                                    ↑ LEAKAGE! Adjacent data is correlated.

    Purged (No Leakage):
    ┌──────────── TRAIN ────────────┐  GAP  ┌──── TEST ────┐
    │ day 1  ...  day 70            │ (5d)  │ day 76 ... 95 │
    └───────────────────────────────┘███████└───────────────┘
                                    ↑ EMBARGO breaks serial correlation.

    For K-Fold inside train window, observations adjacent to each
    validation fold boundary are also purged from the train fold:

    Train folds with Purged K-Fold:
    ┌── Fold 0 (train) ──┐  GAP  ┌── Fold 1 (val) ──┐  GAP  ┌── Fold 2 (train) ──┐
    │ discarded near edge │ (Nd) │  validation data   │ (Nd) │ discarded near edge │
    └─────────────────────┘      └────────────────────┘      └─────────────────────┘

References:
    - De Prado, M. L. (2018). Advances in Financial Machine Learning. Wiley.
      Chapter 7: "Cross-Validation in Finance"

Default: 52 weeks training, 4 weeks testing, 5 days purge embargo.

Example:
    >>> df = load_data()
    >>> strategy = RollingWindowStrategy(df, train_weeks=14, test_weeks=4, purge_days=5)
    >>> for train_df, test_df, window_idx, dates in strategy.generate_rolling_windows():
    ...     train_agent(train_df)
    ...     evaluate_agent(test_df)
"""

import logging
from typing import Tuple, Dict, List, Generator, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RollingWindowStrategy:
    """
    Implements Purged Rolling Window (Walk-Forward) strategy for backtesting.
    
    Adds an **Embargo Gap** between train and test sets to prevent data leakage
    caused by serial correlation in financial time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with 'data' column for dates
    train_weeks : int
        Number of weeks for training (default: 14)
    test_weeks : int
        Number of weeks for testing (default: 4)
    overlap_weeks : int
        Overlap between windows in weeks (default: 0)
    purge_days : int
        Number of trading days to embargo (gap) between train and test.
        This prevents data leakage from serial correlation.
        Default: 0 (backward compatible, no gap).
        Recommended: 5 (1 trading week) for daily data.
    with_validation_fold : bool
        Enable K-fold validation inside each train window (default: False)
    k_fold : int
        Number of folds for cross-validation (default: 3)
    purge_kfold_days : int
        Number of trading days to purge around each K-fold boundary.
        Default: same as `purge_days`. Set to 0 to disable K-fold purging.
    
    Attributes
    ----------
    train_days : int
        ~5 trading days per week
    test_days : int
        ~5 trading days per week
    shift_days : int
        Days to shift window (train_weeks - overlap_weeks) * 5
    purge_days : int
        Embargo gap in trading days between train end and test start
    
    Example
    -------
    >>> strategy = RollingWindowStrategy(df, train_weeks=14, test_weeks=4, purge_days=5)
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
        purge_days: int = 0,
        with_validation_fold: bool = False,
        k_fold: int = 3,
        purge_kfold_days: Optional[int] = None,
    ):
        """Initialize purged rolling window strategy."""
        self.df = df.copy()
        self.train_weeks = train_weeks
        self.test_weeks = test_weeks
        self.overlap_weeks = overlap_weeks
        self.with_validation_fold = with_validation_fold
        self.k_fold = k_fold
        
        # --- Purge / Embargo Configuration ---
        self.purge_days = max(0, purge_days)
        # For K-fold purging, default to same as main purge unless explicitly set
        self.purge_kfold_days = self.purge_days if purge_kfold_days is None else max(0, purge_kfold_days)
        
        # Convert weeks to trading days (~5 days per week)
        self.train_days = train_weeks * 5
        self.test_days = test_weeks * 5
        self.shift_days = (train_weeks - overlap_weeks) * 5
        
        # Ensure date column exists
        if 'data' not in self.df.columns:
            logger.warning("'data' column not found. Using index as dates.")
            self.df['data'] = pd.to_datetime(self.df.index)
        
        # --- Logging ---
        logger.info(f"PurgedRollingWindowStrategy initialized:")
        logger.info(f"  Training window: {self.train_weeks} weeks ({self.train_days} days)")
        logger.info(f"  Testing window: {self.test_weeks} weeks ({self.test_days} days)")
        logger.info(f"  Overlap: {self.overlap_weeks} weeks ({self.overlap_weeks * 5} days)")
        logger.info(f"  Window shift: {self.shift_days} days")
        
        if self.purge_days > 0:
            logger.info(f"  ✓ PURGE EMBARGO: {self.purge_days} trading days gap between train/test")
            logger.info(f"    (Prevents data leakage from serial correlation)")
        else:
            logger.warning(f"  ⚠ PURGE DISABLED (purge_days=0). Risk of data leakage!")
        
        if self.with_validation_fold and self.purge_kfold_days > 0:
            logger.info(f"  ✓ K-FOLD PURGE: {self.purge_kfold_days} days purged around fold boundaries")
        
        logger.info(f"  K-fold validation: {with_validation_fold} (k={k_fold if with_validation_fold else 'N/A'})")
        logger.info(f"  Total data points: {len(self.df)}")
    
    def generate_rolling_windows(self) -> Generator[Tuple, None, None]:
        """
        Generate rolling train/test window pairs with purged embargo gap.
        
        The embargo removes `purge_days` observations between the end of
        the training window and the start of the test window, breaking
        serial correlation that would otherwise cause data leakage.
        
        Layout per window (with purge_days=5):
        
            ┌───────── TRAIN ─────────┐ EMBARGO ┌──── TEST ────┐
            │ idx[0] ... idx[train-1] │  5 days │ ... idx[end] │
            └─────────────────────────┘█████████└──────────────┘
        
        Yields
        ------
        train_df : pd.DataFrame
            Training data for this window (before embargo)
        test_df : pd.DataFrame
            Testing data for this window (after embargo)
        window_idx : int
            Window index (0, 1, 2, ...)
        date_range : dict
            Dictionary with 'train_start', 'train_end', 'test_start', 'test_end',
            'purge_start', 'purge_end', 'purge_days'
        
        Example
        -------
        >>> for train_df, test_df, idx, dates in strategy.generate_rolling_windows():
        ...     print(f"Window {idx}: {dates['train_start']} to {dates['test_end']}")
        ...     print(f"  Embargo gap: {dates['purge_days']} days")
        """
        total_length = len(self.df)
        # Minimum required: train + purge + test
        min_required = self.train_days + self.purge_days + self.test_days
        
        if total_length < min_required:
            logger.warning(
                f"Data too short ({total_length} days) for at least one window "
                f"({min_required} days required: {self.train_days} train + "
                f"{self.purge_days} purge + {self.test_days} test)"
            )
            return
        
        n_windows = ((total_length - min_required) // self.shift_days) + 1
        logger.info(f"Generating {n_windows} purged rolling windows (embargo={self.purge_days}d)...")
        
        for window_idx in range(n_windows):
            # Calculate indices for this window
            start_idx = window_idx * self.shift_days
            train_end_idx = start_idx + self.train_days
            
            # --- PURGE EMBARGO: skip `purge_days` observations ---
            purge_start_idx = train_end_idx
            purge_end_idx = train_end_idx + self.purge_days
            
            # Test starts AFTER the embargo gap
            test_start_idx = purge_end_idx
            test_end_idx = test_start_idx + self.test_days
            
            # Check bounds
            if test_end_idx > len(self.df):
                logger.debug(f"Window {window_idx} exceeds data length, stopping")
                break
            
            # Extract data (train and test are NOT adjacent — embargo gap in between)
            train_df = self.df.iloc[start_idx:train_end_idx].copy()
            test_df = self.df.iloc[test_start_idx:test_end_idx].copy()
            
            # Get date range info including embargo details
            train_start = train_df['data'].iloc[0]
            train_end = train_df['data'].iloc[-1]
            test_start = test_df['data'].iloc[0]
            test_end = test_df['data'].iloc[-1]
            
            # Embargo dates (the purged observations)
            if self.purge_days > 0:
                purge_df = self.df.iloc[purge_start_idx:purge_end_idx]
                purge_start_date = purge_df['data'].iloc[0] if len(purge_df) > 0 else None
                purge_end_date = purge_df['data'].iloc[-1] if len(purge_df) > 0 else None
            else:
                purge_start_date = None
                purge_end_date = None
            
            date_range = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                # Purge/Embargo metadata
                'purge_start': purge_start_date,
                'purge_end': purge_end_date,
                'purge_days': self.purge_days,
            }
            
            if self.purge_days > 0:
                logger.debug(
                    f"Window {window_idx}: "
                    f"Train[{start_idx}:{train_end_idx}] "
                    f"EMBARGO[{purge_start_idx}:{purge_end_idx}] ({self.purge_days}d) "
                    f"Test[{test_start_idx}:{test_end_idx}]"
                )
            else:
                logger.debug(
                    f"Window {window_idx}: "
                    f"Train[{start_idx}:{train_end_idx}] "
                    f"Test[{test_start_idx}:{test_end_idx}]"
                )
            
            yield train_df, test_df, window_idx, date_range
    
    def generate_rolling_windows_with_kfold(self) -> Generator[Tuple, None, None]:
        """
        Generate rolling windows with Purged K-fold validation inside train window.
        
        For each fold, observations adjacent to the validation fold boundary
        are purged from the training data to prevent information leakage
        within the cross-validation itself.
        
        Purged K-Fold layout:
        
            ┌─── Train A ──┐ purge ┌── Val ──┐ purge ┌─── Train B ──┐
            │ ... idx[v-p]  │ (Nd)  │ val data│ (Nd)  │ idx[v+s+p]...│
            └───────────────┘██████└─────────┘██████└───────────────┘
            
            Train = Train_A ∪ Train_B  (observations near val removed)
        
        Yields
        ------
        train_fold_df : pd.DataFrame
            Training data for this fold (with purged observations removed)
        val_fold_df : pd.DataFrame
            Validation data for this fold
        test_df : pd.DataFrame
            Test data for this window (after main embargo)
        window_idx : int
            Rolling window index
        fold_idx : int
            K-fold index
        date_range : dict
            Dictionary with date ranges and purge details
        
        Example
        -------
        >>> for train_fold, val_fold, test_df, win_idx, fold_idx, dates in \
        ...         strategy.generate_rolling_windows_with_kfold():
        ...     print(f"Window {win_idx}, Fold {fold_idx}")
        ...     print(f"  Train: {len(train_fold)}, Val: {len(val_fold)}")
        ...     print(f"  Observations purged from train: {dates.get('purged_count', 0)}")
        """
        if not self.with_validation_fold:
            logger.warning("K-fold validation disabled. Use generate_rolling_windows() instead.")
            for train_df, test_df, window_idx, date_range in self.generate_rolling_windows():
                yield train_df, None, test_df, window_idx, 0, date_range
            return
        
        for train_df, test_df, window_idx, date_range in self.generate_rolling_windows():
            # Generate Purged K-fold splits within training data
            kfold_splits = self._kfold_split_train(
                train_df,
                k=self.k_fold,
                purge_days=self.purge_kfold_days,
            )
            
            for fold_idx, (train_fold, val_fold, purged_count) in enumerate(kfold_splits):
                # Dates for this fold
                fold_date_range = {
                    'window_idx': window_idx,
                    'fold_idx': fold_idx,
                    'train_start': train_fold['data'].iloc[0] if len(train_fold) > 0 else None,
                    'train_end': train_fold['data'].iloc[-1] if len(train_fold) > 0 else None,
                    'val_start': val_fold['data'].iloc[0] if len(val_fold) > 0 else None,
                    'val_end': val_fold['data'].iloc[-1] if len(val_fold) > 0 else None,
                    'test_start': test_df['data'].iloc[0] if len(test_df) > 0 else None,
                    'test_end': test_df['data'].iloc[-1] if len(test_df) > 0 else None,
                    # Purge metadata
                    'purge_days_main': self.purge_days,
                    'purge_days_kfold': self.purge_kfold_days,
                    'purged_count': purged_count,
                }
                
                logger.debug(
                    f"Window {window_idx}, Fold {fold_idx}: "
                    f"Train {train_fold.shape[0]} (purged {purged_count}), "
                    f"Val {val_fold.shape[0]}, Test {test_df.shape[0]}"
                )
                
                yield train_fold, val_fold, test_df, window_idx, fold_idx, fold_date_range
    
    @staticmethod
    def _kfold_split_train(
        train_df: pd.DataFrame,
        k: int = 3,
        shuffle: bool = False,
        purge_days: int = 0,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
        """
        Split training data into k folds with Purged Cross-Validation.
        
        For each fold, observations within `purge_days` of the validation
        fold boundary are removed from the training data to prevent
        information leakage through serial correlation.
        
        Maintains temporal order for time series (no shuffling).
        
        Parameters
        ----------
        train_df : pd.DataFrame
            Training data to split
        k : int
            Number of folds
        shuffle : bool
            If False (default), maintains temporal order. 
            WARNING: True is NOT recommended for time series.
        purge_days : int
            Number of observations to purge from training data
            adjacent to each validation fold boundary. Default: 0.
        
        Returns
        -------
        list of (train_fold, val_fold, purged_count) tuples
            purged_count: number of observations removed from training data
        
        Example
        -------
        Purged K-Fold with purge_days=3 on 100-observation dataset (k=3):
        
        Fold 0: Val=[0:33]
            Train = [33+3 : 100] = [36:100]  →  3 obs purged (after val)
            
        Fold 1: Val=[33:66]
            Train = [0 : 33-3] ∪ [66+3 : 100] = [0:30] ∪ [69:100]  →  6 obs purged
            
        Fold 2: Val=[66:100]
            Train = [0 : 66-3] = [0:63]  →  3 obs purged (before val)
        """
        n = len(train_df)
        fold_size = n // k
        
        folds = []
        
        for i in range(k):
            val_start_idx = i * fold_size
            val_end_idx = (i + 1) * fold_size if i < k - 1 else n
            
            val_fold = train_df.iloc[val_start_idx:val_end_idx].copy()
            
            # === PURGED CROSS-VALIDATION ===
            # Remove `purge_days` observations adjacent to validation boundaries
            
            # Training data BEFORE the validation fold (with purge at end)
            train_before_end = max(0, val_start_idx - purge_days)
            train_before = train_df.iloc[:train_before_end]
            
            # Training data AFTER the validation fold (with purge at start)
            train_after_start = min(n, val_end_idx + purge_days)
            train_after = train_df.iloc[train_after_start:]
            
            # Count purged observations
            purged_before = val_start_idx - train_before_end  # purged before val
            purged_after = train_after_start - val_end_idx    # purged after val
            purged_count = purged_before + purged_after
            
            # Combine non-purged training parts
            train_fold_parts = [train_before, train_after]
            train_fold_parts = [p for p in train_fold_parts if len(p) > 0]
            
            if len(train_fold_parts) > 0:
                train_fold = pd.concat(train_fold_parts, ignore_index=True).reset_index(drop=True)
            else:
                train_fold = pd.DataFrame(columns=train_df.columns)
            
            folds.append((train_fold, val_fold, purged_count))
        
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
        Print a summary of a rolling window, including embargo info.
        
        Parameters
        ----------
        date_range : dict
            Dictionary with 'train_start', 'train_end', 'test_start', 'test_end',
            optionally 'purge_start', 'purge_end', 'purge_days'
        """
        print("\n" + "="*70)
        print("Rolling Window Summary (Purged Cross-Validation)")
        print("="*70)
        print(f"Training:  {date_range['train_start']} to {date_range['train_end']}")
        
        purge_days = date_range.get('purge_days', 0)
        if purge_days > 0:
            purge_start = date_range.get('purge_start', '?')
            purge_end = date_range.get('purge_end', '?')
            print(f"EMBARGO:   {purge_start} to {purge_end} ({purge_days} days purged)")
        
        print(f"Testing:   {date_range['test_start']} to {date_range['test_end']}")
        print("="*70)


class ExpandingWindowStrategy:
    """
    Alternative: Expanding window strategy with optional purge embargo.
    Data grows over time: [start, time_t], [start, time_t+1], etc.
    
    Supports the same purge_days embargo gap as RollingWindowStrategy.
    Useful for online learning scenarios.
    """
    
    def __init__(self, df: pd.DataFrame, test_weeks: int = 4, purge_days: int = 0):
        """Initialize expanding window strategy with optional embargo."""
        self.df = df.copy()
        self.test_weeks = test_weeks
        self.test_days = test_weeks * 5
        self.purge_days = max(0, purge_days)
        
        logger.info(f"ExpandingWindowStrategy initialized:")
        logger.info(f"  Testing window: {test_weeks} weeks ({self.test_days} days)")
        if self.purge_days > 0:
            logger.info(f"  ✓ PURGE EMBARGO: {self.purge_days} trading days gap")
        else:
            logger.warning(f"  ⚠ PURGE DISABLED (purge_days=0). Risk of data leakage!")
    
    def generate_expanding_windows(self) -> Generator[Tuple, None, None]:
        """
        Generate expanding train/test window pairs with embargo gap.
        Training data grows, test data is fixed, embargo separates them.
        """
        min_train_days = 20  # Minimum training data
        
        for end_idx in range(min_train_days, len(self.df) - self.purge_days - self.test_days):
            train_df = self.df.iloc[:end_idx].copy()
            
            # Test starts AFTER embargo gap
            test_start = end_idx + self.purge_days
            test_end = test_start + self.test_days
            test_df = self.df.iloc[test_start:test_end].copy()
            
            date_range = {
                'train_start': train_df['data'].iloc[0],
                'train_end': train_df['data'].iloc[-1],
                'test_start': test_df['data'].iloc[0],
                'test_end': test_df['data'].iloc[-1],
                'purge_days': self.purge_days,
            }
            
            window_idx = end_idx - min_train_days
            yield train_df, test_df, window_idx, date_range

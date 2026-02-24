"""
Composite Reward Calculator
============================

Implements dynamic composite reward function for RL agents trading B3 assets.

The reward function combines:
1. Excess Return (portfolio return - risk-free rate/Selic)
2. Downside Risk Penalty (Sortino-like, only negative vol)
3. Differential Return (Alpha vs. Ibovespa benchmark)
4. Transaction Costs (explicit B3 fees)

Weights are dynamically adjusted based on PINN regime detection.

Reference:
- Srivastava et al. (2023) on composite rewards
- B3 pricing for emolumentos and corretagem
- Sortino ratio for downside risk
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from collections import deque
import logging

from .regime_detector import RegimeDetector, MarketRegime

logger = logging.getLogger(__name__)


class CompositeRewardCalculator:
    """
    Calculates composite reward for B3 trading with PINN regime detection.
    
    The reward function is:
    
    R_t = w₁(s_pinn) · R_exc - w₂(s_pinn) · σ_down + w₃(s_pinn) · R_diff - Custos
    
    Where:
    - R_exc: Excess return (portfolio - risk-free rate)
    - σ_down: Downside risk (Sortino component)
    - R_diff: Differential return (portfolio - benchmark)
    - s_pinn: PINN regime signal
    - Custos: Transaction costs
    
    Attributes
    ----------
    regime_detector : RegimeDetector
        Detector for market regime switches
    base_weights : dict
        Base reward component weights before dynamic adjustment
    selic_rates : list
        Historical Selic/CDI daily rates
    ibovespa_returns : list
        Historical Ibovespa returns for benchmark
    window_size : int
        Size of rolling window for volatility calculations
    taxa_selic_path : str
        Path to the historic Selic rate dataset (CSV)
    """
    
    def __init__(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        base_weights: Optional[Dict[str, float]] = None,
        selic_daily_rate: float = 0.00025,  # Fallback default
        transaction_cost_pct: float = 0.0015,  # 0.15% typical for B3
        emolumentos_pct: float = 0.0005,  # ~0.05% B3 emolumentos
        window_size: int = 20,
        weight_ema_alpha: float = 0.9,  # EMA smoothing factor for dynamic weights
        taxa_selic_path: str = "data/raw/taxa_selic.csv",
    ):
        """
        Initialize composite reward calculator.
        
        Parameters
        ----------
        regime_detector : RegimeDetector, optional
            Detector for market regimes (creates default if None)
        base_weights : dict, optional
            Base weights for reward components:
            - 'excess_return': weight for R_exc (default: 1.0)
            - 'downside_risk': weight for σ_down (default: 1.0)
            - 'alpha_return': weight for R_diff (default: 1.0)
            - 'transaction_cost': weight for costs (default: 1.0)
        selic_daily_rate : float
            Daily risk-free rate (Selic/252)
        transaction_cost_pct : float
            Transaction cost as % (corretagem + spread)
        emolumentos_pct : float
            B3 emolumentos as %
        window_size : int
            Window for volatility calculations
        """
        self.regime_detector = regime_detector or RegimeDetector()
        
        # Default base weights
        self.base_weights = base_weights or {
            'excess_return': 1.0,
            'downside_risk': 1.0,
            'alpha_return': 1.0,
            'transaction_cost': 1.0,
        }
        
        self.selic_daily_rate = selic_daily_rate
        self.transaction_cost_pct = transaction_cost_pct
        self.emolumentos_pct = emolumentos_pct
        self.total_cost_pct = transaction_cost_pct + emolumentos_pct
        self.window_size = window_size
        self.weight_ema_alpha = weight_ema_alpha  # EMA smoothing factor
        
        # History tracking
        self.portfolio_returns: deque = deque(maxlen=window_size)
        self.benchmark_returns: deque = deque(maxlen=window_size)
        self.reward_history: deque = deque(maxlen=500)
        
        # Current weights (will be updated each step with EMA smoothing)
        self.current_weights = self.base_weights.copy()
        self.previous_weights = self.base_weights.copy()  # For EMA
        self.current_regime: Optional[MarketRegime] = None
        
        # Load historical Selic data
        self.taxa_selic_path = taxa_selic_path
        self._load_selic_data()
        
        logger.info(
            f"CompositeRewardCalculator initialized:\n"
            f"  Base weights: {self.base_weights}\n"
            f"  Daily risk-free rate: {selic_daily_rate:.6f}\n"
            f"  Transaction costs: {transaction_cost_pct*100:.3f}%\n"
            f"  B3 Emolumentos: {emolumentos_pct*100:.3f}%\n"
            f"  Total costs: {self.total_cost_pct*100:.3f}%\n"
            f"  SELIC data path: {self.taxa_selic_path}"
        )
    
    def _load_selic_data(self) -> None:
        """Load and parse historical SELIC data from CSV."""
        try:
            from pathlib import Path
            import os
            
            # Find the file, allowing for different running directories
            current_dir = Path.cwd()
            possible_paths = [
                Path(self.taxa_selic_path),
                current_dir / self.taxa_selic_path,
                current_dir.parent / self.taxa_selic_path,
                Path("d:/UERJ/Programacao_e_Codigos/DRL_ICproject") / self.taxa_selic_path
            ]
            
            df_selic = None
            for p in possible_paths:
                if p.exists():
                    df_selic = pd.read_csv(p)
                    break
                    
            if df_selic is not None:
                # Convert date matching formats accurately 
                df_selic['data'] = pd.to_datetime(df_selic['data'], format='%d/%m/%Y')
                
                # Convert 'valor' from string "14,9" to float 14.9
                df_selic['valor'] = df_selic['valor'].astype(str).str.replace(',', '.').astype(float)
                
                # Convert annual percentage to daily fractional rate
                # e.g., 14.9% a.a -> (1 + 0.149)^(1/252) - 1 or simple division
                df_selic['daily_rate'] = (1 + (df_selic['valor'] / 100)) ** (1/252) - 1
                
                # Sort and set index for easy lookup
                df_selic = df_selic.sort_values('data').set_index('data')
                
                # Forward fill to handle weekends and missing trading days
                # Map using a complete date range
                full_range = pd.date_range(start=df_selic.index.min(), end=df_selic.index.max())
                self.selic_df = df_selic.reindex(full_range).ffill()
                logger.info(f"Loaded historic SELIC data. Range: {self.selic_df.index.min().date()} to {self.selic_df.index.max().date()}")
            else:
                logger.warning(f"Could not find taxa_selic.csv at {self.taxa_selic_path}. Using fallback rate: {self.selic_daily_rate}")
                self.selic_df = None
        except Exception as e:
            logger.error(f"Failed to load SELIC data from {self.taxa_selic_path}: {e}")
            self.selic_df = None
            
    def _get_daily_selic(self, current_date: Optional[pd.Timestamp]) -> float:
        """Get the applicable daily SELIC given a date, or fallback to default."""
        if current_date is None or self.selic_df is None:
            return self.selic_daily_rate
            
        try:
            date_only = pd.to_datetime(current_date).normalize()
            if date_only in self.selic_df.index:
                return float(self.selic_df.loc[date_only, 'daily_rate'])
            
            # If date is out of range, use the closest bound
            if date_only > self.selic_df.index.max():
                return float(self.selic_df.iloc[-1]['daily_rate'])
            if date_only < self.selic_df.index.min():
                return float(self.selic_df.iloc[0]['daily_rate'])
                
        except Exception as e:
            logger.debug(f"Error extracting SELIC for date {current_date}: {e}")
            
        return self.selic_daily_rate
    
    def calculate_reward(
        self,
        portfolio_return: float,
        benchmark_return: float,
        downside_return: float,
        num_trades: int,
        heston_params: Optional[Dict[str, float]] = None,
        current_volatility: float = 0.01,
        vol_history: Optional[List[float]] = None,
        current_date: Optional[Any] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite reward for the current step.
        
        Parameters
        ----------
        portfolio_return : float
            Portfolio return in this step (e.g., 0.002 = 0.2%)
        benchmark_return : float
            Benchmark (Ibovespa) return
        downside_return : float
            Negative return magnitude (for Sortino)
        num_trades : int
            Number of trades executed in this step
        heston_params : dict, optional
            PINN Heston parameters for regime detection
        current_volatility : float
            Current realized volatility
        vol_history : list, optional
            Historical volatility values
        current_date : any, optional
            Current date to lookup the dynamic SELIC rate
        
        Returns
        -------
        reward : float
            Calculated composite reward
        component_dict : dict
            Breakdown of reward components for logging
        """
        
        # 1. Update regime if PINN available
        if heston_params is not None:
            regime, confidence = self.regime_detector.detect_regime(
                heston_params=heston_params,
                current_volatility=current_volatility,
                vol_history=vol_history,
            )
            self.current_regime = regime
            
            # Get new regime-based weights
            new_weights = self.regime_detector.get_regime_weights(
                regime=regime,
                base_weights=self.base_weights,
            )
            
            # Apply EMA smoothing to prevent abrupt changes in reward signal
            # This stabilizes policy learning when regime switches frequently
            self.current_weights = {
                k: self.weight_ema_alpha * self.previous_weights[k] + 
                   (1 - self.weight_ema_alpha) * new_weights[k]
                for k in new_weights
            }
            self.previous_weights = self.current_weights.copy()
            
            logger.debug(
                f"Weight update for regime {regime.value}: "
                f"EMA(prev, new)={self.current_weights}"
            )
        
        # 2. Calculate reward components
        
        # Determine the applicable risk-free rate for today
        daily_rf_rate = self._get_daily_selic(current_date)
        
        # Component 1: Excess Return (portfolio - risk-free rate)
        excess_return = portfolio_return - daily_rf_rate
        reward_excess = self.current_weights['excess_return'] * excess_return
        
        # Component 2: Downside Risk Penalty (only negative vol)
        # Sortino-like: penalize negative returns more than positive ones
        downside_penalty = max(0, -downside_return) ** 2  # Quadratic penalty
        reward_downside = -self.current_weights['downside_risk'] * downside_penalty
        
        # Component 3: Differential Return (Alpha vs Ibovespa)
        alpha_return = portfolio_return - benchmark_return
        reward_alpha = self.current_weights['alpha_return'] * alpha_return
        
        # Component 4: Transaction Costs
        # Cost is proportional to number of trades
        # Typical: each buy/sell incurs cost_pct% of trade size
        transaction_cost_penalty = self.current_weights['transaction_cost'] * (num_trades * self.total_cost_pct)
        
        # 5. Composite Reward
        reward = (
            reward_excess +
            reward_downside +
            reward_alpha -
            transaction_cost_penalty
        )
        
        # Store in history
        self.portfolio_returns.append(portfolio_return)
        self.benchmark_returns.append(benchmark_return)
        self.reward_history.append(reward)
        
        # Component breakdown for logging
        components = {
            'excess_return': float(excess_return),
            'reward_excess_weighted': float(reward_excess),
            'downside_risk': float(downside_penalty),
            'reward_downside': float(reward_downside),
            'alpha_return': float(alpha_return),
            'reward_alpha_weighted': float(reward_alpha),
            'transaction_cost_penalty': float(transaction_cost_penalty),
            'num_trades': int(num_trades),
            'portfolio_return': float(portfolio_return),
            'benchmark_return': float(benchmark_return),
            'risk_free_rate': float(daily_rf_rate),
            'reward_total': float(reward),
            'regime': self.current_regime.value if self.current_regime else 'unknown',
            'weights': self.current_weights.copy(),
        }
        
        return reward, components
    
    def calculate_sharpe_ratio(self, lookback: int = 252) -> float:
        """
        Calculate Sharpe ratio from recent rewards.
        
        Parameters
        ----------
        lookback : int
            Number of periods to consider
        
        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        if len(self.reward_history) < 2:
            return 0.0
        
        recent_rewards = list(self.reward_history)[-lookback:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        if std_reward == 0:
            return 0.0
        
        sharpe = (mean_reward / std_reward) * np.sqrt(252)
        return float(sharpe)
    
    def calculate_sortino_ratio(self, lookback: int = 252, current_date: Optional[Any] = None) -> float:
        """
        Calculate Sortino ratio using downside deviation only.
        
        Parameters
        ----------
        lookback : int
            Number of periods to consider
        
        Returns
        -------
        float
            Annualized Sortino ratio
        """
        if len(self.portfolio_returns) < 2:
            return 0.0
        
        recent_returns = list(self.portfolio_returns)[-lookback:]
        mean_return = np.mean(recent_returns)
        
        daily_rf_rate = self._get_daily_selic(current_date)
        
        # Downside deviation (only negative returns relative to risk-free)
        downside_returns = [r for r in recent_returns if r < daily_rf_rate]
        if len(downside_returns) == 0:
            downside_dev = 0.0
        else:
            downside_dev = np.sqrt(np.mean(np.array(downside_returns) ** 2))
        
        if downside_dev == 0:
            return float('inf')
        
        excess_return = mean_return - daily_rf_rate
        sortino = (excess_return / downside_dev) * np.sqrt(252)
        return float(sortino)
    
    def get_metrics(self, lookback: int = 252, current_date: Optional[Any] = None) -> Dict[str, float]:
        """
        Get comprehensive reward metrics.
        
        Parameters
        ----------
        lookback : int
            Number of periods for statistics
        current_date : any, optional
            Current date for dynamic SELIC
        
        Returns
        -------
        dict
            Metrics including Sharpe, Sortino, win rate, alpha, etc.
        """
        if len(self.portfolio_returns) < 1:
            return {}
        
        recent_portfolio = list(self.portfolio_returns)[-lookback:]
        recent_benchmark = list(self.benchmark_returns)[-lookback:]
        
        portfolio_cumul = np.prod(1 + np.array(recent_portfolio)) - 1
        benchmark_cumul = np.prod(1 + np.array(recent_benchmark)) - 1
        
        daily_rf_rate = self._get_daily_selic(current_date)
        
        # Win rate (periods beating risk-free rate)
        wins = sum(1 for r in recent_portfolio if r > daily_rf_rate)
        win_rate = wins / len(recent_portfolio) if recent_portfolio else 0.0
        
        # Alpha (cumulative outperformance)
        alpha = portfolio_cumul - benchmark_cumul
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + np.array(recent_portfolio))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        metrics = {
            'sharpe_ratio': self.calculate_sharpe_ratio(lookback),
            'sortino_ratio': self.calculate_sortino_ratio(lookback, current_date),
            'portfolio_return_cumul': float(portfolio_cumul),
            'benchmark_return_cumul': float(benchmark_cumul),
            'alpha': float(alpha),
            'win_rate': float(win_rate),
            'max_drawdown': float(max_drawdown),
            'avg_reward': float(np.mean(list(self.reward_history)[-lookback:])) if self.reward_history else 0.0,
            'regime': self.current_regime.value if self.current_regime else 'unknown',
        }
        
        return metrics
    
    def reset(self) -> None:
        """Reset for new episode."""
        self.portfolio_returns.clear()
        self.benchmark_returns.clear()
        self.reward_history.clear()
        self.regime_detector.reset()
        self.current_regime = None
        self.current_weights = self.base_weights.copy()
        logger.debug("CompositeRewardCalculator reset")


class B3CompositeRewardWrapper:
    """
    Wrapper to integrate CompositeRewardCalculator with StockTradingEnv.
    
    This wrapper intercepts the env.step() reward and processes it through
    the composite reward calculator, returning the adjusted reward.
    
    Usage:
        env = StockTradingEnv(...)
        env.set_reward_function(B3CompositeRewardWrapper(...))
    """
    
    def __init__(
        self,
        reward_calculator: CompositeRewardCalculator,
        pinn_engine=None,
    ):
        """
        Initialize wrapper.
        
        Parameters
        ----------
        reward_calculator : CompositeRewardCalculator
            calculator instance
        pinn_engine : PINNInferenceEngine, optional
            Engine for PINN inference
        """
        self.reward_calculator = reward_calculator
        self.pinn_engine = pinn_engine
        self.prev_portfolio_value = None
        
        logger.info("B3CompositeRewardWrapper initialized")
    
    def transform_reward(
        self,
        env,  # StockTradingEnv instance
        reward: float,
        info: Dict,
    ) -> Tuple[float, Dict]:
        """
        Transform reward from base environment.
        
        Parameters
        ----------
        env : StockTradingEnv
            Trading environment
        reward : float
            Original reward from environment
        info : dict
            Info dict from environment
        
        Returns
        -------
        reward : float
            Composite reward
        info : dict
            Updated info with component breakdown
        """
        
        # Extract necessary info from environment
        portfolio_return = info.get('daily_return', 0.0)
        benchmark_return = info.get('market_return', 0.0)
        num_trades = info.get('trades', 0)
        
        # Get downside return
        downside_return = min(0, portfolio_return)
        
        # Extract current date for SELIC lookup
        current_date_str = info.get('date', None)
        current_date = pd.to_datetime(current_date_str) if current_date_str else None
        
        # Get PINN features if available
        heston_params = None
        if self.pinn_engine is not None and hasattr(env, 'data'):
            try:
                # Extract PINN features from environment state
                # This is simplified; in production, extract from actual data
                heston_params = {}
            except Exception as e:
                logger.debug(f"Could not extract PINN params: {e}")
        
        # Calculate composite reward
        composite_reward, components = self.reward_calculator.calculate_reward(
            portfolio_return=portfolio_return,
            benchmark_return=benchmark_return,
            downside_return=downside_return,
            num_trades=num_trades,
            heston_params=heston_params,
            current_date=current_date,
        )
        
        # Update info
        info['composite_reward'] = composite_reward
        info['reward_components'] = components
        info['current_regime'] = self.reward_calculator.current_regime.value if self.reward_calculator.current_regime else 'unknown'
        
        return composite_reward, info
    
    def reset(self) -> None:
        """Reset for new episode."""
        self.reward_calculator.reset()
        self.prev_portfolio_value = None

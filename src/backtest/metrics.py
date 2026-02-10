"""
Backtest Metrics
================
Performance metrics for trading strategy evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path


class BacktestMetrics:
    """Calculate and track backtesting metrics."""
    
    def __init__(self, initial_capital: float = 100_000.0):
        """
        Initialize metrics calculator.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self) -> None:
        """Reset metrics."""
        self.returns: List[float] = []
        self.portfolio_values: List[float] = [self.initial_capital]
        self.timestamps: List[datetime] = []
        self.trades: List[Dict] = []
    
    def add_return(self, ret: float, timestamp: Optional[datetime] = None) -> None:
        """
        Record daily return.
        
        Args:
            ret: Daily return (as percentage, e.g., 0.01 for 1%)
            timestamp: Timestamp of return
        """
        self.returns.append(ret)
        new_value = self.portfolio_values[-1] * (1 + ret)
        self.portfolio_values.append(new_value)
        
        if timestamp:
            self.timestamps.append(timestamp)
    
    def add_trade(self, 
                  asset: str,
                  action: str,
                  price: float,
                  quantity: float,
                  timestamp: Optional[datetime] = None) -> None:
        """
        Record a trade.
        
        Args:
            asset: Asset symbol
            action: 'BUY' or 'SELL'
            price: Execution price
            quantity: Number of shares
            timestamp: Trade timestamp
        """
        self.trades.append({
            'asset': asset,
            'action': action,
            'price': price,
            'quantity': quantity,
            'value': price * quantity,
            'timestamp': timestamp or datetime.now(),
        })
    
    def sharpe_ratio(self, risk_free_rate: float = 0.10) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 10% for Brazilian rate)
        
        Returns:
            Sharpe ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def sortino_ratio(self, risk_free_rate: float = 0.10) -> float:
        """
        Calculate annualized Sortino ratio (downside deviation).
        
        Args:
            risk_free_rate: Annual risk-free rate
        
        Returns:
            Sortino ratio
        """
        if len(self.returns) < 2:
            return 0.0
        
        returns_array = np.array(self.returns)
        excess_returns = returns_array - (risk_free_rate / 252)
        
        # Downside deviation
        downside_returns = np.minimum(excess_returns, 0)
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns:
            Max drawdown as negative percentage (e.g., -0.15 for 15% DD)
        """
        if len(self.portfolio_values) < 2:
            return 0.0
        
        values_array = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(values_array)
        drawdown = (values_array - running_max) / running_max
        
        return np.min(drawdown)
    
    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Returns:
            Calmar ratio
        """
        annual_return = self.total_return() ** (252 / len(self.returns)) - 1 if len(self.returns) > 0 else 0
        max_dd = abs(self.max_drawdown())
        
        if max_dd == 0:
            return 0.0 if annual_return == 0 else float('inf')
        
        return annual_return / max_dd
    
    def total_return(self) -> float:
        """
        Calculate total return.
        
        Returns:
            Total return as percentage change
        """
        if len(self.portfolio_values) < 2:
            return 0.0
        
        return (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital
    
    def annual_return(self) -> float:
        """
        Calculate annualized return.
        
        Returns:
            Annualized return
        """
        total_ret = self.total_return()
        n_years = len(self.returns) / 252
        
        if n_years <= 0:
            return 0.0
        
        return (1 + total_ret) ** (1 / n_years) - 1
    
    def volatility(self) -> float:
        """
        Calculate annualized volatility.
        
        Returns:
            Annualized volatility
        """
        if len(self.returns) < 2:
            return 0.0
        
        daily_vol = np.std(self.returns)
        return daily_vol * np.sqrt(252)
    
    def win_rate(self) -> float:
        """
        Calculate percentage of winning days.
        
        Returns:
            Win rate (0 to 1)
        """
        if len(self.returns) == 0:
            return 0.0
        
        winning_days = sum(1 for r in self.returns if r > 0)
        return winning_days / len(self.returns)
    
    def profit_factor(self) -> float:
        """
        Calculate profit factor (sum of gains / sum of losses).
        
        Returns:
            Profit factor
        """
        if len(self.returns) == 0:
            return 0.0
        
        returns_array = np.array(self.returns)
        gains = np.sum(returns_array[returns_array > 0])
        losses = abs(np.sum(returns_array[returns_array < 0]))
        
        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        
        return gains / losses
    
    def recovery_factor(self) -> float:
        """
        Calculate recovery factor (net profit / max drawdown).
        
        Returns:
            Recovery factor
        """
        net_profit = self.portfolio_values[-1] - self.initial_capital
        max_dd_value = abs(self.max_drawdown() * self.initial_capital)
        
        if max_dd_value == 0:
            return float('inf') if net_profit > 0 else 0.0
        
        return net_profit / max_dd_value
    
    def diversification_ratio(self, asset_returns: Dict[str, List[float]]) -> float:
        """
        Calculate diversification ratio.
        
        Args:
            asset_returns: Dict of asset -> returns list
        
        Returns:
            Diversification ratio
        """
        if not asset_returns:
            return 0.0
        
        # Weighted average volatility of individual assets
        asset_vols = {}
        for asset, returns in asset_returns.items():
            daily_vol = np.std(returns)
            asset_vols[asset] = daily_vol * np.sqrt(252)
        
        weighted_vol = np.mean(list(asset_vols.values()))
        portfolio_vol = self.volatility()
        
        if portfolio_vol == 0:
            return 0.0
        
        return weighted_vol / portfolio_vol
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get complete metrics summary.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            "Total Return": self.total_return(),
            "Annual Return": self.annual_return(),
            "Volatility": self.volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Sortino Ratio": self.sortino_ratio(),
            "Max Drawdown": self.max_drawdown(),
            "Calmar Ratio": self.calmar_ratio(),
            "Win Rate": self.win_rate(),
            "Profit Factor": self.profit_factor(),
            "Recovery Factor": self.recovery_factor(),
            "Final Portfolio Value": self.portfolio_values[-1],
            "Number of Trades": len(self.trades),
        }
    
    def print_report(self) -> None:
        """Print formatted metrics report."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\nInitial Capital:        R$ {self.initial_capital:>15,.2f}")
        print(f"Final Portfolio Value:  R$ {summary['Final Portfolio Value']:>15,.2f}")
        print(f"Total Return:           {summary['Total Return']:>15.2%}")
        print(f"Annual Return:          {summary['Annual Return']:>15.2%}")
        
        print(f"\nVolatility:             {summary['Volatility']:>15.2%}")
        print(f"Sharpe Ratio:           {summary['Sharpe Ratio']:>15.2f}")
        print(f"Sortino Ratio:          {summary['Sortino Ratio']:>15.2f}")
        
        print(f"\nMax Drawdown:           {summary['Max Drawdown']:>15.2%}")
        print(f"Calmar Ratio:           {summary['Calmar Ratio']:>15.2f}")
        print(f"Recovery Factor:        {summary['Recovery Factor']:>15.2f}")
        
        print(f"\nWin Rate:               {summary['Win Rate']:>15.2%}")
        print(f"Profit Factor:          {summary['Profit Factor']:>15.2f}")
        print(f"Number of Trades:       {summary['Number of Trades']:>15.0f}")
        
        print("\n" + "="*60 + "\n")
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to DataFrame.
        
        Returns:
            DataFrame with daily portfolio values
        """
        df = pd.DataFrame({
            'portfolio_value': self.portfolio_values[1:],  # Skip initial value
            'returns': self.returns,
            'cumulative_return': np.cumprod(np.array(self.returns) + 1) - 1,
        })
        
        if self.timestamps:
            df['timestamp'] = self.timestamps
        
        return df
    
    def save_report(self, filepath: str) -> None:
        """
        Save metrics report to file.
        
        Args:
            filepath: Path to save report
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            summary = self.get_summary()
            f.write("BACKTEST PERFORMANCE METRICS\n")
            f.write("=" * 60 + "\n\n")
            
            for metric, value in summary.items():
                if isinstance(value, float):
                    if metric in ["Total Return", "Annual Return", "Volatility", 
                                   "Max Drawdown", "Win Rate"]:
                        f.write(f"{metric:<25} {value:>15.2%}\n")
                    else:
                        f.write(f"{metric:<25} {value:>15.2f}\n")
                else:
                    f.write(f"{metric:<25} {value:>15}\n")
            
            f.write("\n" + "=" * 60)

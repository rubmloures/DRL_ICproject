"""
Visualization Module
====================
Generate trading performance visualizations using pyfolio and matplotlib.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

logger = logging.getLogger(__name__)


class TradingVisualizer:
    """Generate trading performance visualizations."""
    
    @staticmethod
    def plot_portfolio_value(
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        title: str = "Portfolio Value Over Time",
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot portfolio value against benchmark.
        
        Args:
            df: DataFrame with 'date' and 'account_value' columns
            benchmark_df: Optional benchmark DataFrame
            title: Plot title
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot portfolio
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
        else:
            dates = df.index
        
        ax.plot(dates, df['account_value'], label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_df is not None:
            if 'date' in benchmark_df.columns:
                bench_dates = pd.to_datetime(benchmark_df['date'])
            else:
                bench_dates = benchmark_df.index
            
            # Normalize benchmark to same initial value
            bench_values = benchmark_df['close'].values
            initial_value = df['account_value'].iloc[0]
            bench_normalized = (bench_values / bench_values[0]) * initial_value
            
            ax.plot(bench_dates, bench_normalized, label='Benchmark (Ibovespa)', 
                   linewidth=2, color='red', alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value (R$)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        return fig
    
    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        figsize: Tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """
        Plot returns distribution and statistics.
        
        Args:
            returns: Series of daily returns
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        axes[0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        axes[0].set_xlabel('Daily Returns')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Daily Returns')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod() - 1
        axes[1].plot(cumulative.index, cumulative.values, linewidth=1)
        axes[1].fill_between(cumulative.index, cumulative.values, alpha=0.3)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Cumulative Returns')
        axes[1].set_title('Cumulative Returns Over Time')
        axes[1].grid(True, alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        return fig
    
    @staticmethod
    def plot_drawdown(
        returns: pd.Series,
        figsize: Tuple[int, int] = (14, 6),
    ) -> plt.Figure:
        """
        Plot underwater/drawdown chart.
        
        Args:
            returns: Series of daily returns
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()
        
        # Calculate running maximum
        running_max = cumulative.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative - running_max) / running_max
        
        # Plot
        ax.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.set_title('Underwater Plot (Drawdown from Peak)')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        return fig
    
    @staticmethod
    def plot_actions(
        prices_df: pd.DataFrame,
        actions_df: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 6),
    ) -> plt.Figure:
        """
        Plot price with buy/sell signals.
        
        Args:
            prices_df: DataFrame with price data
            actions_df: DataFrame with trading actions
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get date and price columns
        if 'date' in prices_df.columns:
            dates = pd.to_datetime(prices_df['date'])
            price_col = [col for col in prices_df.columns if 'close' in col.lower()][0]
        else:
            dates = prices_df.index
            price_col = prices_df.columns[0]
        
        prices = prices_df[price_col].values
        
        # Plot price
        ax.plot(dates, prices, label='Price', linewidth=1, color='black', alpha=0.5)
        
        # Plot buy signals (action > 0)
        if 'action' in actions_df.columns:
            buys = actions_df[actions_df['action'] > 0]
            if len(buys) > 0:
                buy_prices = prices_df.loc[buys.index, price_col] if 'date' not in actions_df.columns else prices[buys.index]
                ax.scatter(buys.index if 'date' not in actions_df.columns else pd.to_datetime(buys['date']), 
                          buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
            
            # Plot sell signals (action < 0)
            sells = actions_df[actions_df['action'] < 0]
            if len(sells) > 0:
                sell_prices = prices_df.loc[sells.index, price_col] if 'date' not in actions_df.columns else prices[sells.index]
                ax.scatter(sells.index if 'date' not in actions_df.columns else pd.to_datetime(sells['date']),
                          sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title('Trading Actions (Buy/Sell Signals)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        return fig
    
    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Compare metrics across agents.
        
        Args:
            metrics_dict: Dictionary mapping agent names to metrics
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        agents = list(metrics_dict.keys())
        
        # Plot different metrics
        metric_keys = ['mean_reward', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        for idx, metric_key in enumerate(metric_keys):
            if idx >= len(axes):
                break
            
            values = [metrics_dict[agent].get(metric_key, 0) for agent in agents]
            colors = ['green' if v > 0 else 'red' for v in values]
            
            axes[idx].bar(agents, values, color=colors, alpha=0.7, edgecolor='black')
            axes[idx].set_ylabel(metric_key)
            axes[idx].set_title(f'{metric_key} Comparison')
            axes[idx].grid(True, alpha=0.3, axis='y')
            axes[idx].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_pyfolio_report(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Generate comprehensive performance report with pyfolio.
        
        Args:
            returns: Series of daily returns
            benchmark_returns: Optional benchmark returns
            save_path: Optional path to save report
        
        Returns:
            Dictionary with key performance metrics
        """
        try:
            import pyfolio
        except ImportError:
            logger.error("pyfolio not installed. Install with: pip install pyfolio")
            return {}
        
        try:
            # Calculate metrics
            metrics = {
                'total_return': (returns + 1).prod() - 1,
                'annual_return': returns.mean() * 252,
                'annual_volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
                'calmar_ratio': 0,  # Placeholder
                'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0,
                'best_day': returns.max(),
                'worst_day': returns.min(),
            }
            
            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()
            
            logger.info("Performance Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            # Generate tear sheet if benchmark provided
            if benchmark_returns is not None:
                logger.info("Generating pyfolio tear sheet...")
                try:
                    pyfolio.create_full_tear_sheet(
                        returns=returns,
                        benchmark_rets=benchmark_returns,
                        set_context=False,
                    )
                    
                    if save_path:
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        logger.info(f"Tear sheet saved to {save_path}")
                except Exception as e:
                    logger.warning(f"Could not generate tear sheet: {e}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error generating pyfolio report: {e}")
            return {}

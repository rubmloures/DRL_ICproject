"""
Custom DRL Visualization Module

Generates DRL-specific plots:
1. Buy/Sell actions overlaid on price charts
2. Return distributions and tail metrics
3. CDI comparison and outperformance
4. Regime switching visualization (for PINN-based agents)
5. Portfolio composition over time
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    plt = None
    mpatches = None

logger = logging.getLogger(__name__)


class CustomVisualizer:
    """
    Custom visualizations for DRL trading agents.
    """
    
    def __init__(self, save_dir: str = "./visualizations", dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
            dpi: Resolution for saved figures
        """
        if plt is None:
            raise ImportError("matplotlib required. Install: pip install matplotlib")
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        
    def plot_trading_actions(
        self,
        prices: pd.Series,
        actions: pd.Series,
        account_values: Optional[pd.Series] = None,
        title: str = "DRL Agent Trading Actions",
        figsize: Tuple[int, int] = (16, 8),
        save_as: Optional[str] = None,
    ) -> None:
        """
        Plot buy/sell actions overlaid on price chart.
        
        Args:
            prices: Series of asset prices indexed by date
            actions: Series of actions (1=buy, -1=sell, 0=hold) indexed by date
            account_values: Optional account value series
            title: Plot title
            figsize: Figure size
            save_as: Filename to save (without save_dir)
        """
        fig, axes = plt.subplots(
            2 if account_values is not None else 1,
            1,
            figsize=figsize,
            sharex=True,
        )
        
        if account_values is None:
            ax_price = axes
            ax_value = None
        else:
            ax_price, ax_value = axes
        
        # Plot prices
        ax_price.plot(prices.index, prices.values, label='Asset Price', color='black', linewidth=2)
        ax_price.set_ylabel('Price', fontsize=12)
        ax_price.set_title(title, fontsize=14, fontweight='bold')
        ax_price.grid(True, alpha=0.3)
        
        # Plot buy signals (triangles pointing up)
        buys = actions[actions > 0.5]
        if len(buys) > 0:
            buy_prices = prices.loc[buys.index]
            ax_price.scatter(
                buys.index,
                buy_prices.values,
                marker='^',
                color='green',
                s=200,
                label='Buy',
                zorder=5,
                edgecolors='darkgreen',
                linewidth=1.5,
            )
        
        # Plot sell signals (triangles pointing down)
        sells = actions[actions < -0.5]
        if len(sells) > 0:
            sell_prices = prices.loc[sells.index]
            ax_price.scatter(
                sells.index,
                sell_prices.values,
                marker='v',
                color='red',
                s=200,
                label='Sell',
                zorder=5,
                edgecolors='darkred',
                linewidth=1.5,
            )
        
        ax_price.legend(loc='upper left', fontsize=11)
        
        # Plot account value if available
        if ax_value is not None and account_values is not None:
            ax_value.plot(
                account_values.index,
                account_values.values,
                label='Account Value',
                color='blue',
                linewidth=2,
            )
            ax_value.set_ylabel('Account Value ($)', fontsize=12)
            ax_value.set_xlabel('Date', fontsize=12)
            ax_value.grid(True, alpha=0.3)
            ax_value.legend(loc='upper left', fontsize=11)
            
            # Format y-axis as currency
            ax_value.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
            )
        else:
            ax_price.set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        if save_as:
            save_path = self.save_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved trading actions plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_returns_distribution(
        self,
        agent_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        bins: int = 50,
        figsize: Tuple[int, int] = (14, 6),
        save_as: Optional[str] = None,
    ) -> None:
        """
        Plot return distributions with tail metrics.
        
        Args:
            agent_returns: Series of agent returns
            benchmark_returns: Optional benchmark returns for comparison
            bins: Number of histogram bins
            figsize: Figure size
            save_as: Filename to save
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram
        ax = axes[0]
        ax.hist(agent_returns.dropna(), bins=bins, alpha=0.7, color='blue', edgecolor='black', label='Agent')
        if benchmark_returns is not None:
            ax.hist(benchmark_returns.dropna(), bins=bins, alpha=0.5, color='orange', edgecolor='black', label='Benchmark')
        
        ax.axvline(agent_returns.mean(), color='blue', linestyle='--', linewidth=2, label=f'Agent Mean: {agent_returns.mean():.2%}')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Daily Return', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Return Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Q-Q plot (Normal distribution)
        ax = axes[1]
        from scipy import stats
        stats.probplot(agent_returns.dropna(), dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (vs. Normal Distribution)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_as:
            save_path = self.save_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved returns distribution plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cumulative_returns(
        self,
        agent_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        cdi_returns: Optional[pd.Series] = None,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (14, 7),
        save_as: Optional[str] = None,
    ) -> None:
        """
        Plot cumulative returns with benchmarks.
        
        Args:
            agent_returns: Agent returns series
            benchmark_returns: Ibovespa returns (optional)
            cdi_returns: CDI/risk-free returns (optional)
            log_scale: Use log scale for y-axis
            figsize: Figure size
            save_as: Filename to save
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns
        agent_cumulative = (1 + agent_returns).cumprod()
        ax.plot(
            agent_cumulative.index,
            agent_cumulative.values,
            label='Agent',
            linewidth=2.5,
            color='blue',
        )
        
        if benchmark_returns is not None:
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            ax.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values,
                label='Ibovespa',
                linewidth=2.5,
                color='orange',
                linestyle='--',
            )
        
        if cdi_returns is not None:
            cdi_cumulative = (1 + cdi_returns).cumprod()
            ax.plot(
                cdi_cumulative.index,
                cdi_cumulative.values,
                label='CDI (Risk-Free)',
                linewidth=2.5,
                color='green',
                linestyle=':',
            )
        
        ax.set_ylabel('Cumulative Return (Log Scale)' if log_scale else 'Cumulative Return', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_as:
            save_path = self.save_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved cumulative returns plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_drawdown(
        self,
        returns: pd.Series,
        title: str = "Underwater Plot (Drawdown)",
        figsize: Tuple[int, int] = (14, 7),
        save_as: Optional[str] = None,
    ) -> None:
        """
        Plot drawdown (underwater plot).
        
        Args:
            returns: Series of returns
            title: Plot title
            figsize: Figure size
            save_as: Filename to save
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate cumulative returns and running maximum
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100  # % format
        
        # Plot drawdown as filled area
        ax.fill_between(
            drawdown.index,
            drawdown.values,
            0,
            alpha=0.6,
            color='red',
            label='Drawdown',
        )
        ax.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=11)
        
        # Highlight worst drawdown
        worst_idx = drawdown.idxmin()
        worst_value = drawdown.min()
        ax.scatter([worst_idx], [worst_value], color='darkred', s=100, zorder=5)
        ax.annotate(
            f'Max Drawdown: {worst_value:.1f}%',
            xy=(worst_idx, worst_value),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
        )
        
        plt.tight_layout()
        
        if save_as:
            save_path = self.save_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved drawdown plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 30,
        metrics_list: List[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_as: Optional[str] = None,
    ) -> None:
        """
        Plot rolling performance metrics.
        
        Args:
            returns: Series of returns
            window: Rolling window size (in trading days)
            metrics_list: List of metrics to plot ('sharpe', 'sortino', 'volatility')
            figsize: Figure size
            save_as: Filename to save
        """
        if metrics_list is None:
            metrics_list = ['sharpe', 'sortino', 'volatility']
        
        num_metrics = len(metrics_list)
        fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
        
        if num_metrics == 1:
            axes = [axes]
        
        # Calculate rolling metrics
        for idx, metric in enumerate(metrics_list):
            ax = axes[idx]
            
            if metric == 'sharpe':
                rolling_mean = returns.rolling(window).mean() * 252
                rolling_std = returns.rolling(window).std() * np.sqrt(252)
                rolling_metric = rolling_mean / rolling_std
                label = 'Rolling Sharpe Ratio'
                ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
                
            elif metric == 'sortino':
                rolling_mean = returns.rolling(window).mean() * 252
                downside = returns[returns < 0].rolling(window).std() * np.sqrt(252)
                rolling_metric = rolling_mean / downside
                label = 'Rolling Sortino Ratio'
                ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
                
            elif metric == 'volatility':
                rolling_metric = returns.rolling(window).std() * np.sqrt(252) * 100
                label = 'Rolling Volatility (%)'
                
            else:
                continue
            
            ax.plot(rolling_metric.index, rolling_metric.values, linewidth=2, color='blue')
            ax.fill_between(rolling_metric.index, rolling_metric.values, alpha=0.3, color='blue')
            
            ax.set_ylabel(label, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{label} (Window={window} days)', fontsize=12, fontweight='bold')
        
        axes[-1].set_xlabel('Date', fontsize=12)
        plt.tight_layout()
        
        if save_as:
            save_path = self.save_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved rolling metrics plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_regime_switching(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        regime_colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_as: Optional[str] = None,
    ) -> None:
        """
        Plot regime switching for PINN-based agents.
        
        Args:
            returns: Series of returns
            regimes: Series of regime labels
            regime_colors: Dictionary mapping regime names to colors
            figsize: Figure size
            save_as: Filename to save
        """
        if regime_colors is None:
            regime_colors = {
                'stable_trending': 'green',
                'normal_ranging': 'blue',
                'elevated_volatility': 'orange',
                'turbulent_shock': 'red',
            }
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Plot returns with regime background
        ax = axes[0]
        ax.plot(returns.index, returns.values * 100, label='Daily Returns', color='black', linewidth=1)
        ax.set_ylabel('Daily Return (%)', fontsize=12)
        ax.set_title('Returns with Detected Regimes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Color background by regime
        unique_regimes = regimes.unique()
        for regime in unique_regimes:
            mask = regimes == regime
            regime_periods = returns[mask].index
            
            if len(regime_periods) > 0:
                color = regime_colors.get(regime, 'gray')
                ax.axvspan(
                    regime_periods[0],
                    regime_periods[-1],
                    alpha=0.2,
                    color=color,
                    label=regime if regime in unique_regimes[:1] else '',
                )
        
        # Plot cumulative returns with regime shading
        cumulative = (1 + returns).cumprod()
        ax = axes[1]
        ax.plot(cumulative.index, cumulative.values, label='Cumulative Return', color='blue', linewidth=2)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_title('Cumulative Returns with Regime Indicators', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Color background by regime
        for regime in unique_regimes:
            mask = regimes == regime
            color = regime_colors.get(regime, 'gray')
            ax.axvspan(
                returns[mask].index[0],
                returns[mask].index[-1],
                alpha=0.1,
                color=color,
            )
        
        # Create legend for regimes
        legend_elements = [
            mpatches.Patch(facecolor=regime_colors.get(r, 'gray'), alpha=0.3, label=r)
            for r in unique_regimes
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        
        if save_as:
            save_path = self.save_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved regime switching plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_performance_metrics_table(
        self,
        metrics: Dict[str, float],
        figsize: Tuple[int, int] = (10, 6),
        save_as: Optional[str] = None,
    ) -> None:
        """
        Plot performance metrics as a formatted table.
        
        Args:
            metrics: Dictionary of metrics to display
            figsize: Figure size
            save_as: Filename to save
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Format metrics for display
        formatted_metrics = []
        for key, value in sorted(metrics.items()):
            # Replace underscores with spaces and capitalize
            display_key = key.replace('_', ' ').title()
            
            # Format value based on type
            if isinstance(value, float):
                if 'ratio' in key.lower() or 'correlation' in key.lower():
                    display_value = f"{value:.4f}"
                elif 'rate' in key.lower() or 'return' in key.lower() or 'drawdown' in key.lower():
                    display_value = f"{value:.2%}"
                else:
                    display_value = f"{value:.4f}"
            else:
                display_value = str(value)
            
            formatted_metrics.append([display_key, display_value])
        
        # Create table
        table = ax.table(
            cellText=formatted_metrics,
            colLabels=['Metric', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.6, 0.4],
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(formatted_metrics) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table[(i, j)].set_facecolor('white')
        
        plt.title('Performance Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_as:
            save_path = self.save_dir / save_as
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved metrics table to {save_path}")
        else:
            plt.show()
        
        plt.close()

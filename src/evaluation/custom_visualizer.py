import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

logger = logging.getLogger(__name__)


class CustomVisualizer:
    """
    Custom visualizations for DRL trading agents using Plotly.
    """
    
    def __init__(self, save_dir: str = "./visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_or_show(self, fig: go.Figure, save_as: Optional[str] = None) -> None:
        """Helper to save as HTML or show."""
        if save_as:
            save_path = self.save_dir / save_as
            if not str(save_path).endswith('.html'):
                save_path = save_path.with_suffix('.html')
            fig.write_html(str(save_path))
            logger.info(f"Saved plot to {save_path}")
        else:
            fig.show()

    def plot_trading_actions(
        self,
        prices: pd.Series,
        actions: pd.Series,
        account_values: Optional[pd.Series] = None,
        title: str = "DRL Agent Trading Actions",
        figsize: Tuple[int, int] = (16, 8),
        save_as: Optional[str] = None,
    ) -> None:
        """Plot buy/sell actions overlaid on price chart using Plotly."""
        
        has_account = account_values is not None
        fig = make_subplots(rows=2 if has_account else 1, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.05,
                           subplot_titles=(title, "Account Value") if has_account else (title,))
        
        # Plot prices
        fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name='Price', line=dict(color='black', width=2)), row=1, col=1)
        
        # Actions
        buys = actions[actions > 0.5]
        if len(buys) > 0:
            fig.add_trace(go.Scatter(
                x=buys.index, y=prices.loc[buys.index],
                mode='markers', name='Buy',
                marker=dict(symbol='triangle-up', size=15, color='green', line=dict(width=1, color='darkgreen'))
            ), row=1, col=1)
            
        sells = actions[actions < -0.5]
        if len(sells) > 0:
            fig.add_trace(go.Scatter(
                x=sells.index, y=prices.loc[sells.index],
                mode='markers', name='Sell',
                marker=dict(symbol='triangle-down', size=15, color='red', line=dict(width=1, color='darkred'))
            ), row=1, col=1)
            
        # Account Value
        if has_account:
            fig.add_trace(go.Scatter(x=account_values.index, y=account_values.values, name='Account Value', line=dict(color='blue', width=2)), row=2, col=1)
            fig.update_yaxes(title_text="Value ($)", row=2, col=1, tickformat="$,.0f")
            
        fig.update_layout(template="plotly_white", height=800 if has_account else 500, hovermode="x unified")
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        self._save_or_show(fig, save_as)
    
    def plot_returns_distribution(
        self,
        agent_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        bins: int = 50,
        figsize: Tuple[int, int] = (14, 6),
        save_as: Optional[str] = None,
    ) -> None:
        """Plot return distributions with tail metrics using Plotly."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Return Distribution", "Q-Q Plot"))
        
        # Histogram
        fig.add_trace(go.Histogram(x=agent_returns.dropna(), nbinsx=bins, name='Agent', marker_color='blue', opacity=0.7), row=1, col=1)
        if benchmark_returns is not None:
            fig.add_trace(go.Histogram(x=benchmark_returns.dropna(), nbinsx=bins, name='Benchmark', marker_color='orange', opacity=0.5), row=1, col=1)
            
        fig.add_vline(x=agent_returns.mean(), line_dash="dash", line_color="blue", annotation_text=f"Mean: {agent_returns.mean():.2%}", row=1, col=1)
        fig.add_vline(x=0, line_color="black", line_width=1, row=1, col=1)
        
        # Q-Q Plot approximation in Plotly
        sorted_returns = np.sort(agent_returns.dropna())
        norm_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_returns)))
        
        fig.add_trace(go.Scatter(x=norm_quantiles, y=sorted_returns, mode='markers', name='Q-Q', marker=dict(color='blue', size=4)), row=1, col=2)
        # Identity line for Q-Q
        line_x = np.linspace(norm_quantiles.min(), norm_quantiles.max(), 100)
        # Linear regression for fit line
        slope, intercept, _, _, _ = stats.linregress(norm_quantiles, sorted_returns)
        fig.add_trace(go.Scatter(x=line_x, y=intercept + slope*line_x, mode='lines', name='Fit', line=dict(color='red', dash='dash')), row=1, col=2)
        
        fig.update_layout(template="plotly_white", title_text="<b>Return Analysis</b>")
        fig.update_xaxes(title_text="Daily Return", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        self._save_or_show(fig, save_as)
    
    def plot_cumulative_returns(
        self,
        agent_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        cdi_returns: Optional[pd.Series] = None,
        log_scale: bool = True,
        figsize: Tuple[int, int] = (14, 7),
        save_as: Optional[str] = None,
    ) -> None:
        """Plot cumulative returns with benchmarks using Plotly."""
        fig = go.Figure()
        
        def add_cum_trace(returns, name, color, dash=None):
            cum = (1 + returns).cumprod()
            fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name=name, line=dict(color=color, width=2.5, dash=dash)))
            
        add_cum_trace(agent_returns, "Agent", "blue")
        if benchmark_returns is not None:
            add_cum_trace(benchmark_returns, "Ibovespa", "orange", "dash")
        if cdi_returns is not None:
            add_cum_trace(cdi_returns, "CDI", "green", "dot")
            
        fig.update_layout(
            template="plotly_white",
            title_text="<b>Cumulative Returns Comparison</b>",
            yaxis_type="log" if log_scale else "linear",
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Cumulative Return" + (" (Log Scale)" if log_scale else ""))
        fig.update_xaxes(title_text="Date")
        
        self._save_or_show(fig, save_as)
    
    def plot_drawdown(
        self,
        returns: pd.Series,
        title: str = "Underwater Plot (Drawdown)",
        figsize: Tuple[int, int] = (14, 7),
        save_as: Optional[str] = None,
    ) -> None:
        """Plot drawdown using Plotly."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill='tozeroy', name='Drawdown',
            line=dict(color='darkred', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)'
        ))
        
        # Highlight worst
        worst_idx = drawdown.idxmin()
        worst_val = drawdown.min()
        fig.add_trace(go.Scatter(
            x=[worst_idx], y=[worst_val],
            mode='markers+text',
            text=[f"Max Drawdown: {worst_val:.1f}%"],
            textposition="bottom center",
            marker=dict(color='black', size=10),
            showlegend=False
        ))
        
        fig.update_layout(template="plotly_white", title_text=f"<b>{title}</b>")
        fig.update_yaxes(title_text="Drawdown (%)")
        
        self._save_or_show(fig, save_as)
    
    def plot_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 30,
        metrics_list: List[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_as: Optional[str] = None,
    ) -> None:
        """Plot rolling performance metrics using Plotly."""
        if metrics_list is None:
            metrics_list = ['sharpe', 'sortino', 'volatility']
            
        fig = make_subplots(rows=len(metrics_list), cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        for idx, metric in enumerate(metrics_list):
            row = idx + 1
            if metric == 'sharpe':
                rolling_mean = returns.rolling(window).mean() * 252
                rolling_std = returns.rolling(window).std() * np.sqrt(252)
                val = rolling_mean / rolling_std
                name = 'Rolling Sharpe'
                color = 'blue'
                fig.add_hline(y=0, line_color="black", row=row, col=1)
            elif metric == 'sortino':
                rolling_mean = returns.rolling(window).mean() * 252
                downside = returns[returns < 0].rolling(window).std() * np.sqrt(252)
                val = rolling_mean / downside
                name = 'Rolling Sortino'
                color = 'green'
                fig.add_hline(y=0, line_color="black", row=row, col=1)
            elif metric == 'volatility':
                val = returns.rolling(window).std() * np.sqrt(252) * 100
                name = 'Rolling Volatility (%)'
                color = 'red'
                
            fig.add_trace(go.Scatter(x=val.index, y=val.values, fill='tozeroy', name=name, line=dict(color=color)), row=row, col=1)
            fig.update_yaxes(title_text=name, row=row, col=1)
            
        fig.update_layout(template="plotly_white", height=300 * len(metrics_list), title_text=f"<b>Rolling Metrics (Window={window})</b>")
        self._save_or_show(fig, save_as)
    
    def plot_regime_switching(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        regime_colors: Optional[Dict[str, str]] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_as: Optional[str] = None,
    ) -> None:
        """Plot regime switching using Plotly."""
        if regime_colors is None:
            regime_colors = {
                'stable_trending': 'green',
                'normal_ranging': 'blue',
                'elevated_volatility': 'orange',
                'turbulent_shock': 'red',
            }
            
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        
        # Returns
        fig.add_trace(go.Scatter(x=returns.index, y=returns.values * 100, name='Returns', line=dict(color='black', width=1)), row=1, col=1)
        
        # Cumulative
        cum = (1 + returns).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name='Cumulative', line=dict(color='blue', width=2)), row=2, col=1)
        
        # Add regime shapes (vrect)
        # To avoid too many traces, we iterate and find contiguous blocks
        curr_regime = None
        start_date = None
        
        for date, regime in regimes.items():
            if regime != curr_regime:
                if curr_regime is not None:
                    # Closing previous block
                    fig.add_vrect(x0=start_date, x1=date, fillcolor=regime_colors.get(curr_regime, 'gray'), opacity=0.15, layer="below", line_width=0)
                start_date = date
                curr_regime = regime
        # Last block
        if curr_regime is not None:
            fig.add_vrect(x0=start_date, x1=regimes.index[-1], fillcolor=regime_colors.get(curr_regime, 'gray'), opacity=0.15, layer="below", line_width=0)
            
        fig.update_layout(template="plotly_white", title_text="<b>Regime Switching Analysis</b>", height=700)
        fig.update_yaxes(title_text="Daily Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
        
        self._save_or_show(fig, save_as)
    
    def plot_performance_metrics_table(
        self,
        metrics: Dict[str, float],
        figsize: Tuple[int, int] = (10, 6),
        save_as: Optional[str] = None,
    ) -> None:
        """Plot metrics as a Plotly Table."""
        
        keys = []
        values = []
        for k, v in sorted(metrics.items()):
            keys.append(f"<b>{k.replace('_', ' ').title()}</b>")
            if isinstance(v, float):
                if any(x in k.lower() for x in ['rate', 'return', 'drawdown']):
                    values.append(f"{v:.2%}")
                else:
                    values.append(f"{v:.4f}")
            else:
                values.append(str(v))
                
        fig = go.Figure(data=[go.Table(
            header=dict(values=['<b>Metric</b>', '<b>Value</b>'], fill_color='royalblue', align='left', font=dict(color='white', size=14)),
            cells=dict(values=[keys, values], fill_color='#f0f2f6', align='left', font=dict(size=12))
        )])
        
        fig.update_layout(title="Performance Metrics Summary")
        self._save_or_show(fig, save_as)


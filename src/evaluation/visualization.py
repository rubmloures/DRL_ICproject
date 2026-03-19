import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pandas as pd
from datetime import datetime
import glob
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class TradingVisualizer:
    """Generate trading performance visualizations using Plotly."""
    
    @staticmethod
    def _apply_premium_layout(fig: go.Figure, title: str, x_title: str = "Date", y_title: str = "Value"):
        """Apply a premium dark-themed layout to the figure."""
        fig.update_layout(
            title={
                'text': f"<b>{title}</b>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            xaxis=dict(
                title=x_title,
                gridcolor='rgba(0,0,0,0.05)',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True
            ),
            yaxis=dict(
                title=y_title,
                gridcolor='rgba(0,0,0,0.05)',
                showline=True,
                linewidth=2,
                linecolor='black',
                mirror=True
            )
        )
        return fig

    @staticmethod
    def plot_portfolio_value(
        df: pd.DataFrame,
        benchmark_df: Optional[pd.DataFrame] = None,
        title: str = "Portfolio Value Over Time",
        figsize: Tuple[int, int] = (14, 6),
    ) -> go.Figure:
        """Plot portfolio value against benchmark using Plotly."""
        fig = go.Figure()
        
        # Portfolio line
        dates = pd.to_datetime(df['date']) if 'date' in df.columns else df.index
        fig.add_trace(go.Scatter(
            x=dates, y=df['account_value'],
            mode='lines',
            name='Portfolio',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='Date: %{x}<br>Value: R$ %{y:,.2f}<extra></extra>'
        ))
        
        # Benchmark line
        if benchmark_df is not None:
            bench_dates = pd.to_datetime(benchmark_df['date']) if 'date' in benchmark_df.columns else benchmark_df.index
            bench_values = benchmark_df['close'].values
            initial_value = df['account_value'].iloc[0]
            bench_normalized = (bench_values / bench_values[0]) * initial_value
            
            fig.add_trace(go.Scatter(
                x=bench_dates, y=bench_normalized,
                mode='lines',
                name='Benchmark (Ibovespa)',
                line=dict(color='#d62728', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Benchmark: R$ %{y:,.2f}<extra></extra>'
            ))
            
        TradingVisualizer._apply_premium_layout(fig, title, y_title="Value (R$)")
        return fig
    
    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        figsize: Tuple[int, int] = (12, 4),
    ) -> go.Figure:
        """Plot returns distribution and statistics using Plotly."""
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution of Daily Returns", "Cumulative Returns Over Time"))
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name="Returns", marker_color='#1f77b4', opacity=0.7),
            row=1, col=1
        )
        fig.add_vline(x=returns.mean(), line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {returns.mean():.4f}", row=1, col=1)
        
        # Cumulative returns
        cumulative = (1 + returns).cumprod() - 1
        fig.add_trace(
            go.Scatter(x=cumulative.index, y=cumulative.values, fill='tozeroy', name="Cum. Returns", line_color='#2ca02c'),
            row=1, col=2
        )
        
        fig.update_layout(template="plotly_white", showlegend=False, title_text="<b>Returns Analysis</b>")
        fig.update_xaxes(title_text="Daily Returns", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=2)
        
        return fig
    
    @staticmethod
    def plot_drawdown(
        returns: pd.Series,
        figsize: Tuple[int, int] = (14, 6),
    ) -> go.Figure:
        """Plot underwater/drawdown chart using Plotly."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill='tozeroy',
            mode='lines',
            name='Drawdown',
            line=dict(color='#d62728', width=1),
            fillcolor='rgba(214, 39, 40, 0.3)'
        ))
        
        TradingVisualizer._apply_premium_layout(fig, "Underwater Plot (Drawdown from Peak)", y_title="Drawdown")
        return fig
    
    @staticmethod
    def plot_actions(
        prices_df: pd.DataFrame,
        actions_df: pd.DataFrame,
        figsize: Tuple[int, int] = (15, 6),
    ) -> go.Figure:
        """Plot price with buy/sell signals using Plotly."""
        fig = go.Figure()
        
        if 'date' in prices_df.columns:
            dates = pd.to_datetime(prices_df['date'])
            price_col = [col for col in prices_df.columns if 'close' in col.lower()][0]
        else:
            dates = prices_df.index
            price_col = prices_df.columns[0]
            
        prices = prices_df[price_col].values
        
        fig.add_trace(go.Scatter(
            x=dates, y=prices,
            mode='lines',
            name='Price',
            line=dict(color='rgba(0,0,0,0.5)', width=1)
        ))
        
        if 'action' in actions_df.columns:
            buys = actions_df[actions_df['action'] > 0]
            if len(buys) > 0:
                buy_dates = buys.index if 'date' not in actions_df.columns else pd.to_datetime(buys['date'])
                buy_prices = prices_df.loc[buys.index, price_col] if 'date' not in actions_df.columns else prices[buys.index]
                fig.add_trace(go.Scatter(
                    x=buy_dates, y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='darkgreen'))
                ))
                
            sells = actions_df[actions_df['action'] < 0]
            if len(sells) > 0:
                sell_dates = sells.index if 'date' not in actions_df.columns else pd.to_datetime(sells['date'])
                sell_prices = prices_df.loc[sells.index, price_col] if 'date' not in actions_df.columns else prices[sells.index]
                fig.add_trace(go.Scatter(
                    x=sell_dates, y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='darkred'))
                ))
        
        TradingVisualizer._apply_premium_layout(fig, "Trading Actions (Buy/Sell Signals)", y_title="Price")
        return fig
    
    @staticmethod
    def plot_metrics_comparison(
        metrics_dict: Dict[str, Dict[str, float]],
        figsize: Tuple[int, int] = (12, 6),
    ) -> go.Figure:
        """Compare metrics across agents using Plotly."""
        agents = list(metrics_dict.keys())
        metric_keys = ['mean_reward', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f"{m.replace('_', ' ').title()} Comparison" for m in metric_keys])
        
        for idx, metric_key in enumerate(metric_keys):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            values = [metrics_dict[agent].get(metric_key, 0) for agent in agents]
            colors = ['#2ca02c' if v > 0 else '#d62728' for v in values]
            
            fig.add_trace(
                go.Bar(x=agents, y=values, marker_color=colors, name=metric_key, showlegend=False),
                row=row, col=col
            )
            
        fig.update_layout(template="plotly_white", title_text="<b>Agent Metrics Comparison</b>", height=700)
        return fig
    
    @staticmethod
    def generate_pyfolio_report(
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        save_path: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Generate metrics summary (Plotly cannot replace pyfolio tear sheet directly)."""
        metrics = {
            'total_return': (returns + 1).prod() - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0,
            'best_day': returns.max(),
            'worst_day': returns.min(),
        }
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = drawdown.min()
        
        logger.info("Performance Metrics calculated successfully.")
        return metrics

    @staticmethod
    def plot_sb3_training_metrics(log_dir: str) -> go.Figure:
        """Plot SB3 training metrics from progress.csv using Plotly."""
        progress_files = glob.glob(f"{log_dir}/**/progress.csv", recursive=True)
        if not progress_files:
            logger.warning(f"No progress.csv found in {log_dir}")
            return go.Figure()
        
        # Take the most recent one
        latest_file = max(progress_files, key=os.path.getmtime)
        try:
            df = pd.read_csv(latest_file)
        except Exception as e:
            logger.error(f"Error reading {latest_file}: {e}")
            return go.Figure()
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=('Episode Reward Mean', 'Entropy Loss', 'Value Loss'))
        
        if 'rollout/ep_rew_mean' in df.columns:
            fig.add_trace(go.Scatter(y=df['rollout/ep_rew_mean'], mode='lines', name='Reward', line_color='#2ca02c'), row=1, col=1)
            
        if 'train/entropy_loss' in df.columns:
            fig.add_trace(go.Scatter(y=df['train/entropy_loss'], mode='lines', name='Entropy', line_color='#ff7f0e'), row=1, col=2)
            
        if 'train/value_loss' in df.columns:
            fig.add_trace(go.Scatter(y=df['train/value_loss'], mode='lines', name='Value Loss', line_color='#d62728'), row=1, col=3)
            
        fig.update_layout(
            template="plotly_white", 
            title_text="<b>Stable Baselines3 Training Health</b>", 
            showlegend=False,
            height=400
        )
        return fig

    @staticmethod
    def _downsample_df(df: pd.DataFrame, max_points: int = 5000) -> pd.DataFrame:
        """Downsample DataFrame for visualization if it exceeds max_points."""
        if len(df) > max_points:
            # We want to keep the trend, so we'll take every N-th point
            nth = len(df) // max_points
            return df.iloc[::nth].copy()
        return df

    @staticmethod
    def plot_regime_efficacy(audit_df: pd.DataFrame, title: str = "Market Volatility Regimes - Strategy Alignment") -> go.Figure:
        """
        Plot portfolio equity with background colored by market volatility regimes.
        Adopted 'Volatility Regime' approach for maximum clarity.
        """
        if 'regime' not in audit_df.columns or 'reward' not in audit_df.columns:
            logger.warning("Required columns (regime, reward) not found in audit log.")
            return go.Figure()

        # Step 1: Clean Data
        audit_df = audit_df.copy()
        # Drop rows with NaN in critical columns
        audit_df = audit_df.dropna(subset=['regime', 'reward'])
        
        # Downsample for performance if needed
        audit_df = TradingVisualizer._downsample_df(audit_df, max_points=3000)
        
        # Reconstruct equity
        if 'equity' not in audit_df.columns:
            audit_df['equity'] = audit_df['reward'].cumsum()
            
        fig = go.Figure()
        
        # Define Volatility Regime color mapping (Modern Palette)
        color_map = {
            'stable_trending': 'rgba(46, 204, 113, 0.2)',    # Soft Green
            'normal_ranging':  'rgba(52, 152, 219, 0.2)',   # Soft Blue
            'elevated_vol':    'rgba(241, 194, 50, 0.2)',   # Soft Yellow/Gold
            'turbulent_shock': 'rgba(231, 76, 60, 0.2)',    # Soft Red
        }
        
        # Mapping for markers (Opaque)
        marker_color_map = {
            'stable_trending': '#27ae60', 
            'normal_ranging':  '#2980b9',
            'elevated_vol':    '#f39c12',
            'turbulent_shock': '#c0392b',
        }

        # Step 2: Add background 'vrect' for each regime segment to create the "Volatility Regime" look
        regime_series = audit_df['regime'].apply(lambda x: str(x).lower().split('.')[-1])
        
        # Find continuous segments of the same regime
        curr_regime = None
        start_idx = 0
        
        indices = audit_df.index.tolist()
        for i, idx in enumerate(indices):
            reg = regime_series.iloc[i]
            if reg != curr_regime:
                if curr_regime is not None:
                    # Add rectangle for previous segment
                    fig.add_vrect(
                        x0=indices[start_idx], x1=idx,
                        fillcolor=color_map.get(curr_regime, 'rgba(128,128,128,0.1)'),
                        layer="below", line_width=0,
                        name=curr_regime.upper()
                    )
                curr_regime = reg
                start_idx = i
        
        # Final segment
        if curr_regime is not None:
             fig.add_vrect(
                x0=indices[start_idx], x1=indices[-1],
                fillcolor=color_map.get(curr_regime, 'rgba(128,128,128,0.1)'),
                layer="below", line_width=0,
            )

        # Step 3: Plot Equity Line
        fig.add_trace(go.Scatter(
            x=audit_df.index, y=audit_df['equity'],
            mode='lines', name='Portfolio Equity', 
            line=dict(color='#2c3e50', width=2.5)
        ))
        
        # Step 4: Add markers to highlight regime changes/points
        for regime in audit_df['regime'].unique():
            reg_str = str(regime).lower().split('.')[-1]
            if reg_str not in marker_color_map: continue # Skip weird IDs
            
            regime_points = audit_df[audit_df['regime'].apply(lambda x: str(x).lower().split('.')[-1]) == reg_str]
            
            fig.add_trace(go.Scatter(
                x=regime_points.index, y=regime_points['equity'],
                mode='markers', name=reg_str.replace('_', ' ').upper(),
                marker=dict(color=marker_color_map.get(reg_str, '#7f7f7f'), size=4, opacity=0.5),
                showlegend=True
            ))
            
        TradingVisualizer._apply_premium_layout(fig, title, x_title="Steps (Audit Log)", y_title="Cumulative Return (Scalado)")
        fig.update_layout(legend_title_text="Regimes")
        return fig

    @staticmethod
    def plot_heston_v_results(audit_df: pd.DataFrame, title: str = "PINN Context: Heston Physics vs. Portfolio Performance") -> go.Figure:
        """
        Compare Heston parameters (Volatility, Risk) with Portfolio Rewards.
        Cleaner visualization with dual axes or separate panels.
        """
        if not any(col.startswith('pinn_') for col in audit_df.columns):
            logger.warning("No PINN features found in audit log for Heston plot.")
            return go.Figure()

        # Downsample
        audit_df = TradingVisualizer._downsample_df(audit_df, max_points=2000)
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1,
                            subplot_titles=("Cumulative Strategic Reward", "Market Physics (Heston Dynamics)"))

        # Cumulative rewards - Use a more distinctive color
        cum_reward = audit_df['reward'].cumsum()
        fig.add_trace(go.Scatter(x=audit_df.index, y=cum_reward, name="Agent Equity", 
                                line=dict(color="#2980b9", width=3), fill='tozeroy'), row=1, col=1)

        # Heston Params - Use 'pinn_raw_' if available, else 'pinn_'
        pinn_cols = [c for c in audit_df.columns if c.startswith('pinn_raw_')]
        if not pinn_cols:
            pinn_cols = [c for c in audit_df.columns if c.startswith('pinn_')]
            
        for col in pinn_cols:
             # Skip internal normalized cols if raw exists
            if 'norm' in col and any(col.replace('norm', 'raw') in pinn_cols for _ in [1]): continue
            
            label = col.replace("pinn_raw_", "").replace("pinn_", "").upper()
            fig.add_trace(go.Scatter(x=audit_df.index, y=audit_df[col], 
                                    name=label,
                                    line=dict(width=1.5), opacity=0.8), row=2, col=1)

        fig.update_layout(height=700, template="plotly_white", title_text=f"<b>{title}</b>")
        fig.update_yaxes(title_text="Reward Units", row=1, col=1)
        fig.update_yaxes(title_text="Latent Parameter Value", row=2, col=1)
        return fig

    @staticmethod
    def plot_regime_clusters(audit_df: pd.DataFrame, title: str = "Market Regime Clusters (Heston Latent Space)") -> go.Figure:
        """
        Scatter plot of Heston Parameters (Volatility vs Vol-of-Vol) colored by detected regime.
        Fixes NaN and large ID issues.
        """
        required = ['regime', 'pinn_nu', 'pinn_xi']
        if not all(col in audit_df.columns for col in required):
            logger.warning(f"Columns {required} not found for regime cluster plot.")
            return go.Figure()

        # Step 1: Clean Data - Critical for the User's report
        audit_df = audit_df.copy()
        
        # Drop NaNs
        audit_df = audit_df.dropna(subset=['pinn_nu', 'pinn_xi', 'regime'])
        
        # Parse regime string properly
        def parse_regime(x):
            s = str(x).lower().split('.')[-1]
            # Valid regimes should be one of these. If it's a huge number or garbage, return 'uncertain'
            valid = ['stable_trending', 'normal_ranging', 'elevated_vol', 'turbulent_shock']
            if s in valid:
                return s
            if s.isdigit() and len(s) > 10: # Handle the user's large number issue
                return 'unknown_cluster'
            return s

        audit_df['regime_clean'] = audit_df['regime'].apply(parse_regime)
        
        # Filter out extreme outliers in PINN space (can mess up scaling)
        for col in ['pinn_nu', 'pinn_xi']:
            p99 = audit_df[col].quantile(0.99)
            audit_df = audit_df[audit_df[col] <= p99 * 1.5]

        # Downsample
        audit_df = TradingVisualizer._downsample_df(audit_df, max_points=2500)
        
        color_map = {
            'stable_trending': '#27ae60',
            'normal_ranging':  '#2980b9',
            'elevated_vol':    '#f39c12',
            'turbulent_shock': '#c0392b',
            'unknown_cluster': '#7f8c8d'
        }

        fig = px.scatter(
            audit_df, 
            x="pinn_nu", 
            y="pinn_xi", 
            color="regime_clean",
            hover_data=['regime', 'reward'],
            color_discrete_map=color_map,
            title=f"<b>{title}</b>",
            labels={'pinn_nu': 'Nu (Mean Volatility)', 'pinn_xi': 'Xi (Vol-of-Vol)'}
        )

        fig.update_traces(marker=dict(size=7, opacity=0.7, line=dict(width=0.5, color='white')))
        fig.update_layout(
            template="plotly_white", 
            legend_title_text="Clustered Regimes",
            xaxis=dict(gridcolor='rgba(0,0,0,0.05)'),
            yaxis=dict(gridcolor='rgba(0,0,0,0.05)')
        )
        return fig

    @staticmethod
    def plot_regime_vs_actions(audit_df: pd.DataFrame, title: str = "Regime Alignment: Actions vs Risk") -> go.Figure:
        """
        Visualizes how the agent's action intensity changes across different regimes.
        """
        if 'regime' not in audit_df.columns or 'action' not in audit_df.columns:
            return go.Figure()

        # Parse action strings if needed
        def get_intensity(act):
            try:
                if isinstance(act, str):
                    import ast
                    act = np.array(ast.literal_eval(act.replace("nan", "0")))
                return np.mean(np.abs(act))
            except:
                return 0.0

        audit_df['action_intensity'] = audit_df['action'].apply(get_intensity)

        # Remove intensity=0 to avoid log scale issues if needed, or just better outliers
        audit_df = audit_df[audit_df['action_intensity'] > 1e-6]

        fig = px.box(audit_df, x="regime", y="action_intensity", color="regime",
                     points="outliers",
                     title=f"<b>{title}</b>",
                     color_discrete_map={
                         'stable_trending': '#2ca02c',
                         'normal_ranging': '#1f77b4',
                         'elevated_vol': '#ff7f0e',
                         'turbulent_shock': '#d62728'
                     })
        
        fig.update_layout(
            template="plotly_white", 
            showlegend=False,
            yaxis_type="log", # Action intensity can vary orders of magnitude
            yaxis_title="Action Intensity (Log Scale)"
        )
        return fig

    @staticmethod
    def plot_strategy_vs_benchmark(portfolio_history: list, benchmark_prices: list, dates: list) -> go.Figure:
        """Compare strategy against benchmark rebased to 100 using Plotly."""
        min_len = min(len(portfolio_history), len(benchmark_prices), len(dates))
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates[:min_len]),
            'Strategy': (np.array(portfolio_history[:min_len]) / portfolio_history[0]) * 100,
            'Benchmark': (np.array(benchmark_prices[:min_len]) / benchmark_prices[0]) * 100
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Strategy'], name='Ensemble DRL + PINN', line=dict(color='#1f77b4', width=3)))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Benchmark'], name='Buy & Hold (Benchmark)', line=dict(color='gray', width=2, dash='dash')))
        
        # Fill area where strategy outperforms
        # Note: Plotly doesn't have a direct 'where' in fill_between, but we can simulate it or just use simple fill
        
        TradingVisualizer._apply_premium_layout(fig, "Strategy vs Benchmark (Base 100)", y_title="Cumulative Return")
        return fig

    @staticmethod
    def plot_rolling_volatility(portfolio_history: list) -> go.Figure:
        """Plot rolling volatility using Plotly."""
        returns = pd.Series(portfolio_history).pct_change().dropna()
        rolling_vol_21d = returns.rolling(window=21).std() * np.sqrt(252)
        rolling_vol_63d = returns.rolling(window=63).std() * np.sqrt(252)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=rolling_vol_21d, name='Rolling Vol (21d)', line=dict(color='purple', width=1), opacity=0.6))
        fig.add_trace(go.Scatter(y=rolling_vol_63d, name='Rolling Vol (63d)', line=dict(color='black', width=2)))
        
        fig.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="Risk Limit (20%)")
        
        TradingVisualizer._apply_premium_layout(fig, "Dynamic Risk Profile (Rolling Volatility)", x_title="Days", y_title="Annualized Volatility")
        return fig


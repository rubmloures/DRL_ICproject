import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional

class ABTestVisualizer:
    """
    Visualizer for A/B Testing results (Baseline vs PINN-Enhanced) using Plotly.
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def load_metric_files(self, prefix_a: str = "group_A_pinn_", prefix_b: str = "group_B_baseline_"):
        """Load result files for both groups."""
        path_a = os.path.join(self.results_dir, f"{prefix_a}rolling_ensemble_metrics.json")
        path_b = os.path.join(self.results_dir, f"{prefix_b}rolling_ensemble_metrics.json")
        
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            return None, None
            
        with open(path_a, 'r') as f:
            data_a = json.load(f)
        with open(path_b, 'r') as f:
            data_b = json.load(f)
            
        return data_a, data_b

    def plot_cumulative_returns(self, data_a: Dict, data_b: Dict, title: str = "A/B Test Comparison"):
        """Plot rolling performance metrics using Plotly."""
        path_a = os.path.join(self.results_dir, "group_A_pinn_rolling_ensemble_windows.csv")
        path_b = os.path.join(self.results_dir, "group_B_baseline_rolling_ensemble_windows.csv")
        
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            return

        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        
        # Sharpe Comparison
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(x=df_a['window_end_date'], y=df_a['ensemble_sharpe'], name='PINN-Enhanced (A)', line=dict(color='blue', width=3), marker=dict(symbol='circle')))
        fig_sharpe.add_trace(go.Scatter(x=df_a['window_end_date'], y=df_b['ensemble_sharpe'], name='Baseline (B)', line=dict(color='gray', width=2, dash='dash'), marker=dict(symbol='x')))
        
        fig_sharpe.update_layout(template="plotly_white", title="<b>Rolling Window Sharpe Ratio Comparison</b>", xaxis_title="Window End Date", yaxis_title="Sharpe Ratio")
        fig_sharpe.write_html(os.path.join(self.plots_dir, "ab_sharpe_comparison.html"))
        
        # Return Comparison
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(x=df_a['window_end_date'], y=df_a['ensemble_annual_return'], name='PINN-Enhanced (A)', line=dict(color='green', width=3), marker=dict(symbol='circle')))
        fig_ret.add_trace(go.Scatter(x=df_a['window_end_date'], y=df_b['ensemble_annual_return'], name='Baseline (B)', line=dict(color='gray', width=2, dash='dash'), marker=dict(symbol='x')))
        
        fig_ret.update_layout(template="plotly_white", title="<b>Rolling Window Annual Return Comparison</b>", xaxis_title="Window End Date", yaxis_title="Annual Return")
        fig_ret.write_html(os.path.join(self.plots_dir, "ab_return_comparison.html"))

    def plot_metrics_comparison(self, data_a: Dict, data_b: Dict):
        """Bar chart comparing aggregated metrics using Plotly."""
        agg_a = data_a['aggregated']
        agg_b = data_b['aggregated']
        
        metrics = ['avg_ensemble_sharpe', 'avg_ensemble_annual_return', 'avg_ensemble_max_drawdown']
        labels = ['Sharpe Ratio', 'Annual Return', 'Max Drawdown']
        
        values_a = [agg_a.get(m, 0) for m in metrics]
        values_b = [agg_b.get(m, 0) for m in metrics]
        
        fig = go.Figure(data=[
            go.Bar(name='PINN (A)', x=labels, y=values_a, marker_color='royalblue', text=[f"{v:.2f}" for v in values_a], textposition='auto'),
            go.Bar(name='Baseline (B)', x=labels, y=values_b, marker_color='lightgray', text=[f"{v:.2f}" for v in values_b], textposition='auto')
        ])
        
        fig.update_layout(barmode='group', template="plotly_white", title="<b>Aggregated Performance Metrics Comparison</b>")
        fig.write_html(os.path.join(self.plots_dir, "ab_metrics_bar.html"))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate A/B Test Visualizations")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing result JSON/CSV files")
    args = parser.parse_args()
    
    viz = ABTestVisualizer(args.results_dir)
    data_a, data_b = viz.load_metric_files()
    
    if data_a and data_b:
        viz.plot_cumulative_returns(data_a, data_b)
        viz.plot_metrics_comparison(data_a, data_b)
        print(f"Plots saved as HTML in {viz.plots_dir}")
    else:
        print("Could not load data.")

if __name__ == "__main__":
    main()



import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

class ABTestVisualizer:
    """
    Visualizer for A/B Testing results (Baseline vs PINN-Enhanced).
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.2)
        
    def load_metric_files(self, prefix_a: str = "group_A_pinn_", prefix_b: str = "group_B_baseline_"):
        """
        Load result files for both groups.
        """
        path_a = os.path.join(self.results_dir, f"{prefix_a}rolling_ensemble_metrics.json")
        path_b = os.path.join(self.results_dir, f"{prefix_b}rolling_ensemble_metrics.json")
        
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print(f"Warning: Result files not found in {self.results_dir}")
            return None, None
            
        with open(path_a, 'r') as f:
            data_a = json.load(f)
        with open(path_b, 'r') as f:
            data_b = json.load(f)
            
        return data_a, data_b

    def plot_cumulative_returns(self, data_a: Dict, data_b: Dict, title: str = "A/B Test: Cumulative Portfolio Performance"):
        """
        Plot equity curves for both models.
        Assumes data_a/b contains 'window_results' which has 'daily_returns' or similar, 
        OR we rely on aggregate metrics. 
        Actually, 'rolling_ensemble_windows' dataframe is better for this.
        Let's try to load the CSVs instead for time series data.
        """
        # Load CSVs for time series
        path_a = os.path.join(self.results_dir, "group_A_pinn_rolling_ensemble_windows.csv")
        path_b = os.path.join(self.results_dir, "group_B_baseline_rolling_ensemble_windows.csv")
        
        if not os.path.exists(path_a) or not os.path.exists(path_b):
            print("Warning: Window CSV files not found. Skipping equity plot.")
            return

        df_a = pd.read_csv(path_a)
        df_b = pd.read_csv(path_b)
        
        plt.figure(figsize=(12, 6))
        
        # Plotting Sharpe per window as a proxy for performance stability over time
        # Or Total Return per window
        
        plt.plot(df_a['window_end_date'], df_a['ensemble_sharpe'], marker='o', label='PINN-Enhanced (A)', linewidth=2, color='blue')
        plt.plot(df_b['window_end_date'], df_b['ensemble_sharpe'], marker='x', label='Baseline (B)', linewidth=2, color='gray', linestyle='--')
        
        plt.title("Rolling Window Sharpe Ratio Comparison")
        plt.xlabel("Window End Date")
        plt.ylabel("Sharpe Ratio")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, "ab_sharpe_comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved Sharpe plot to {save_path}")
        plt.close()
        
        # Also plot Average Annual Return per window
        plt.figure(figsize=(12, 6))
        plt.plot(df_a['window_end_date'], df_a['ensemble_annual_return'], marker='o', label='PINN-Enhanced (A)', linewidth=2, color='green')
        plt.plot(df_b['window_end_date'], df_b['ensemble_annual_return'], marker='x', label='Baseline (B)', linewidth=2, color='gray', linestyle='--')
        
        plt.title("Rolling Window Annual Return Comparison")
        plt.xlabel("Window End Date")
        plt.ylabel("Annual Return")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(self.plots_dir, "ab_return_comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved Return plot to {save_path}")
        plt.close()

    def plot_metrics_comparison(self, data_a: Dict, data_b: Dict):
        """
        Bar chart comparing aggregated metrics.
        """
        agg_a = data_a['aggregated']
        agg_b = data_b['aggregated']
        
        metrics = ['avg_ensemble_sharpe', 'avg_ensemble_annual_return', 'avg_ensemble_max_drawdown']
        labels = ['Sharpe Ratio', 'Annual Return', 'Max Drawdown']
        
        values_a = [agg_a.get(m, 0) for m in metrics]
        values_b = [agg_b.get(m, 0) for m in metrics]
        
        x = range(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(10, 6))
        plt.bar([i - width/2 for i in x], values_a, width, label='PINN (A)', color='royalblue')
        plt.bar([i + width/2 for i in x], values_b, width, label='Baseline (B)', color='lightgray')
        
        plt.xticks(x, labels)
        plt.title("Aggregated Performance Metrics (Average across all windows)")
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(values_a):
            plt.text(i - width/2, v + (0.01 if v>0 else -0.05), f"{v:.2f}", ha='center', fontweight='bold')
        for i, v in enumerate(values_b):
            plt.text(i + width/2, v + (0.01 if v>0 else -0.05), f"{v:.2f}", ha='center')
            
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, "ab_metrics_bar.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved Metrics Bar plot to {save_path}")
        plt.close()

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
        print("Done.")
    else:
        print("Could not load data.")

if __name__ == "__main__":
    main()

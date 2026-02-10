"""
Example 10: Comprehensive DRL Agent Evaluation Pipeline

Demonstrates the complete three-phase evaluation framework:
1. Training Monitoring (convergence metrics)
2. Backtesting with Pyfolio
3. Custom DRL-specific visualizations

This example shows:
- Setting up TensorBoard-connected training
- Converting account values to returns for Pyfolio
- Generating tear sheets with Ibovespa/CDI benchmarks
- Visualizing trading actions on price charts
- Analyzing regime switching behavior
- Comparing performance metrics

Run with:
    python examples/10_evaluation_pipeline.py --assets VALE3 PETR4 --episodes 5
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import A2C, PPO
    from gymnasium import Env
except ImportError:
    print("ERROR: stable_baselines3 and gymnasium required")
    print("Install: pip install stable-baselines3 gymnasium")
    exit(1)

from src.data import load_and_prepare_data, TemporalDataSplitter, PINNDataPreprocessor
from src.env import StockTradingEnv
from src.evaluation import (
    BacktestEvaluator,
    BenchmarkData,
    CustomVisualizer,
    TrainingMonitor,
    setup_backtesting,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Complete evaluation pipeline for DRL agents."""
    
    def __init__(
        self,
        assets: list = None,
        start_date: str = "2023-01-01",
        test_start_date: str = "2024-01-01",
        episodes: int = 10,
        pinn_enabled: bool = False,
        output_dir: str = "./evaluation_results",
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            assets: List of ticker symbols
            start_date: Training period start
            test_start_date: Test/backtest period start
            episodes: Number of training episodes
            pinn_enabled: Whether to use PINN features
            output_dir: Output directory for results
        """
        self.assets = assets or ['VALE3', 'PETR4']
        self.start_date = start_date
        self.test_start_date = test_start_date
        self.episodes = episodes
        self.pinn_enabled = pinn_enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.training_monitor = None
        self.backtest_evaluator = None
        self.visualizer = None
        self.agent = None
        self.env = None
        self.test_results = None
        
        logger.info(f"EvaluationPipeline initialized")
        logger.info(f"Assets: {self.assets}")
        logger.info(f"Training period: {start_date} to {test_start_date}")
    
    def setup_phase1_training_monitoring(self) -> TrainingMonitor:
        """
        Setup Phase 1: Training Convergence Monitoring.
        
        Returns:
            Configured TrainingMonitor
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: TRAINING MONITORING SETUP")
        logger.info("="*70)
        
        self.training_monitor = TrainingMonitor(
            log_dir=str(self.output_dir / "training_logs"),
            model_name=f"pinn" if self.pinn_enabled else "standard",
            enable_tensorboard=True,
        )
        
        logger.info("✓ TrainingMonitor initialized")
        logger.info(f"  Model: {'PINN-enabled' if self.pinn_enabled else 'Standard'}")
        logger.info(f"  Logs: {self.output_dir / 'training_logs'}")
        
        return self.training_monitor
    
    def prepare_data_and_environment(self) -> Env:
        """
        Prepare data and create training environment.
        
        Returns:
            Configured trading environment
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING DATA AND ENVIRONMENT")
        logger.info("="*70)
        
        # Load data
        logger.info(f"Loading data for {self.assets}...")
        df = load_and_prepare_data(
            tickers=self.assets,
            start_date=self.start_date,
            end_date=self.test_start_date,
        )
        
        logger.info(f"✓ Loaded {len(df)} trading days")
        
        # Create environment
        logger.info("Creating trading environment...")
        self.env = StockTradingEnv(
            df=df,
            stock_dim=len(self.assets),
            hmax=100,
            initial_amount=100000,
            transaction_cost_pct=0.001,
            reward_scaling=1e-4,
            render_mode=None,
        )
        
        logger.info(f"✓ Environment created")
        logger.info(f"  State space: {self.env.observation_space}")
        logger.info(f"  Action space: {self.env.action_space}")
        
        return self.env
    
    def train_agent_with_monitoring(self):
        """
        Phase 1: Train agent with convergence monitoring.
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: TRAINING DRL AGENT")
        logger.info("="*70)
        
        # Create agent (A2C recommended for composite rewards)
        logger.info(f"Initializing A2C agent (steps={self.episodes * 252})...")
        self.agent = A2C(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=7e-4,
            n_steps=2048,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(self.output_dir / "training_logs" / "tensorboard"),
        )
        
        logger.info("✓ Agent created")
        
        # Training loop with monitoring
        total_steps = self.episodes * 252
        logger.info(f"Training for {self.episodes} episodes ({total_steps} steps)...")
        
        for episode in range(self.episodes):
            self.agent.learn(total_timesteps=252, reset_num_timesteps=False)
            
            # Record metrics
            self.training_monitor.record_episode(
                episode=episode,
                reward=self.env.episode_return if hasattr(self.env, 'episode_return') else 0,
                length=252,
                trades=self.env.trades_count if hasattr(self.env, 'trades_count') else 0,
            )
            
            if (episode + 1) % 2 == 0:
                logger.info(f"  Episode {episode + 1}/{self.episodes} completed")
        
        logger.info("✓ Training completed")
        
        # Check convergence
        is_converged, issues = self.training_monitor.check_convergence()
        logger.info(f"Convergence Check: {'✓ PASSED' if is_converged else '✗ ISSUES DETECTED'}")
        if not is_converged:
            for issue, msg in issues.items():
                logger.warning(f"  {issue}: {msg}")
        
        # Print summary
        self.training_monitor.print_summary()
        self.training_monitor.export_to_csv()
        
        return self.agent
    
    def prepare_testing_data(self) -> pd.DataFrame:
        """
        Prepare test/backtest data.
        
        Returns:
            Test data DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: PREPARING TEST DATA")
        logger.info("="*70)
        
        logger.info(f"Loading test data...")
        test_df = load_and_prepare_data(
            tickers=self.assets,
            start_date=self.test_start_date,
            end_date="2025-01-01",
        )
        
        logger.info(f"✓ Loaded {len(test_df)} test trading days")
        
        return test_df
    
    def run_backtest(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 2: Run backtest on unseen data.
        
        Returns:
            Backtest results DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: RUNNING BACKTEST")
        logger.info("="*70)
        
        # Create test environment
        test_env = StockTradingEnv(
            df=test_df,
            stock_dim=len(self.assets),
            hmax=100,
            initial_amount=100000,
            transaction_cost_pct=0.001,
            render_mode=None,
        )
        
        logger.info("Running agent on test data...")
        
        # Run episode
        obs, _ = test_env.reset()
        results = {
            'date': [],
            'account_value': [],
            'action': [],
            'num_trades': [],
            'portfolio_value': [],
        }
        
        done = False
        step = 0
        while not done:
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            
            results['date'].append(test_df.index[step])
            results['account_value'].append(info.get('account_value', 0))
            results['action'].append(action)
            results['num_trades'].append(info.get('trades', 0))
            results['portfolio_value'].append(info.get('portfolio_value', 0))
            
            step += 1
        
        logger.info(f"✓ Backtest completed ({step} steps)")
        
        self.test_results = pd.DataFrame(results)
        self.test_results.set_index('date', inplace=True)
        
        return self.test_results
    
    def evaluate_performance(self) -> BacktestEvaluator:
        """
        Phase 2: Comprehensive performance evaluation with Pyfolio.
        
        Returns:
            Configured BacktestEvaluator
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: BACKTESTING EVALUATION (Pyfolio)")
        logger.info("="*70)
        
        logger.info("Setting up backtesting evaluator...")
        self.backtest_evaluator, ibov_ret, cdi_ret = setup_backtesting(
            results_df=self.test_results,
            initial_cash=100000.0,
        )
        
        logger.info("✓ Backtesting evaluator configured")
        logger.info(f"  Returns points: {len(self.backtest_evaluator.returns_series)}")
        logger.info(f"  Sharpe ratio: {self.backtest_evaluator.metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Sortino ratio: {self.backtest_evaluator.metrics.get('sortino_ratio', 0):.2f}")
        
        # Print comparison
        self.backtest_evaluator.print_summary()
        self.backtest_evaluator.save_metrics()
        
        return self.backtest_evaluator
    
    def generate_visualizations(self):
        """
        Phase 3: Generate custom DRL visualizations.
        """
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: DRLCUSTOM VISUALIZATIONS")
        logger.info("="*70)
        
        self.visualizer = CustomVisualizer(
            save_dir=str(self.output_dir / "visualizations")
        )
        
        logger.info("Generating visualization plots...")
        
        # Plot 1: Trading actions on price chart
        logger.info("  1. Plotting trading actions...")
        if 'account_value' in self.test_results.columns:
            self.visualizer.plot_trading_actions(
                prices=self.test_results['account_value'],
                actions=self.test_results['action'],
                account_values=self.test_results['account_value'],
                title=f"DRL Agent Trading Actions ({', '.join(self.assets)})",
                save_as="01_trading_actions.png",
            )
        
        # Plot 2: Returns distribution
        logger.info("  2. Plotting returns distribution...")
        self.visualizer.plot_returns_distribution(
            agent_returns=self.backtest_evaluator.returns_series,
            bins=50,
            save_as="02_returns_distribution.png",
        )
        
        # Plot 3: Cumulative returns
        logger.info("  3. Plotting cumulative returns...")
        self.visualizer.plot_cumulative_returns(
            agent_returns=self.backtest_evaluator.returns_series,
            cdi_returns=BenchmarkData.create_cdi_returns(
                self.test_results.index[0],
                self.test_results.index[-1],
            ),
            log_scale=True,
            save_as="03_cumulative_returns.png",
        )
        
        # Plot 4: Drawdown
        logger.info("  4. Plotting drawdown...")
        self.visualizer.plot_drawdown(
            returns=self.backtest_evaluator.returns_series,
            save_as="04_drawdown.png",
        )
        
        # Plot 5: Rolling metrics
        logger.info("  5. Plotting rolling metrics...")
        self.visualizer.plot_rolling_metrics(
            returns=self.backtest_evaluator.returns_series,
            window=30,
            metrics_list=['sharpe', 'sortino', 'volatility'],
            save_as="05_rolling_metrics.png",
        )
        
        # Plot 6: Metrics table
        logger.info("  6. Plotting metrics summary table...")
        self.visualizer.plot_performance_metrics_table(
            metrics=self.backtest_evaluator.metrics,
            save_as="06_metrics_table.png",
        )
        
        logger.info("✓ All visualizations generated")
        logger.info(f"  Saved to: {self.output_dir / 'visualizations'}")
    
    def run_complete_pipeline(self):
        """Run complete evaluation pipeline."""
        logger.info("\n\n")
        logger.info("█" * 70)
        logger.info("█ DRL AGENT COMPLETE EVALUATION PIPELINE".ljust(69) + "█")
        logger.info("█" * 70)
        
        try:
            # Phase 1: Training Monitoring
            self.setup_phase1_training_monitoring()
            self.prepare_data_and_environment()
            self.train_agent_with_monitoring()
            
            # Phase 2: Backtesting
            test_df = self.prepare_testing_data()
            self.run_backtest(test_df)
            self.evaluate_performance()
            
            # Phase 3: Visualizations
            self.generate_visualizations()
            
            logger.info("\n" + "█" * 70)
            logger.info("█ PIPELINE COMPLETED SUCCESSFULLY".ljust(69) + "█")
            logger.info("█" * 70)
            logger.info(f"\nResults saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DRL Agent Evaluation Pipeline'
    )
    parser.add_argument(
        '--assets',
        type=str,
        nargs='+',
        default=['VALE3', 'PETR4'],
        help='Asset tickers',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of training episodes',
    )
    parser.add_argument(
        '--pinn-enabled',
        action='store_true',
        help='Enable PINN features',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory',
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = EvaluationPipeline(
        assets=args.assets,
        episodes=args.episodes,
        pinn_enabled=args.pinn_enabled,
        output_dir=args.output_dir,
    )
    
    pipeline.run_complete_pipeline()


if __name__ == '__main__':
    main()

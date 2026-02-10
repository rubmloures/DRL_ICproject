"""
Complete Training & Evaluation Pipeline Example
===============================================

Demonstrates:
1. Training with automatic metric/model saving
2. Post-training visualization with pyfolio
3. Results analysis and reporting
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import PROJECT_ROOT, DATA_RAW, RESULTS, TRAINED_MODELS
from src.data import DataLoader, DataProcessor
from src.env import StockTradingEnv
from src.agents import PPOAgent, A2CAgent, DDPGAgent, EnsembleAgent
from src.evaluation.results_manager import ResultsManager
from src.evaluation.visualization import TradingVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(asset: str = "PETR4", split_ratio: float = 0.8):
    """Load and prepare data for training."""
    logger.info(f"Loading data for {asset}...")
    
    loader = DataLoader(data_path=DATA_RAW)
    df = loader.load_multiple_assets([asset])
    
    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()
    
    logger.info(f"✓ Loaded {len(df)} records")
    return df


def train_and_save_models(train_env, test_env):
    """Train agents and save results."""
    logger.info("\nTraining ensemble agents...")
    
    # Initialize agents
    ppo = PPOAgent(env=train_env, model_name="ppo")
    ddpg = DDPGAgent(env=train_env, model_name="ddpg")
    a2c = A2CAgent(env=train_env, model_name="a2c")
    
    # Train
    ppo.train(total_timesteps=50_000)
    ddpg.train(total_timesteps=50_000)
    a2c.train(total_timesteps=50_000)
    
    # Evaluate
    ppo_metrics = ppo.evaluate(num_episodes=5, env=test_env)
    ddpg_metrics = ddpg.evaluate(num_episodes=5, env=test_env)
    a2c_metrics = a2c.evaluate(num_episodes=5, env=test_env)
    
    # Create ensemble
    ensemble = EnsembleAgent(
        env=test_env,
        agents={'PPO': ppo, 'DDPG': ddpg, 'A2C': a2c}
    )
    ensemble_metrics = ensemble.evaluate(
        n_episodes=5,
        env=test_env
    )
    
    # Save all results
    logger.info("\nSaving results...")
    results_mgr = ResultsManager(RESULTS)
    
    results = {
        'ppo_metrics': ppo_metrics,
        'ddpg_metrics': ddpg_metrics,
        'a2c_metrics': a2c_metrics,
        'ensemble_metrics': ensemble_metrics,
    }
    
    # Save metrics
    results_mgr.save_metrics(results, 'training_results')
    
    # Save models
    for agent_name, agent in [('ppo', ppo), ('ddpg', ddpg), ('a2c', a2c)]:
        metric = results[f'{agent_name}_metrics']
        results_mgr.save_model(agent.model, agent_name, metric)
    
    results_mgr.save_model(ensemble.model, 'ensemble', ensemble_metrics)
    
    logger.info("✓ All results saved!")
    
    return results, {'ppo': ppo, 'ddpg': ddpg, 'a2c': a2c, 'ensemble': ensemble}


def generate_visualizations(df: pd.DataFrame, results: dict):
    """Generate and save visualizations."""
    logger.info("\nGenerating visualizations...")
    
    results_mgr = ResultsManager(RESULTS)
    visualizer = TradingVisualizer()
    
    # 1. Returns distribution
    if 'account_value' in df.columns:
        returns = df['account_value'].pct_change().dropna()
        
        fig = visualizer.plot_returns_distribution(returns)
        results_mgr.save_plot(fig, 'returns_distribution')
        
        # 2. Drawdown
        fig = visualizer.plot_drawdown(returns)
        results_mgr.save_plot(fig, 'drawdown_underwater')
    
    # 3. Metrics comparison
    metrics_dict = {
        'PPO': results['ppo_metrics'],
        'DDPG': results['ddpg_metrics'],
        'A2C': results['a2c_metrics'],
        'Ensemble': results['ensemble_metrics'],
    }
    
    fig = visualizer.plot_metrics_comparison(metrics_dict)
    results_mgr.save_plot(fig, 'metrics_comparison')
    
    logger.info("✓ Visualizations saved!")


def generate_report(results: dict):
    """Generate a summary report."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY REPORT")
    logger.info("="*70)
    
    logger.info("\nAgent Performance:")
    logger.info("-" * 70)
    
    for agent_name, metrics in results.items():
        if '_metrics' in agent_name:
            agent = agent_name.replace('_metrics', '').upper()
            logger.info(f"\n{agent}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.6f}")
    
    # Find best agent
    agent_rewards = {
        'PPO': results['ppo_metrics']['mean_reward'],
        'DDPG': results['ddpg_metrics']['mean_reward'],
        'A2C': results['a2c_metrics']['mean_reward'],
    }
    
    best_agent = max(agent_rewards, key=agent_rewards.get)
    logger.info(f"\n{'='*70}")
    logger.info(f"Best Individual Agent: {best_agent}")
    logger.info(f"Ensemble Mean Reward: {results['ensemble_metrics']['mean_reward']:.6f}")
    logger.info(f"{'='*70}")


def main():
    """Run complete training pipeline."""
    # Load data
    df = load_and_prepare_data("PETR4")
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Create environments
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=1,
        initial_amount=100_000,
        buy_cost_pct=0.0005,
        sell_cost_pct=0.0005,
    )
    
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=1,
        initial_amount=100_000,
        buy_cost_pct=0.0005,
        sell_cost_pct=0.0005,
    )
    
    # Train and save
    results, agents = train_and_save_models(train_env, test_env)
    
    # Generate visualizations
    try:
        generate_visualizations(df, results)
    except Exception as e:
        logger.warning(f"Could not generate some visualizations: {e}")
    
    # Generate report
    generate_report(results)
    
    logger.info("\n✓ Complete pipeline finished!")
    logger.info(f"Results saved to: {RESULTS}")


if __name__ == "__main__":
    main()

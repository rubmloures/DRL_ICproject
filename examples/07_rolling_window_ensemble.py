"""
Example 7: Rolling Window with Ensemble Strategy
=================================================

Demonstrates:
1. Loading and processing data
2. Using rolling window cross-validation (14s train / 4s test)
3. Training ensemble of PPO, DDPG, A2C on each window
4. Evaluating on test set and adjusting weights
5. Aggregating results across all windows
6. Comparing algorithms

This is the complete production-grade pipeline.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.data import DataLoader, DataProcessor, RollingWindowStrategy
from src.env import StockTradingEnv
from src.agents import PPOAgent, DDPGAgent, A2CAgent, EnsembleAgent
from src.backtest.metrics import PortfolioMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_process_data(data_dir: Path, assets: list, date_col: str = 'data'):
    """Load and preprocess data for all assets."""
    logger.info("="*70)
    logger.info("STEP 1: Data Loading and Preprocessing")
    logger.info("="*70)
    
    # Load data
    loader = DataLoader(data_path=data_dir)
    logger.info(f"Loading {len(assets)} assets: {assets}")
    
    df = loader.load_multiple_assets(assets=assets)
    logger.info(f"Loaded {len(df)} records")
    
    # Process data
    processor = DataProcessor()
    logger.info("Cleaning data...")
    df = processor.clean_data(df)
    
    logger.info("Adding technical indicators...")
    df = processor.add_technical_indicators(df)
    
    logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def create_rolling_windows(df: pd.DataFrame, train_weeks: int = 14, 
                          test_weeks: int = 4, overlap_weeks: int = 2,
                          purge_days: int = 5):
    """Create purged rolling window strategy with embargo gap."""
    logger.info("="*70)
    logger.info("STEP 2: Purged Rolling Window Configuration")
    logger.info("="*70)
    
    strategy = RollingWindowStrategy(
        df=df,
        train_weeks=train_weeks,
        test_weeks=test_weeks,
        overlap_weeks=overlap_weeks,
        purge_days=purge_days,
    )
    
    logger.info(f"Purged rolling windows created successfully (embargo={purge_days}d)")
    return strategy


def train_and_evaluate_window(train_df: pd.DataFrame, test_df: pd.DataFrame,
                             window_idx: int, date_range: dict) -> dict:
    """
    Train ensemble on one rolling window.
    
    Returns:
        Dictionary with window results
    """
    logger.info("="*70)
    logger.info(f"Window {window_idx}: {date_range['train_start'].date()} "
                f"→ {date_range['test_end'].date()}")
    logger.info("="*70)
    
    # Create environments
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=3,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0005,
        sell_cost_pct=0.0005,
    )
    
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=3,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0005,
        sell_cost_pct=0.0005,
    )
    
    # Create and train individual agents
    logger.info("\nTraining individual agents...")
    
    logger.info("  PPO training...")
    ppo = PPOAgent(env=train_env, learning_rate=3e-4, n_steps=2048, 
                   batch_size=64, verbose=0)
    ppo.train(total_timesteps=20_000)
    
    logger.info("  DDPG training...")
    ddpg = DDPGAgent(env=train_env, learning_rate=1e-3, buffer_size=50_000, 
                     verbose=0)
    ddpg.train(total_timesteps=20_000)
    
    logger.info("  A2C training...")
    a2c = A2CAgent(env=train_env, learning_rate=7e-4, n_steps=5, verbose=0)
    a2c.train(total_timesteps=20_000)
    
    # Evaluate individual agents on test set
    logger.info("\nEvaluating individual agents on test set...")
    ppo_metrics = ppo.evaluate(n_episodes=3, env=test_env)
    logger.info(f"  PPO Sharpe: {ppo_metrics['mean_reward']:.4f}")
    
    ddpg_metrics = ddpg.evaluate(n_episodes=3, env=test_env)
    logger.info(f"  DDPG Sharpe: {ddpg_metrics['mean_reward']:.4f}")
    
    a2c_metrics = a2c.evaluate(n_episodes=3, env=test_env)
    logger.info(f"  A2C Sharpe: {a2c_metrics['mean_reward']:.4f}")
    
    # Create ensemble
    logger.info("\nCreating and evaluating ensemble...")
    ensemble = EnsembleAgent(
        env=test_env,
        agents={
            'PPO': ppo,
            'DDPG': ddpg,
            'A2C': a2c
        },
        voting_strategy='weighted'
    )
    
    # Adjust weights based on individual performance
    ppo_sharpe = ppo_metrics['mean_reward']
    ddpg_sharpe = ddpg_metrics['mean_reward']
    a2c_sharpe = a2c_metrics['mean_reward']
    
    total_sharpe = ppo_sharpe + ddpg_sharpe + a2c_sharpe
    if total_sharpe > 0:
        ensemble.set_agent_weights({
            'PPO': ppo_sharpe / total_sharpe,
            'DDPG': ddpg_sharpe / total_sharpe,
            'A2C': a2c_sharpe / total_sharpe,
        })
    
    # Evaluate ensemble
    ensemble_metrics = ensemble.evaluate(n_episodes=3, env=test_env)
    logger.info(f"  Ensemble Sharpe: {ensemble_metrics['mean_reward']:.4f}")
    
    # Return results
    result = {
        'window_idx': window_idx,
        'date_range': date_range,
        'ppo_sharpe': ppo_metrics['mean_reward'],
        'ddpg_sharpe': ddpg_metrics['mean_reward'],
        'a2c_sharpe': a2c_metrics['mean_reward'],
        'ensemble_sharpe': ensemble_metrics['mean_reward'],
        'ppo_return': ppo_metrics.get('max_reward', 0),
        'ddpg_return': ddpg_metrics.get('max_reward', 0),
        'a2c_return': a2c_metrics.get('max_reward', 0),
        'ensemble_return': ensemble_metrics.get('max_reward', 0),
    }
    
    return result


def main():
    """Main pipeline: rolling window + ensemble."""
    
    logger.info("\n" + "="*70)
    logger.info("ROLLING WINDOW ENSEMBLE STRATEGY - COMPLETE PIPELINE")
    logger.info("="*70)
    logger.info(f"Start time: {datetime.now()}\n")
    
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "raw"
    ASSETS = ["PETR4", "VALE3", "BBAS3"]
    TRAIN_WEEKS = 14
    TEST_WEEKS = 4
    OVERLAP_WEEKS = 2
    
    try:
        # Step 1: Load and process data
        df = load_and_process_data(DATA_DIR, ASSETS)
        
        # Step 2: Create rolling windows
        rolling_strategy = create_rolling_windows(
            df,
            train_weeks=TRAIN_WEEKS,
            test_weeks=TEST_WEEKS,
            overlap_weeks=OVERLAP_WEEKS
        )
        
        # Step 3: Train and evaluate on each window
        logger.info("="*70)
        logger.info("STEP 3: Training and Evaluation on Rolling Windows")
        logger.info("="*70 + "\n")
        
        window_results = []
        
        for train_df, test_df, window_idx, date_range in rolling_strategy.generate_rolling_windows():
            result = train_and_evaluate_window(
                train_df, test_df, window_idx, date_range
            )
            window_results.append(result)
            
            # Limit to first 3 windows for demo
            if window_idx >= 2:
                logger.info("\n(Limiting to 3 windows for demo. Remove this check in production.)")
                break
        
        # Step 4: Aggregate results
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Aggregated Results Across Windows")
        logger.info("="*70 + "\n")
        
        aggregated = rolling_strategy.get_metrics_across_windows(window_results)
        
        # Summary statistics
        logger.info("Performance Summary:")
        logger.info("-"*70)
        logger.info(f"{'Algorithm':<15} {'Avg Sharpe':<15} {'Std Sharpe':<15} {'Min/Max':<20}")
        logger.info("-"*70)
        
        # PPO
        if 'avg_ppo_sharpe' in aggregated:
            logger.info(
                f"{'PPO':<15} "
                f"{aggregated['avg_ppo_sharpe']:>14.4f} "
                f"{aggregated['std_ppo_sharpe']:>14.4f} "
                f"{aggregated['min_ppo_sharpe']:>8.4f} / {aggregated['max_ppo_sharpe']:<8.4f}"
            )
        
        # DDPG
        if 'avg_ddpg_sharpe' in aggregated:
            logger.info(
                f"{'DDPG':<15} "
                f"{aggregated['avg_ddpg_sharpe']:>14.4f} "
                f"{aggregated['std_ddpg_sharpe']:>14.4f} "
                f"{aggregated['min_ddpg_sharpe']:>8.4f} / {aggregated['max_ddpg_sharpe']:<8.4f}"
            )
        
        # A2C
        if 'avg_a2c_sharpe' in aggregated:
            logger.info(
                f"{'A2C':<15} "
                f"{aggregated['avg_a2c_sharpe']:>14.4f} "
                f"{aggregated['std_a2c_sharpe']:>14.4f} "
                f"{aggregated['min_a2c_sharpe']:>8.4f} / {aggregated['max_a2c_sharpe']:<8.4f}"
            )
        
        # Ensemble
        if 'avg_ensemble_sharpe' in aggregated:
            logger.info(
                f"{'ENSEMBLE':<15} "
                f"{aggregated['avg_ensemble_sharpe']:>14.4f} "
                f"{aggregated['std_ensemble_sharpe']:>14.4f} "
                f"{aggregated['min_ensemble_sharpe']:>8.4f} / {aggregated['max_ensemble_sharpe']:<8.4f}"
            )
        
        logger.info("-"*70)
        logger.info(f"Total Windows Evaluated: {aggregated['total_windows']}")
        
        # Determine best algorithm
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Algorithm Comparison and Winner")
        logger.info("="*70 + "\n")
        
        algo_avg_sharpes = {}
        if 'avg_ppo_sharpe' in aggregated:
            algo_avg_sharpes['PPO'] = aggregated['avg_ppo_sharpe']
        if 'avg_ddpg_sharpe' in aggregated:
            algo_avg_sharpes['DDPG'] = aggregated['avg_ddpg_sharpe']
        if 'avg_a2c_sharpe' in aggregated:
            algo_avg_sharpes['A2C'] = aggregated['avg_a2c_sharpe']
        if 'avg_ensemble_sharpe' in aggregated:
            algo_avg_sharpes['ENSEMBLE'] = aggregated['avg_ensemble_sharpe']
        
        if algo_avg_sharpes:
            best_algo = max(algo_avg_sharpes, key=algo_avg_sharpes.get)
            best_sharpe = algo_avg_sharpes[best_algo]
            
            logger.info(f"🏆 Best Algorithm: {best_algo}")
            logger.info(f"   Average Sharpe: {best_sharpe:.4f}")
            logger.info(f"\n{'Algorithm':<15} {'Avg Sharpe':<15} {'Ranking':<10}")
            logger.info("-"*40)
            
            for i, (algo, sharpe) in enumerate(sorted(algo_avg_sharpes.items(), 
                                                       key=lambda x: x[1], reverse=True), 1):
                marker = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
                logger.info(f"{algo:<15} {sharpe:>14.4f} {marker:>8} #{i}")
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"End time: {datetime.now()}")
        logger.info("\nNext Steps:")
        logger.info("1. Deploy best algorithm to production")
        logger.info("2. Monitor performance on live data")
        logger.info("3. Re-train on newest data periodically")
        logger.info("4. Adjust ensemble weights based on recent performance")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

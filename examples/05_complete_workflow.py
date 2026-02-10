"""
Example 5: Complete End-to-End Workflow
=======================================

This example demonstrates:
1. Loading and processing real data
2. Creating trading environment
3. Training DRL agent with multiple algorithms
4. Evaluating performance
5. Backtesting on test set
6. Comparing results
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data import DataLoader, DataProcessor
from src.env import StockTradingEnvB3
from src.agents import PPOAgent, DDPGAgent
from src.core.constants import DEFAULT_PPO_PARAMS, DEFAULT_DDPG_PARAMS


def load_and_process_data():
    """Step 1: Complete data pipeline"""
    print("\n[Step 1] Data Pipeline")
    print("-" * 60)
    
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    loader = DataLoader(data_path=data_dir)
    processor = DataProcessor()
    
    # Load
    print("  Loading raw CSV data...")
    try:
        df = loader.load_multiple_assets(
            assets=["PETR4", "VALE3", "BBAS3"],
            start_date="2022-01-01",
            end_date="2024-12-31"
        )
        print(f"  ✓ Loaded {len(df)} records from 3 assets")
    except FileNotFoundError:
        print("  ✗ CSV files not found in data/raw/")
        return None, None
    
    # Process
    print("  Processing: clean, indicators, normalize...")
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)
    
    # Scale
    print("  Fitting scalers...")
    indicator_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'ATR_14', 'acao_close_ajustado']
    available_cols = [c for c in indicator_cols if c in df.columns]
    processor.fit_scaler(df, columns=available_cols, scaler_name="features")
    df = processor.transform(df, scaler_name="features")
    
    # Split
    print("  Splitting train/test (80/20)...")
    train_data, test_data = DataProcessor.split_data(df, train_ratio=0.8)
    
    print(f"  ✓ Final: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data


def create_environments(train_data, test_data):
    """Step 2: Create train and test environments"""
    print("\n[Step 2] Environment Setup")
    print("-" * 60)
    
    print("  Creating training environment...")
    train_env = StockTradingEnvB3(
        df=train_data,
        stock_dim=3,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0003,
        sell_cost_pct=0.0003,
    )
    print(f"  ✓ Train env: {len(train_data)} trading days")
    
    print("  Creating test environment...")
    test_env = StockTradingEnvB3(
        df=test_data,
        stock_dim=3,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0003,
        sell_cost_pct=0.0003,
    )
    print(f"  ✓ Test env: {len(test_data)} trading days")
    
    return train_env, test_env


def train_agents(train_env):
    """Step 3: Train multiple agents"""
    print("\n[Step 3] Agent Training")
    print("-" * 60)
    
    agents = {}
    
    # Train PPO
    print("  Training PPO (stable, recommended)...")
    ppo_agent = PPOAgent(
        env=train_env,
        **DEFAULT_PPO_PARAMS,
        learning_rate=3e-4,
        verbose=0
    )
    ppo_agent.train(total_timesteps=50_000, verbose=0)
    agents["PPO"] = ppo_agent
    print("  ✓ PPO trained (50K timesteps)")
    
    # Train DDPG
    print("  Training DDPG (sample efficient)...")
    ddpg_agent = DDPGAgent(
        env=train_env,
        **DEFAULT_DDPG_PARAMS,
        learning_rate=1e-3,
        verbose=0
    )
    ddpg_agent.train(total_timesteps=50_000, verbose=0)
    agents["DDPG"] = ddpg_agent
    print("  ✓ DDPG trained (50K timesteps)")
    
    return agents


def evaluate_agents(agents, test_env):
    """Step 4: Evaluate on test set"""
    print("\n[Step 4] Evaluation")
    print("-" * 60)
    
    results = {}
    
    print(f"\n{'Algorithm':<12} {'Reward':<12} {'Std':<12} {'Episodes':<12}")
    print("-" * 48)
    
    for algo_name, agent in agents.items():
        metrics = agent.evaluate(n_episodes=5, env=test_env)
        results[algo_name] = metrics
        
        print(f"{algo_name:<12} {metrics['mean_reward']:>11.4f} "
              f"{metrics['std_reward']:>11.4f} {5:>11}")
    
    return results


def backtest_portfolio(agents, test_env):
    """Step 5: Detailed backtest"""
    print("\n[Step 5] Backtesting")
    print("-" * 60)
    
    backtest_results = {}
    
    for algo_name, agent in agents.items():
        print(f"\n  {algo_name} Backtest:")
        
        obs, info = test_env.reset()
        total_return = 0
        episode_rewards = []
        
        for step in range(len(test_env.df) - 1):
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_rewards.append(reward)
            
            if terminated or truncated:
                break
        
        total_return = np.sum(episode_rewards)
        sharpe = np.mean(episode_rewards) / (np.std(episode_rewards) + 1e-6)
        max_dd = calculate_max_drawdown(info.get('portfolio_values', []))
        
        backtest_results[algo_name] = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_trades': len([r for r in episode_rewards if r != 0])
        }
        
        print(f"    Total Return: {total_return:.4f}")
        print(f"    Sharpe Ratio: {sharpe:.4f}")
        print(f"    Max Drawdown: {max_dd:.4f}")
    
    return backtest_results


def calculate_max_drawdown(values):
    """Calculate maximum drawdown"""
    if len(values) < 2:
        return 0
    
    values = np.array(values)
    running_max = np.maximum.accumulate(values)
    drawdown = (running_max - values) / running_max
    return np.max(drawdown) if len(drawdown) > 0 else 0


def compare_algorithms(eval_results, backtest_results):
    """Step 6: Compare results"""
    print("\n[Step 6] Performance Comparison")
    print("-" * 60)
    
    print("\nEvaluation Scores (Test Set):")
    print(f"{'Algorithm':<12} {'Reward':<12} {'Winner':<10}")
    print("-" * 34)
    
    eval_rewards = {algo: metrics['mean_reward'] 
                   for algo, metrics in eval_results.items()}
    best_eval = max(eval_rewards, key=eval_rewards.get)
    
    for algo, reward in eval_rewards.items():
        winner = "★ Best" if algo == best_eval else ""
        print(f"{algo:<12} {reward:>11.4f} {winner}")
    
    if backtest_results:
        print("\nBacktest Performance:")
        print(f"{'Algorithm':<12} {'Sharpe':<12} {'Max DD':<12}")
        print("-" * 36)
        
        for algo, results in backtest_results.items():
            print(f"{algo:<12} {results['sharpe']:>11.4f} "
                  f"{results['max_drawdown']:>11.4f}")


def main():
    print("\n" + "="*70)
    print("EXAMPLE 5: Complete End-to-End Workflow")
    print("="*70)
    
    print("\nWorkflow summary:")
    print("  1. Load and process real B3 data")
    print("  2. Create training and test environments")
    print("  3. Train PPO and DDPG agents")
    print("  4. Evaluate on test set")
    print("  5. Detailed backtesting")
    print("  6. Compare algorithm performance")
    
    try:
        # Step 1: Data
        train_data, test_data = load_and_process_data()
        if train_data is None:
            return
        
        # Step 2: Environments
        train_env, test_env = create_environments(train_data, test_data)
        
        # Step 3: Training
        agents = train_agents(train_env)
        
        # Step 4: Evaluation
        eval_results = evaluate_agents(agents, test_env)
        
        # Step 5: Backtesting
        backtest_results = backtest_portfolio(agents, test_env)
        
        # Step 6: Comparison
        compare_algorithms(eval_results, backtest_results)
        
    except Exception as e:
        print(f"\n✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "="*70)
    print("Summary: Complete Workflow Successful")
    print("="*70)
    print("\nFiles created/used:")
    print("  • data/processed/ (from data_pipeline example)")
    print("  • trained_models/ (agent checkpoints)")
    
    print("\nNext steps:")
    print("  1. Use Example 4 for hyperparameter optimization")
    print("  2. Implement risk management and position sizing")
    print("  3. Add live market data connection")
    print("  4. Deploy best model to production")
    
    print("\n" + "="*70)
    print("Example 5 Completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

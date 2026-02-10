"""
Quick Start Guide
=================

This is a minimal example showing how to use the complete system in just a few minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data import DataLoader, DataProcessor
from src.env import StockTradingEnvB3
from src.agents import PPOAgent


def quick_start():
    """Complete pipeline in 10 lines of code"""
    
    # 1. Load data
    loader = DataLoader(Path(__file__).parent.parent / "data" / "raw")
    df = loader.load_multiple_assets(["PETR4", "VALE3"], start_date="2023-01-01")
    
    # 2. Process data
    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)
    train_data, test_data = DataProcessor.split_data(df, train_ratio=0.8)
    
    # 3. Create environment
    env = StockTradingEnvB3(df=train_data, stock_dim=2, initial_amount=100_000)
    
    # 4. Train agent
    agent = PPOAgent(env=env, learning_rate=3e-4, n_steps=2048)
    agent.train(total_timesteps=10_000)
    
    # 5. Evaluate
    metrics = agent.evaluate(n_episodes=5, env=env)
    print(f"Mean Reward: {metrics['mean_reward']:.4f}")
    
    return agent


if __name__ == "__main__":
    print("DRL Stock Trading Agent - Quick Start")
    print("=" * 50)
    
    try:
        agent = quick_start()
        print("\n✓ Success! Agent trained and ready to use.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("  1. You have CSV files in data/raw/")
        print("  2. You've run examples/01_data_pipeline.py first")
        print("  3. All dependencies are installed (pip install -r requirements.txt)")

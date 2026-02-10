"""
Example 2: Trading Environment
==============================

This example demonstrates:
1. Creating a trading environment (StockTradingEnvB3)
2. Resetting the environment
3. Stepping through trading days
4. Monitoring portfolio state
5. Understanding rewards and costs
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.env import StockTradingEnvB3


def main():
    print("\n" + "="*60)
    print("EXAMPLE 2: Trading Environment")
    print("="*60)
    
    # Load processed data
    print("\n[Step 1] Loading processed data...")
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    if not processed_dir.exists():
        print("✗ Processed data not found. Run Example 1 first!")
        return
    
    try:
        df = pd.read_csv(processed_dir / "train_data.csv")
        print(f"✓ Loaded {len(df)} records")
    except FileNotFoundError:
        print("✗ train_data.csv not found. Run Example 1 first!")
        return
    
    # Create environment
    print("\n[Step 2] Creating StockTradingEnvB3...")
    env = StockTradingEnvB3(
        df=df,
        stock_dim=3,  # Number of stocks to trade
        hmax=100,     # Max shares per transaction
        initial_amount=100_000,  # $100k starting capital
        buy_cost_pct=0.0003,     # 0.03% transaction cost
        sell_cost_pct=0.0003,
        gamma=0.99,              # Discount factor
        turbulence_threshold=None,  # Optional market stress detection
        reward='sharpe_ratio'     # Reward based on Sharpe ratio
    )
    print(f"✓ Environment created with observation_space: {env.observation_space}")
    print(f"             action_space: {env.action_space}")
    
    # Reset environment
    print("\n[Step 3] Resetting environment...")
    obs, info = env.reset()
    print(f"✓ Initial observation shape: {obs.shape}")
    print(f"  Initial cash: ${info.get('cash', 'unknown'):.2f}")
    print(f"  Initial portfolio value: ${info.get('portfolio_value', 'unknown'):.2f}")
    
    # Run simulation for 30 days
    print("\n[Step 4] Simulating 30 trading days...")
    print("\n{:<6} {:<12} {:<12} {:<12} {:<10}".format(
        "Day", "Action", "Cash ($)", "Portfolio ($)", "Reward"
    ))
    print("-" * 60)
    
    total_reward = 0.0
    max_action_magnitude = 0.0
    trades_made = 0
    
    for day in range(30):
        # Random action (continuous control per stock)
        action = env.action_space.sample()  # Random [-1, +1] per stock
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        total_reward += reward
        max_action_magnitude = max(max_action_magnitude, np.max(np.abs(action)))
        if np.any(action != 0):
            trades_made += 1
        
        if day % 5 == 0:  # Print every 5 days
            print("{:<6} {:<12.4f} ${:<11.2f} ${:<11.2f} {:<10.6f}".format(
                day,
                np.mean(action),
                info.get('cash', 0),
                info.get('portfolio_value', 0),
                reward
            ))
        
        if done:
            print(f"Episode terminated at day {day}")
            break
    
    # Summary statistics
    print("\n[Summary] Environment Simulation Complete")
    print(f"  Days simulated: 30")
    print(f"  Total reward (cumulative): {total_reward:.6f}")
    print(f"  Average reward per day: {total_reward/30:.6f}")
    print(f"  Trades executed: {trades_made}")
    print(f"  Max action magnitude: {max_action_magnitude:.4f}")
    
    # Show final portfolio state
    print("\n[Final State]")
    print(f"  Cash: ${info.get('cash', 0):.2f}")
    print(f"  Portfolio Value: ${info.get('portfolio_value', 0):.2f}")
    print(f"  Holdings per stock: {info.get('holdings', [])}")
    
    # Environment details
    print("\n[Environment Details]")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Number of stocks: {env.stock_dim}")
    print(f"  Hmax (max shares/trade): {env.hmax}")
    print(f"  Transaction cost: {env.buy_cost_pct*100:.4f}%")
    print(f"  Reward function: Sharpe ratio")
    
    print("\n" + "="*60)
    print("Example 2 Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

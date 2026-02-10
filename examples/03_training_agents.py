"""
Example 3: Training DRL Agents
=============================

This example demonstrates:
1. Creating a trading environment
2. Training PPO agent (recommended)
3. Evaluating trained agent
4. Saving/loading models
5. Brief comparison with DDPG and A2C
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.env import StockTradingEnvB3
from src.agents import PPOAgent, DDPGAgent, A2CAgent


def create_environment():
    """Create training environment"""
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    
    if not (processed_dir / "train_data.csv").exists():
        print("✗ Processed data not found. Run Example 1 first!")
        return None
    
    df = pd.read_csv(processed_dir / "train_data.csv")
    
    env = StockTradingEnvB3(
        df=df,
        stock_dim=3,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0003,
        sell_cost_pct=0.0003,
    )
    return env


def train_ppo_agent():
    """Train PPO agent (recommended starting point)"""
    print("\n" + "="*60)
    print("Training PPO Agent (Recommended)")
    print("="*60)
    
    env = create_environment()
    if env is None:
        return
    
    print("\n[Step 1] Creating PPO Agent...")
    agent = PPOAgent(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1
    )
    print("✓ PPO Agent created")
    print("  Learning rate: 3e-4")
    print("  n_steps: 2048")
    print("  Gamma (discount): 0.99")
    print("  Clip range: 0.2")
    
    print("\n[Step 2] Training PPO Agent...")
    print("  (Training for 100,000 timesteps)")
    agent.train(
        total_timesteps=100_000,
        save_dir="trained_models/ppo/",
        eval_freq=10_000,
        n_eval_episodes=3
    )
    print("✓ PPO Agent trained")
    
    print("\n[Step 3] Evaluating PPO Agent...")
    metrics = agent.evaluate(
        n_episodes=5,
        env=env
    )
    print(f"✓ Evaluation complete")
    print(f"  Mean reward: {metrics['mean_reward']:.4f}")
    print(f"  Std reward: {metrics['std_reward']:.4f}")
    print(f"  Min reward: {metrics['min_reward']:.4f}")
    print(f"  Max reward: {metrics['max_reward']:.4f}")
    
    print("\n[Step 4] Saving PPO Agent...")
    agent.save("trained_models/ppo/best_model")
    print("✓ Saved to trained_models/ppo/best_model")
    
    print("\n[Step 5] Loading and testing...")
    loaded_agent = PPOAgent(env=env)
    loaded_agent.load("trained_models/ppo/best_model")
    
    # Quick prediction
    obs, _ = env.reset()
    action, _states = loaded_agent.predict(obs)
    print(f"✓ Loaded agent prediction shape: {action.shape}")
    print(f"  Sample action: {action}")
    
    return agent


def train_ddpg_agent():
    """Train DDPG agent (sample efficient, off-policy)"""
    print("\n" + "="*60)
    print("Training DDPG Agent (Off-policy, Sample Efficient)")
    print("="*60)
    
    env = create_environment()
    if env is None:
        return
    
    print("\n[Step 1] Creating DDPG Agent...")
    agent = DDPGAgent(
        env=env,
        learning_rate=1e-3,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        verbose=1
    )
    print("✓ DDPG Agent created")
    print("  Learning rate: 1e-3")
    print("  Buffer size: 100K")
    print("  Tau (soft update): 0.005")
    
    print("\n[Step 2] Training DDPG Agent...")
    print("  (Training for 50,000 timesteps)")
    agent.train(
        total_timesteps=50_000,
        save_dir="trained_models/ddpg/",
        eval_freq=10_000,
        n_eval_episodes=3
    )
    print("✓ DDPG Agent trained")
    
    print("\n[Step 3] Brief evaluation...")
    metrics = agent.evaluate(n_episodes=3, env=env)
    print(f"  Mean reward: {metrics['mean_reward']:.4f}")
    print(f"  Mean episode length: {metrics['mean_length']:.1f}")


def train_a2c_agent():
    """Train A2C agent (fast, parallel)"""
    print("\n" + "="*60)
    print("Training A2C Agent (Fast, Parallel)")
    print("="*60)
    
    env = create_environment()
    if env is None:
        return
    
    print("\n[Step 1] Creating A2C Agent...")
    agent = A2CAgent(
        env=env,
        learning_rate=7e-4,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        verbose=1
    )
    print("✓ A2C Agent created")
    print("  Learning rate: 7e-4")
    print("  n_steps: 5 (parallel)")
    print("  GAE lambda: 1.0")
    
    print("\n[Step 2] Training A2C Agent...")
    print("  (Training for 50,000 timesteps)")
    agent.train(
        total_timesteps=50_000,
        save_dir="trained_models/a2c/",
        eval_freq=10_000,
        n_eval_episodes=3
    )
    print("✓ A2C Agent trained")
    
    print("\n[Step 3] Brief evaluation...")
    metrics = agent.evaluate(n_episodes=3, env=env)
    print(f"  Mean reward: {metrics['mean_reward']:.4f}")


def main():
    print("\n" + "="*60)
    print("EXAMPLE 3: Training DRL Agents")
    print("="*60)
    
    print("\n[Overview] 3 Algorithm Strategies:")
    print("  • PPO (Proximal Policy Optimization)")
    print("    → Stable, recommended for beginners, on-policy")
    print("  • DDPG (Deep Deterministic Policy Gradient)")
    print("    → Off-policy, sample efficient, requires tuning")
    print("  • A2C (Advantage Actor-Critic)")
    print("    → Fast, parallel, good for distributed training")
    
    # Train all three agents
    try:
        train_ppo_agent()
        train_ddpg_agent()
        train_a2c_agent()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print("  Make sure you've run Example 1 to generate processed data")
        return
    
    # Summary
    print("\n" + "="*60)
    print("Summary: Agent Training Complete")
    print("="*60)
    print("\nModels saved to:")
    print("  • trained_models/ppo/")
    print("  • trained_models/ddpg/")
    print("  • trained_models/a2c/")
    
    print("\nNext steps:")
    print("  1. Use Example 4 for hyperparameter optimization")
    print("  2. Test agents on validation/test data")
    print("  3. Fine-tune learning rates and network architectures")
    print("  4. Deploy best model for live trading")
    
    print("\n" + "="*60)
    print("Example 3 Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

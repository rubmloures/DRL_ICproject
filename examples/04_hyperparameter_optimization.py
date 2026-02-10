"""
Example 4: Hyperparameter Optimization with Optuna
===================================================

This example demonstrates:
1. Setting up Optuna hyperparameter optimizer
2. Configuring search space for PPO
3. Running Bayesian optimization
4. Analyzing results
5. Training final model with best parameters
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.env import StockTradingEnvB3
from src.optimization import HyperparameterOptimizer
from src.agents import PPOAgent


def create_environment_fn():
    """Factory function to create environments for optimization"""
    
    def _make_env():
        processed_dir = Path(__file__).parent.parent / "data" / "processed"
        
        if not (processed_dir / "train_data.csv").exists():
            return None
        
        df = pd.read_csv(processed_dir / "train_data.csv")
        
        # Use only first N days for faster optimization cycles
        df = df.iloc[:252]  # ~1 year of trading days
        
        env = StockTradingEnvB3(
            df=df,
            stock_dim=3,
            hmax=100,
            initial_amount=100_000,
            buy_cost_pct=0.0003,
            sell_cost_pct=0.0003,
        )
        return env
    
    return _make_env


def optimize_ppo_hyperparameters():
    """Optimize PPO hyperparameters using Optuna"""
    print("\n" + "="*60)
    print("Hyperparameter Optimization: PPO")
    print("="*60)
    
    print("\n[Step 1] Creating Optimizer...")
    optimizer = HyperparameterOptimizer(
        agent_type="PPO",
        env_fn=create_environment_fn(),
        n_jobs=1  # Use 1 for single process, -1 for all cores
    )
    print("✓ HyperparameterOptimizer created")
    print("  Algorithm: PPO (Proximal Policy Optimization)")
    print("  Sampler: Tree Parzen Estimator (Bayesian)")
    print("  Pruner: Median (early stopping)")
    
    print("\n[Step 2] Configuring search space...")
    print("  Search parameters:")
    print("    • learning_rate: [1e-5, 1e-3]")
    print("    • n_steps: [512, 4096]")
    print("    • batch_size: [32, 256]")
    print("    • n_epochs: [5, 20]")
    print("    • gamma: [0.95, 0.9999]")
    print("    • gae_lambda: [0.8, 1.0]")
    print("    • clip_range: [0.1, 0.4]")
    
    print("\n[Step 3] Running optimization...")
    print("  (10 trials with 10,000 timesteps each)")
    print("  This may take 5-10 minutes...\n")
    
    try:
        results = optimizer.optimize(
            n_trials=10,
            timeout=600  # 10 minute max
        )
        
        print("\n✓ Optimization complete")
        
    except Exception as e:
        print(f"\n✗ Optimization failed: {e}")
        print("  Make sure you've run Example 1 to generate processed data")
        return None
    
    # Display results
    print("\n[Results] Best Configuration Found")
    print("-" * 60)
    
    best_params = results.get("best_params", {})
    best_value = results.get("best_value", 0)
    n_trials = results.get("n_trials_completed", 0)
    
    print(f"  Best objective value (Sharpe ratio): {best_value:.6f}")
    print(f"  Trials completed: {n_trials}")
    print("\n  Best parameters:")
    for param_name, param_value in best_params.items():
        if isinstance(param_value, float):
            print(f"    • {param_name}: {param_value:.6f}")
        else:
            print(f"    • {param_name}: {param_value}")
    
    return best_params


def train_with_best_params(best_params):
    """Train final agent with optimized hyperparameters"""
    if best_params is None:
        print("\n✗ Skipping training: No best parameters found")
        return
    
    print("\n" + "="*60)
    print("Training Final Model with Optimized Parameters")
    print("="*60)
    
    # Create environment
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    
    if not (processed_dir / "train_data.csv").exists():
        print("✗ Processed data not found")
        return
    
    df = pd.read_csv(processed_dir / "train_data.csv")
    
    env = StockTradingEnvB3(
        df=df,
        stock_dim=3,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0003,
        sell_cost_pct=0.0003,
    )
    
    print("\n[Step 1] Creating agent with optimized parameters...")
    agent = PPOAgent(
        env=env,
        learning_rate=best_params.get("learning_rate", 3e-4),
        n_steps=best_params.get("n_steps", 2048),
        batch_size=best_params.get("batch_size", 64),
        n_epochs=best_params.get("n_epochs", 10),
        gamma=best_params.get("gamma", 0.99),
        gae_lambda=best_params.get("gae_lambda", 0.95),
        clip_range=best_params.get("clip_range", 0.2),
        verbose=1
    )
    print("✓ Agent created with optimized parameters")
    
    print("\n[Step 2] Training on full dataset...")
    print("  (500,000 timesteps for convergence)")
    agent.train(
        total_timesteps=500_000,
        save_dir="trained_models/ppo_optimized/",
        eval_freq=50_000,
        n_eval_episodes=5
    )
    print("✓ Training complete")
    
    print("\n[Step 3] Final evaluation...")
    metrics = agent.evaluate(n_episodes=10, env=env)
    
    print(f"✓ Final metrics:")
    print(f"  Mean reward: {metrics['mean_reward']:.4f}")
    print(f"  Std reward: {metrics['std_reward']:.4f}")
    print(f"  Min reward: {metrics['min_reward']:.4f}")
    print(f"  Max reward: {metrics['max_reward']:.4f}")
    print(f"  Mean episode length: {metrics['mean_length']:.1f}")
    
    # Save final model
    agent.save("trained_models/ppo_optimized/final_model")
    print(f"\n✓ Saved final model to trained_models/ppo_optimized/final_model")


def compare_algorithms():
    """Show comparison of optimization for different algorithms"""
    print("\n" + "="*60)
    print("Algorithm Comparison")
    print("="*60)
    
    algorithms = {
        "PPO": {
            "stability": "⭐⭐⭐",
            "sample_efficiency": "⭐⭐⭐",
            "speed": "⭐⭐",
            "recommended": "✓ Recommended for beginners"
        },
        "DDPG": {
            "stability": "⭐⭐",
            "sample_efficiency": "⭐⭐⭐⭐",
            "speed": "⭐⭐⭐",
            "recommended": "For continuous control experts"
        },
        "A2C": {
            "stability": "⭐⭐",
            "sample_efficiency": "⭐⭐",
            "speed": "⭐⭐⭐⭐",
            "recommended": "For parallel distributed training"
        }
    }
    
    for algo, props in algorithms.items():
        print(f"\n{algo}:")
        print(f"  Stability: {props['stability']}")
        print(f"  Sample Efficiency: {props['sample_efficiency']}")
        print(f"  Training Speed: {props['speed']}")
        print(f"  {props['recommended']}")


def main():
    print("\n" + "="*60)
    print("EXAMPLE 4: Hyperparameter Optimization")
    print("="*60)
    
    print("\n[Overview] Bayesian Optimization with Optuna")
    print("  Objective: Find hyperparameters that maximize Sharpe ratio")
    print("  Method: Tree Parzen Estimator (TPE)")
    print("  Efficiency: Median pruner stops unpromising trials early")
    
    # Compare algorithms
    compare_algorithms()
    
    # Run optimization
    print("\n" + "="*60)
    print("Starting Optimization Process")
    print("="*60)
    
    best_params = optimize_ppo_hyperparameters()
    
    if best_params:
        # Train with best parameters
        train_with_best_params(best_params)
    
    # Final summary
    print("\n" + "="*60)
    print("Summary: Optimization Complete")
    print("="*60)
    print("\nWorkflow:")
    print("  1. ✓ Defined search space (learning rate, n_steps, etc.)")
    print("  2. ✓ Ran Bayesian optimization (10 trials)")
    print("  3. ✓ Found best hyperparameters")
    print("  4. ✓ Trained final model on full dataset")
    print("\nNext steps:")
    print("  • Use trained_models/ppo_optimized/final_model for backtesting")
    print("  • Compare with models from Example 3")
    print("  • Deploy best model to live trading")
    
    print("\n" + "="*60)
    print("Example 4 Completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

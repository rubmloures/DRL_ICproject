"""
Example 6: Ensemble Strategy
=============================

Demonstrates:
1. Creating individual agents (PPO, DDPG, A2C)
2. Combining predictions using ensemble voting
3. Setting weights based on performance
4. Evaluating ensemble vs individual agents
5. Analyzing voting patterns

Ensemble voting strategies:
- 'mean': Simple average of actions
- 'weighted': Actions weighted by agent performance (Sharpe ratio)
- 'majority': Discrete voting (buy/hold/sell)
- 'best': Always use best performing agent
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import numpy as np

from src.data import DataLoader, DataProcessor
from src.env import StockTradingEnv
from src.agents import PPOAgent, DDPGAgent, A2CAgent, EnsembleAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_process_data():
    """Load and preprocess data."""
    print("\n" + "="*60)
    print("Step 1: Data Loading and Processing")
    print("="*60)
    
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    loader = DataLoader(data_path=data_dir)
    
    try:
        df = loader.load_multiple_assets(
            assets=["PETR4", "VALE3"],
            start_date="2023-01-01",
            end_date="2024-12-31"
        )
        print(f"✓ Loaded {len(df)} records")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        print("  Make sure you have data in data/raw/")
        return None
    
    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)
    
    print(f"✓ Processed: {len(df)} records with {len(df.columns)} columns")
    
    return df


def create_environments(df):
    """Create train and test environments."""
    print("\n" + "="*60)
    print("Step 2: Environment Setup")
    print("="*60)
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=2,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0005,
        sell_cost_pct=0.0005,
    )
    
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=2,
        hmax=100,
        initial_amount=100_000,
        buy_cost_pct=0.0005,
        sell_cost_pct=0.0005,
    )
    
    print(f"✓ Training environment: {len(train_df)} days")
    print(f"✓ Testing environment: {len(test_df)} days")
    
    return train_env, test_env


def train_individual_agents(train_env):
    """Train PPO, DDPG, a2C agents independently."""
    print("\n" + "="*60)
    print("Step 3: Training Individual Agents")
    print("="*60)
    
    agents = {}
    
    # Train PPO
    print("\nTraining PPO (Proximal Policy Optimization)...")
    ppo = PPOAgent(
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=0
    )
    ppo.train(total_timesteps=50_000)
    agents['PPO'] = ppo
    print("✓ PPO trained (50K timesteps)")
    
    # Train DDPG
    print("\nTraining DDPG (Deep Deterministic Policy Gradient)...")
    ddpg = DDPGAgent(
        env=train_env,
        learning_rate=1e-3,
        buffer_size=100_000,
        verbose=0
    )
    ddpg.train(total_timesteps=50_000)
    agents['DDPG'] = ddpg
    print("✓ DDPG trained (50K timesteps)")
    
    # Train A2C
    print("\nTraining A2C (Advantage Actor-Critic)...")
    a2c = A2CAgent(
        env=train_env,
        learning_rate=7e-4,
        n_steps=5,
        verbose=0
    )
    a2c.train(total_timesteps=50_000)
    agents['A2C'] = a2c
    print("✓ A2C trained (50K timesteps)")
    
    return agents


def evaluate_individual_agents(agents, test_env):
    """Evaluate each agent individually."""
    print("\n" + "="*60)
    print("Step 4: Evaluating Individual Agents")
    print("="*60)
    
    individual_metrics = {}
    
    print(f"\n{'Algorithm':<12} {'Reward':<12} {'Std Dev':<12} {'Min/Max':<20}")
    print("-"*56)
    
    for agent_name, agent in agents.items():
        metrics = agent.evaluate(n_episodes=5, env=test_env)
        individual_metrics[agent_name] = metrics
        
        print(
            f"{agent_name:<12} "
            f"{metrics['mean_reward']:>11.4f} "
            f"{metrics['std_reward']:>11.4f} "
            f"{metrics['min_reward']:>8.4f} / {metrics['max_reward']:<8.4f}"
        )
    
    return individual_metrics


def create_and_evaluate_ensemble(agents, test_env, individual_metrics):
    """Create ensemble and evaluate different voting strategies."""
    print("\n" + "="*60)
    print("Step 5: Ensemble Configuration and Evaluation")
    print("="*60)
    
    ensemble_results = {}
    
    # Test each voting strategy
    voting_strategies = ['mean', 'weighted', 'majority', 'best']
    
    for strategy in voting_strategies:
        print(f"\nEnsemble with '{strategy}' voting strategy:")
        
        # Create ensemble
        ensemble = EnsembleAgent(
            env=test_env,
            agents=agents,
            voting_strategy=strategy
        )
        
        # For weighted strategy, set weights based on individual performance
        if strategy == 'weighted':
            sharpes = {name: metrics['mean_reward'] 
                      for name, metrics in individual_metrics.items()}
            total = sum(sharpes.values())
            if total > 0:
                weights = {name: s / total for name, s in sharpes.items()}
                ensemble.set_agent_weights(weights)
                print(f"  Weights: {weights}")
        
        # Evaluate ensemble
        metrics = ensemble.evaluate(n_episodes=5, env=test_env)
        ensemble_results[strategy] = metrics
        
        print(f"  → Mean Reward: {metrics['mean_reward']:.4f}")
        print(f"  → Std: {metrics['std_reward']:.4f}")
    
    return ensemble_results


def compare_results(individual_metrics, ensemble_results):
    """Compare individual agents vs ensemble strategies."""
    print("\n" + "="*60)
    print("Step 6: Performance Comparison")
    print("="*60)
    
    print("\nIndividual Agent Performance:")
    print("-"*50)
    print(f"{'Algorithm':<15} {'Sharpe Ratio':<15} {'Ranking'}")
    print("-"*50)
    
    individual_sharpes = {
        name: metrics['mean_reward'] 
        for name, metrics in individual_metrics.items()
    }
    
    for i, (algo, sharpe) in enumerate(
        sorted(individual_sharpes.items(), key=lambda x: x[1], reverse=True), 1
    ):
        print(f"{algo:<15} {sharpe:>14.4f} #{i}")
    
    best_individual = max(individual_sharpes.values())
    
    print("\nEnsemble Strategies Performance:")
    print("-"*50)
    print(f"{'Strategy':<15} {'Sharpe Ratio':<15} {'vs Best'}")
    print("-"*50)
    
    for strategy, metrics in ensemble_results.items():
        sharpe = metrics['mean_reward']
        improvement = ((sharpe - best_individual) / abs(best_individual) 
                      if best_individual != 0 else 0) * 100
        
        marker = "✓ BETTER" if sharpe > best_individual else "✗ Worse"
        
        print(f"{strategy:<15} {sharpe:>14.4f} {improvement:>+7.1f}% {marker}")
    
    # Overall winner
    best_ensemble = max(
        ensemble_results.items(), 
        key=lambda x: x[1]['mean_reward']
    )
    
    print("\n" + "="*60)
    print(f"🏆 Best Overall: {best_ensemble[0].upper()} Ensemble")
    print(f"   Sharpe Ratio: {best_ensemble[1]['mean_reward']:.4f}")
    print("="*60)


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("ENSEMBLE STRATEGY EXAMPLE")
    print("="*60)
    
    try:
        # Load data
        df = load_and_process_data()
        if df is None:
            return 1
        
        # Create environments
        train_env, test_env = create_environments(df)
        
        # Train individual agents
        agents = train_individual_agents(train_env)
        
        # Evaluate individual agents
        individual_metrics = evaluate_individual_agents(agents, test_env)
        
        # Create and evaluate ensemble
        ensemble_results = create_and_evaluate_ensemble(
            agents, test_env, individual_metrics
        )
        
        # Compare results
        compare_results(individual_metrics, ensemble_results)
        
        # Final notes
        print("\n" + "="*60)
        print("Summary and Next Steps")
        print("="*60)
        print("""
Key Insights:
1. Individual agents have different strengths
2. Ensemble voting combines strengths and reduces variance
3. Weighted voting (based on Sharpe) often performs best
4. Ensemble is more robust across different market conditions

Next Steps:
1. Use rolling window cross-validation (Example 7)
2. Deploy ensemble strategy to live trading
3. Monitor and re-weight agents based on recent performance
4. Add new agents (SAC, TD3) for further improvement
        """)
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

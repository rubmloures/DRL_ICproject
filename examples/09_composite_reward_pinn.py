"""
Example: PINN-Driven Dynamic Composite Reward for B3 Trading
=============================================================

Demonstrates:
1. PINN regime detection (stable/turbulent/shocked markets)
2. Dynamic reward weighting based on market regime
3. Integration with StockTradingEnv
4. Training with A2C (recommended agent for this reward)

Key Features:
- Excess return over Selic/CDI (Brazilian risk-free rate)
- Downside risk penalty (Sortino-like)
- Alpha over Ibovespa benchmark
- Transaction cost penalties (B3 emolumentos + corretagem)
- Regime-adaptive reward weights via PINN

Usage:
    python examples/09_composite_reward_pinn.py --assets VALE3 PETR4 --episodes 10
    python examples/09_composite_reward_pinn.py --reward-type composite --pinn-enabled
"""

import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    DATA_RAW, DATA_PROCESSED, TRAINED_MODELS, RESULTS,
    INITIAL_CAPITAL, TRANSACTION_COST, PRIMARY_ASSETS,
    PINN_CHECKPOINT_PATH, PINN_DATA_STATS_PATH,
)
from config.hyperparameters import (
    A2C_PARAMS, COMPOSITE_REWARD_CONFIG, PINN_FEATURE_WEIGHTS
)

from src.env import StockTradingEnv
from src.agents import A2CAgent, EnsembleAgent
from src.data import DataLoader, DataProcessor
from src.reward import CompositeRewardCalculator, RegimeDetector
from src.reward.composite_reward import B3CompositeRewardWrapper
from src.pinn.inference_wrapper import PINNInferenceEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(assets: list) -> pd.DataFrame:
    """Load and prepare data for trading."""
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Data Loading and Preprocessing")
    logger.info("="*70)
    
    loader = DataLoader(data_path=DATA_RAW)
    df = loader.load_multiple_assets(assets)
    logger.info(f"✓ Loaded {len(df)} records from {len(assets)} assets")
    
    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)
    logger.info(f"✓ Processed: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def initialize_reward_function(
    use_composite: bool = True,
    pinn_engine=None,
) -> B3CompositeRewardWrapper:
    """Initialize composite reward calculator."""
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Initialize Composite Reward Function")
    logger.info("="*70)
    
    if not use_composite:
        logger.info("Using default reward function (Sharpe-based)")
        return None
    
    # Create regime detector
    regime_detector = RegimeDetector(
        threshold_stable=COMPOSITE_REWARD_CONFIG['regime_thresholds']['stable_vol'],
        threshold_xi_high=COMPOSITE_REWARD_CONFIG['regime_thresholds']['high_vol_of_vol'],
        threshold_rho_negative=COMPOSITE_REWARD_CONFIG['regime_thresholds']['negative_correlation'],
    )
    logger.info("✓ RegimeDetector initialized")
    
    # Create reward calculator
    reward_calc = CompositeRewardCalculator(
        regime_detector=regime_detector,
        base_weights=COMPOSITE_REWARD_CONFIG['base_weights'],
        selic_daily_rate=COMPOSITE_REWARD_CONFIG['selic_daily_rate'],
        transaction_cost_pct=COMPOSITE_REWARD_CONFIG['transaction_cost_pct'],
        emolumentos_pct=COMPOSITE_REWARD_CONFIG['emolumentos_pct'],
        window_size=COMPOSITE_REWARD_CONFIG['window_size'],
    )
    logger.info("✓ CompositeRewardCalculator initialized")
    
    # Wrap with B3Wrapper
    wrapper = B3CompositeRewardWrapper(
        reward_calculator=reward_calc,
        pinn_engine=pinn_engine,
    )
    logger.info("✓ B3CompositeRewardWrapper created")
    
    return wrapper


def train_with_composite_reward(
    df: pd.DataFrame,
    assets: list,
    reward_wrapper: B3CompositeRewardWrapper,
    episodes: int = 5,
    pinn_engine=None,
) -> dict:
    """Train agent with composite reward function."""
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Train A2C Agent with Composite Reward")
    logger.info("="*70)
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train set: {len(train_df)} days | Test set: {len(test_df)} days")
    
    # Create environments
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=len(assets),
        initial_amount=INITIAL_CAPITAL,
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
        pinn_engine=pinn_engine,
        include_pinn_features=(pinn_engine is not None),
    )
    
    # Set composite reward
    if reward_wrapper is not None:
        train_env.set_reward_function(reward_wrapper)
        logger.info("✓ Composite reward function attached to environment")
    
    # Train A2C (recommended for this reward function)
    logger.info("\nTraining A2C agent...")
    agent = A2CAgent(env=train_env, **A2C_PARAMS, verbose=1)
    agent.train(total_timesteps=50_000)
    logger.info("✓ A2C training complete")
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=len(assets),
        initial_amount=INITIAL_CAPITAL,
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
        pinn_engine=pinn_engine,
        include_pinn_features=(pinn_engine is not None),
    )
    
    test_results = []
    for episode in range(3):
        obs, _ = test_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
            episode_reward += reward
        
        final_value = test_env.portfolio_value
        test_results.append({
            'episode': episode,
            'final_value': final_value,
            'return': (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            'reward_sum': episode_reward,
        })
        
        logger.info(
            f"Test Episode {episode}: Return {test_results[-1]['return']*100:+.2f}%, "
            f"Final Value ${final_value:,.0f}"
        )
    
    # Get reward metrics
    if reward_wrapper is not None:
        metrics = reward_wrapper.reward_calculator.get_metrics()
        logger.info("\n" + "-"*70)
        logger.info("Reward Metrics:")
        logger.info("-"*70)
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.4f}")
        logger.info(f"Alpha: {metrics.get('alpha', 0):.4f}")
        logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Current Regime: {metrics.get('regime', 'unknown')}")
    
    return {
        'agent': agent,
        'test_results': test_results,
        'reward_wrapper': reward_wrapper,
    }


def demonstrate_regime_switching(reward_wrapper: B3CompositeRewardWrapper):
    """Demonstrate dynamic weight adjustment with regime changes."""
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Demonstrate Regime-Adaptive Reward Weighting")
    logger.info("="*70)
    
    if reward_wrapper is None:
        logger.info("Composite reward not enabled, skipping demo")
        return
    
    regime_detector = reward_wrapper.reward_calculator.regime_detector
    
    # Simulate different market regimes
    regimes_to_test = [
        {
            'name': 'Stable Trending Market',
            'heston': {'nu': 0.08, 'theta': 0.1, 'kappa': 0.2, 'xi': 0.15, 'rho': -0.2},
            'vol': 0.008,
        },
        {
            'name': 'Elevated Volatility',
            'heston': {'nu': 0.18, 'theta': 0.2, 'kappa': 0.3, 'xi': 0.35, 'rho': -0.4},
            'vol': 0.018,
        },
        {
            'name': 'Turbulent / Shock',
            'heston': {'nu': 0.25, 'theta': 0.3, 'kappa': 0.4, 'xi': 0.5, 'rho': -0.7},
            'vol': 0.030,
        },
    ]
    
    logger.info("\nTesting regime detection and weight adjustment:\n")
    
    for regime_test in regimes_to_test:
        detected_regime, confidence = regime_detector.detect_regime(
            heston_params=regime_test['heston'],
            current_volatility=regime_test['vol'],
        )
        
        weights = regime_detector.get_regime_weights(
            regime=detected_regime,
            base_weights=reward_wrapper.reward_calculator.base_weights,
        )
        
        logger.info(f"Scenario: {regime_test['name']}")
        logger.info(f"  Heston Params: ν={regime_test['heston']['nu']:.3f}, "
                   f"ξ={regime_test['heston']['xi']:.3f}, "
                   f"ρ={regime_test['heston']['rho']:.3f}")
        logger.info(f"  Detected Regime: {detected_regime.value}")
        logger.info(f"  Confidence: {confidence}")
        logger.info(f"  Reward Weights:")
        logger.info(f"    - Excess Return:   {weights['excess_return']:.2f}")
        logger.info(f"    - Downside Risk:   {weights['downside_risk']:.2f}")
        logger.info(f"    - Alpha Return:    {weights['alpha_return']:.2f}")
        logger.info(f"    - Transaction Cost: {weights['transaction_cost']:.2f}")
        logger.info()


def main():
    """Main workflow."""
    parser = argparse.ArgumentParser(
        description='PINN-Driven Dynamic Composite Reward for B3 Trading'
    )
    parser.add_argument('--assets', nargs='+', default=['VALE3', 'PETR4'],
                       help='Assets to trade')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of training episodes')
    parser.add_argument('--reward-type', choices=['default', 'composite'],
                       default='composite',
                       help='Reward function type')
    parser.add_argument('--pinn-enabled', action='store_true',
                       help='Use PINN for regime detection')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*70)
    logger.info("PINN-Driven Dynamic Composite Reward for B3 Trading")
    logger.info("="*70)
    logger.info(f"Assets: {args.assets}")
    logger.info(f"Reward Type: {args.reward_type}")
    logger.info(f"PINN Regime Detection: {'Enabled' if args.pinn_enabled else 'Disabled'}")
    
    try:
        # 1. Load data
        df = load_and_prepare_data(args.assets)
        
        # 2. Initialize PINN if enabled
        pinn_engine = None
        if args.pinn_enabled:
            logger.info("\nInitializing PINN engine...")
            try:
                pinn_engine = PINNInferenceEngine(
                    checkpoint_path=str(PINN_CHECKPOINT_PATH),
                    data_stats_path=str(PINN_DATA_STATS_PATH),
                )
                logger.info("✓ PINN engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize PINN: {e}")
                logger.warning("Proceeding without PINN regime detection")
        
        # 3. Initialize reward function
        use_composite = (args.reward_type == 'composite')
        reward_wrapper = initialize_reward_function(
            use_composite=use_composite,
            pinn_engine=pinn_engine,
        )
        
        # 4. Train agent
        results = train_with_composite_reward(
            df=df,
            assets=args.assets,
            reward_wrapper=reward_wrapper,
            episodes=args.episodes,
            pinn_engine=pinn_engine,
        )
        
        # 5. Demonstrate regime switching
        demonstrate_regime_switching(reward_wrapper)
        
        # 6. Save results
        logger.info("\n" + "="*70)
        logger.info("Results")
        logger.info("="*70)
        
        test_results_df = pd.DataFrame(results['test_results'])
        logger.info(f"\nAvg Test Return: {test_results_df['return'].mean()*100:+.2f}%")
        logger.info(f"Test Std Dev: {test_results_df['return'].std()*100:.2f}%")
        
        if args.save_models:
            save_dir = TRAINED_MODELS / "example_09_composite_reward"
            save_dir.mkdir(parents=True, exist_ok=True)
            results['agent'].save(str(save_dir / "agent.pkl"))
            logger.info(f"\n✓ Models saved to {save_dir}")
        
        logger.info("\n" + "="*70)
        logger.info("Example Complete!")
        logger.info("="*70)
        
        return 0
    
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

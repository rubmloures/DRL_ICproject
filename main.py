"""
DRL Stock Trading Agent - Main Pipeline
========================================

Refactored main orchestration script supporting:
1. Generic data loading (multi-format CSV)
2. Generic trading environment (any market)
3. Ensemble agent training (PPO + DDPG + A2C)
4. Rolling window cross-validation
5. Comprehensive backtesting

Usage:
    python main.py --mode rolling-ensemble
    python main.py --mode simple-pipeline
    python main.py --help
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
import logging
import sys
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent))

# ⭐ REPRODUCIBILITY: Set seeds FIRST before any other imports
from src.core.reproducibility import set_all_seeds, assert_reproducible
from src.core.validation import validate_input_safety

from config.config import (
    PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, TRAINED_MODELS, RESULTS,
    PRIMARY_ASSETS, INITIAL_CAPITAL, TRANSACTION_COST, SLIPPAGE,
    ROLLING_WINDOW_CONFIG, ENSEMBLE_CONFIG,
    PINN_ENABLED, PINN_CHECKPOINT_PATH, PINN_DATA_STATS_PATH,
)
from config.hyperparameters import (
    PPO_PARAMS, DDPG_PARAMS, A2C_PARAMS,
    PINN_FEATURE_WEIGHTS, AB_TESTING_CONFIG, TRAINING_CONFIG
)

from src.agents.training_utils import (
    CheckpointManager, safe_train_with_timeout, TrainingTimeoutError
)

from src.data import DataLoader, DataProcessor, RollingWindowStrategy
from src.env import StockTradingEnv
from src.agents import PPOAgent, DDPGAgent, A2CAgent, EnsembleAgent
from src.optimization.hyperparameter_optimizer import HyperparameterOptimizer
from src.evaluation.results_manager import ResultsManager
from src.evaluation.visualization import TradingVisualizer

# Optional PINN support
try:
    from src.pinn.inference_wrapper import PINNInferenceEngine
    from src.data.pinn_data_preprocessor import PINNDataPreprocessor
    from src.data.temporal_data_splitter import TemporalDataSplitter
    from src.tools.ab_testing import ABTestingFramework
    PINN_AVAILABLE = True
except ImportError as e:
    PINN_AVAILABLE = False
    logger.warning(f"PINN module unavailable: {e}. Proceeding without PINN features.")
    if PINN_ENABLED:  # Se config diz que precisa de PINN
        logger.error("PINN_ENABLED=True mas módulo indisponível. Abortando!")
        sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """Create necessary output directories."""
    for directory in [DATA_PROCESSED, TRAINED_MODELS, RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ {directory}")


def load_and_preprocess_data(assets: List[str] = None) -> pd.DataFrame:
    """
    Load raw data and preprocess.
    
    Parameters
    ----------
    assets : list, optional
        Assets to load (default: PRIMARY_ASSETS)
    
    Returns
    -------
    pd.DataFrame
        Processed data with technical indicators
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Data Loading and Preprocessing")
    logger.info("="*70)
    
    if assets is None:
        assets = PRIMARY_ASSETS
    
    # Load data
    logger.info(f"Loading {len(assets)} assets...")
    loader = DataLoader(data_path=DATA_RAW)
    
    try:
        df = loader.load_multiple_assets(assets)
        logger.info(f"✓ Loaded {len(df)} records from {len(assets)} assets")
    except FileNotFoundError as e:
        logger.error(f"✗ {e}")
        logger.error("Please ensure CSV files exist in data/raw/")
        raise
    
    # Preprocess
    logger.info("Preprocessing data...")
    processor = DataProcessor()
    df = processor.clean_data(df)
    df = processor.add_technical_indicators(df)
    
    # Fill NaN values from technical indicators (SMA, RSI, etc.)
    # Initial rows will have NaN because indicators need warmup
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].ffill().bfill()
    
    logger.info(f"✓ Processed: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def simple_pipeline(df: pd.DataFrame, assets: Optional[List[str]] = None) -> Dict:
    """
    Simple pipeline: load data, split, train, evaluate.
    
    Uses 80/20 train/test split (no rolling windows).
    
    Args:
        df: Preprocessed DataFrame with price data
        assets: List of asset tickers (used to determine stock_dim)
    """
    logger.info("\n" + "="*70)
    logger.info("SIMPLE PIPELINE: Fixed Train/Test Split")
    logger.info("="*70)
    
    # Determine number of stocks from data or assets list
    if assets is None:
        assets = PRIMARY_ASSETS
    stock_dim = len(assets)
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(train_df)} days | Test: {len(test_df)} days")
    
    # Create environments
    logger.info("\nCreating environments...")
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        initial_amount=INITIAL_CAPITAL,
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
    )
    
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        initial_amount=INITIAL_CAPITAL,
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
    )
    
    logger.info("✓ Environments created")
    
    # Setup checkpointing and timeouts
    checkpoint_dir = TRAINED_MODELS / "checkpoints"
    timeout_seconds = TRAINING_CONFIG.get("timeout_seconds", 600)
    
    # Train ensemble with timeout protection
    logger.info("\nTraining ensemble...")
    ppo = PPOAgent(env=train_env, **PPO_PARAMS)
    ddpg = DDPGAgent(env=train_env, **DDPG_PARAMS)
    a2c = A2CAgent(env=train_env, **A2C_PARAMS)
    
    # Create checkpoint managers
    ppo_checkpoint = CheckpointManager(checkpoint_dir, "ppo")
    ddpg_checkpoint = CheckpointManager(checkpoint_dir, "ddpg")
    a2c_checkpoint = CheckpointManager(checkpoint_dir, "a2c")
    
    # Training with timeouts
    logger.info("  Training PPO...")
    try:
        success = safe_train_with_timeout(
            ppo.model,
            total_timesteps=50_000,
            timeout_seconds=timeout_seconds,
            save_callback=lambda: ppo_checkpoint.save_checkpoint(ppo.model, 50_000),
            callback=None,
            progress_bar=True,
        )
        if not success:
            logger.warning("PPO training timeout - using available model")
    except Exception as e:
        logger.error(f"PPO training failed: {e}")
    
    logger.info("  Training DDPG...")
    try:
        success = safe_train_with_timeout(
            ddpg.model,
            total_timesteps=50_000,
            timeout_seconds=timeout_seconds,
            save_callback=lambda: ddpg_checkpoint.save_checkpoint(ddpg.model, 50_000),
            callback=None,
            progress_bar=True,
        )
        if not success:
            logger.warning("DDPG training timeout - using available model")
    except Exception as e:
        logger.error(f"DDPG training failed: {e}")
    
    logger.info("  Training A2C...")
    try:
        success = safe_train_with_timeout(
            a2c.model,
            total_timesteps=50_000,
            timeout_seconds=timeout_seconds,
            save_callback=lambda: a2c_checkpoint.save_checkpoint(a2c.model, 50_000),
            callback=None,
            progress_bar=True,
        )
        if not success:
            logger.warning("A2C training timeout - using available model")
    except Exception as e:
        logger.error(f"A2C training failed: {e}")
    
    # Cleanup old checkpoints3
    ppo_checkpoint.cleanup_old_checkpoints()
    ddpg_checkpoint.cleanup_old_checkpoints()
    a2c_checkpoint.cleanup_old_checkpoints()
    
    # Evaluate
    logger.info("\nEvaluating on test set...")
    ppo_metrics = ppo.evaluate(num_episodes=5, env=test_env)
    ddpg_metrics = ddpg.evaluate(num_episodes=5, env=test_env)
    a2c_metrics = a2c.evaluate(num_episodes=5, env=test_env)
    
    # Create ensemble
    logger.info("\nCreating ensemble...")
    ensemble = EnsembleAgent(
        env=test_env,
        agents={'PPO': ppo, 'DDPG': ddpg, 'A2C': a2c},
        voting_strategy='weighted'
    )
    
    # Adjust weights using robust method (handles non-positive Sharpe ratios)
    try:
        ensemble.set_agent_weights({
            'PPO': ppo_metrics['mean_reward'],
            'DDPG': ddpg_metrics['mean_reward'],
            'A2C': a2c_metrics['mean_reward'],
        })
    except ValueError as e:
        logger.error(f"Could not set ensemble weights: {e}")
        logger.info("Using uniform equal weights (1/3 each)")
        ensemble.set_agent_weights({
            'PPO': 1.0,
            'DDPG': 1.0,
            'A2C': 1.0,
        })
    
    ensemble_metrics = ensemble.evaluate(n_episodes=5, env=test_env)
    
    # Results
    logger.info("\n" + "-"*70)
    logger.info("Results Summary:")
    logger.info("-"*70)
    logger.info(f"PPO Sharpe:      {ppo_metrics['mean_reward']:.4f}")
    logger.info(f"DDPG Sharpe:     {ddpg_metrics['mean_reward']:.4f}")
    logger.info(f"A2C Sharpe:      {a2c_metrics['mean_reward']:.4f}")
    logger.info(f"Ensemble Sharpe: {ensemble_metrics['mean_reward']:.4f}")
    logger.info("-"*70)
    
    # Save results and models
    logger.info("\nSaving results...")
    results_mgr = ResultsManager(RESULTS)
    
    # Prepare complete results
    complete_results = {
        'ppo_metrics': ppo_metrics,
        'ddpg_metrics': ddpg_metrics,
        'a2c_metrics': a2c_metrics,
        'ensemble_metrics': ensemble_metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save metrics as JSON
    results_mgr.save_metrics(complete_results, 'simple_pipeline_metrics')
    
    # Save models with metrics
    agents_to_save = {
        'PPO': ppo,
        'DDPG': ddpg,
        'A2C': a2c,
        'Ensemble': ensemble,
    }
    
    for agent_name, agent in agents_to_save.items():
        metrics = complete_results.get(f'{agent_name.lower()}_metrics', {})
        results_mgr.save_model(agent.model, agent_name.lower(), metrics)
    
    logger.info("✓ Models and metrics saved to results/")
    # Visualização
    logger.info("\nGerando visualizações...")
    
    # 1. Executar um episódio completo no ambiente de teste para gerar o histórico
    obs, _ = test_env.reset()
    done = False
    while not done:
        # Usa o ensemble para decidir (modo determinístico para gráfico limpo)
        action, _ = ensemble.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
    # 2. Criar DataFrame com o histórico de 'account_value'
    # O asset_memory tem o valor inicial + valor a cada dia. Ajustamos o tamanho.
    account_memory = test_env.asset_memory
    dates = test_env.df['date'].values if 'date' in test_env.df.columns else test_env.df.index
    
    # Garantir que os tamanhos batem (pode haver deslocamento de 1 dia pelo valor inicial)
    if len(account_memory) > len(dates):
        account_memory = account_memory[:len(dates)]
    
    df_visualizacao = pd.DataFrame({
        'date': dates[:len(account_memory)],
        'account_value': account_memory
    })
    # 3. Gerar e Salvar os Gráficos
    visualizer = TradingVisualizer()
    # Gráfico de Valor do Portfólio
    fig_portfolio = visualizer.plot_portfolio_value(df_visualizacao, title="Evolução do Patrimônio - Teste")
    results_mgr.save_plot(fig_portfolio, 'equity_curve')
    # Gráfico de Drawdown (Picos de queda)
    returns = df_visualizacao['account_value'].pct_change().dropna()
    fig_drawdown = visualizer.plot_drawdown(returns)
    results_mgr.save_plot(fig_drawdown, 'drawdown_underwater')
    # Comparativo de Métricas (Barras)
    metrics_dict = {
        'PPO': ppo_metrics, 'DDPG': ddpg_metrics, 
        'A2C': a2c_metrics, 'Ensemble': ensemble_metrics
    }
    fig_metrics = visualizer.plot_metrics_comparison(metrics_dict)
    results_mgr.save_plot(fig_metrics, 'metrics_comparison')
    
    logger.info("✓ Gráficos gerados e salvos em results/plots/")
    return complete_results


def rolling_window_ensemble(
    df: pd.DataFrame,
    pinn_engine: Optional[Any] = None,
    pinn_features_enabled: bool = False,
    ab_testing_enabled: bool = False
) -> Dict:
    """
    Production pipeline with rolling window cross-validation.
    
    Uses 14 weeks training + 4 weeks testing with sliding windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed data with technical indicators
    pinn_engine : PINNInferenceEngine, optional
        PINN inference engine for Heston parameter extraction
    pinn_features_enabled : bool
        Whether to include PINN features in environment
    ab_testing_enabled : bool
        Whether to run A/B testing (with/without PINN)
    """
    logger.info("\n" + "="*70)
    logger.info("ROLLING WINDOW ENSEMBLE STRATEGY (Production)") 
    if pinn_features_enabled:
        logger.info("+ PINN Features: ENABLED")
    if ab_testing_enabled:
        logger.info("+ A/B Testing: ENABLED")
    logger.info("="*70)
    
    # Create rolling window strategy
    rolling = RollingWindowStrategy(
        df=df,
        train_weeks=ROLLING_WINDOW_CONFIG['train_weeks'],
        test_weeks=ROLLING_WINDOW_CONFIG['test_weeks'],
        overlap_weeks=ROLLING_WINDOW_CONFIG['overlap_weeks'],
    )
    
    window_results = []
    
    for train_df, test_df, window_idx, date_range in rolling.generate_rolling_windows():
        logger.info(f"\n{'='*70}")
        logger.info(f"Window {window_idx}: {date_range['train_start'].date()} "
                   f"→ {date_range['test_end'].date()}")
        logger.info(f"{'='*70}")
        
        # Create environments
        train_env = StockTradingEnv(
            df=train_df,
            stock_dim=len(PRIMARY_ASSETS),
            initial_amount=INITIAL_CAPITAL,
            buy_cost_pct=TRANSACTION_COST,
            sell_cost_pct=TRANSACTION_COST,
            pinn_engine=pinn_engine,
            include_pinn_features=pinn_features_enabled,
        )
        
        test_env = StockTradingEnv(
            df=test_df,
            stock_dim=len(PRIMARY_ASSETS),
            initial_amount=INITIAL_CAPITAL,
            buy_cost_pct=TRANSACTION_COST,
            sell_cost_pct=TRANSACTION_COST,
            pinn_engine=pinn_engine,
            include_pinn_features=pinn_features_enabled,
        )
        
        # Train agents
        logger.info("Training agents...")
        ppo = PPOAgent(env=train_env, **PPO_PARAMS)
        ddpg = DDPGAgent(env=train_env, **DDPG_PARAMS)
        a2c = A2CAgent(env=train_env, **A2C_PARAMS)
        
        ppo.train(total_timesteps=20_000)
        ddpg.train(total_timesteps=20_000)
        a2c.train(total_timesteps=20_000)
        
        # Evaluate
        logger.info("Evaluating...")
        ppo_metrics = ppo.evaluate(num_episodes=3, env=test_env)
        ddpg_metrics = ddpg.evaluate(num_episodes=3, env=test_env)
        a2c_metrics = a2c.evaluate(num_episodes=3, env=test_env)
        
        # Create ensemble
        ensemble = EnsembleAgent(
            env=test_env,
            agents={'PPO': ppo, 'DDPG': ddpg, 'A2C': a2c},
            voting_strategy='weighted'
        )
        
        # Adjust weights using robust method
        try:
            ensemble.set_agent_weights({
                'PPO': ppo_metrics['mean_reward'],
                'DDPG': ddpg_metrics['mean_reward'],
                'A2C': a2c_metrics['mean_reward'],
            })
        except ValueError as e:
            logger.warning(f"Could not set ensemble weights: {e}")
            logger.info("Using uniform equal weights")
            ensemble.set_agent_weights({
                'PPO': 1.0,
                'DDPG': 1.0,
                'A2C': 1.0,
            })
        
        ensemble_metrics = ensemble.evaluate(n_episodes=3, env=test_env)
        
        # Store results
        result = {
            'window_idx': window_idx,
            'ppo_sharpe': ppo_metrics['mean_reward'],
            'ddpg_sharpe': ddpg_metrics['mean_reward'],
            'a2c_sharpe': a2c_metrics['mean_reward'],
            'ensemble_sharpe': ensemble_metrics['mean_reward'],
        }
        window_results.append(result)
        
        logger.info(f"PPO: {ppo_metrics['mean_reward']:.4f} | "
                   f"DDPG: {ddpg_metrics['mean_reward']:.4f} | "
                   f"A2C: {a2c_metrics['mean_reward']:.4f} | "
                   f"Ensemble: {ensemble_metrics['mean_reward']:.4f}")
        
        # Limit to 3 windows for demo
        if window_idx >= 2:
            logger.info("\n(Demo: limiting to 3 windows. Remove this in production.)")
            break
    
    # Aggregate results
    logger.info("\n" + "="*70)
    logger.info("Aggregated Results Across All Windows")
    logger.info("="*70)
    
    aggregated = rolling.get_metrics_across_windows(window_results)
    
    logger.info(f"\n{'Algorithm':<12} {'Avg Sharpe':<14} {'Std':<14}")
    logger.info("-"*40)
    
    if 'avg_ppo_sharpe' in aggregated:
        logger.info(f"{'PPO':<12} {aggregated['avg_ppo_sharpe']:>13.4f} "
                   f"{aggregated['std_ppo_sharpe']:>13.4f}")
    if 'avg_ddpg_sharpe' in aggregated:
        logger.info(f"{'DDPG':<12} {aggregated['avg_ddpg_sharpe']:>13.4f} "
                   f"{aggregated['std_ddpg_sharpe']:>13.4f}")
    if 'avg_a2c_sharpe' in aggregated:
        logger.info(f"{'A2C':<12} {aggregated['avg_a2c_sharpe']:>13.4f} "
                   f"{aggregated['std_a2c_sharpe']:>13.4f}")
    if 'avg_ensemble_sharpe' in aggregated:
        logger.info(f"{'ENSEMBLE':<12} {aggregated['avg_ensemble_sharpe']:>13.4f} "
                   f"{aggregated['std_ensemble_sharpe']:>13.4f}")
    
    logger.info(f"Total Windows: {aggregated['total_windows']}")
    
    # Save rolling window results
    logger.info("\nSaving rolling window results...")
    results_mgr = ResultsManager(RESULTS)
    
    complete_results = {
        'window_results': window_results,
        'aggregated': aggregated,
        'timestamp': datetime.now().isoformat(),
        'pipeline_type': 'rolling_window_ensemble',
    }
    
    # Save as JSON
    results_mgr.save_metrics(complete_results, 'rolling_ensemble_metrics')
    
    # Save as CSV for easier analysis
    df_results = pd.DataFrame(window_results)
    results_mgr.save_metrics_dataframe(df_results, 'rolling_ensemble_windows')
    
    logger.info("✓ Rolling window results saved to results/")
    
    # Visualização
    logger.info("\nGerando visualizações...")
    
    # 1. Executar um episódio completo no ambiente de teste para gerar o histórico
    obs, _ = test_env.reset()
    done = False
    while not done:
        # Usa o ensemble para decidir (modo determinístico para gráfico limpo)
        action, _ = ensemble.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = test_env.step(action)
        done = terminated or truncated
    # 2. Criar DataFrame com o histórico de 'account_value'
    # O asset_memory tem o valor inicial + valor a cada dia. Ajustamos o tamanho.
    account_memory = test_env.asset_memory
    dates = test_env.df['date'].values if 'date' in test_env.df.columns else test_env.df.index
    
    # Garantir que os tamanhos batem (pode haver deslocamento de 1 dia pelo valor inicial)
    if len(account_memory) > len(dates):
        account_memory = account_memory[:len(dates)]
    
    df_visualizacao = pd.DataFrame({
        'date': dates[:len(account_memory)],
        'account_value': account_memory
    })
    # 3. Gerar e Salvar os Gráficos
    visualizer = TradingVisualizer()
    # Gráfico de Valor do Portfólio
    fig_portfolio = visualizer.plot_portfolio_value(df_visualizacao, title="Evolução do Patrimônio - Teste")
    results_mgr.save_plot(fig_portfolio, 'equity_curve')
    # Gráfico de Drawdown (Picos de queda)
    returns = df_visualizacao['account_value'].pct_change().dropna()
    fig_drawdown = visualizer.plot_drawdown(returns)
    results_mgr.save_plot(fig_drawdown, 'drawdown_underwater')
    # Comparativo de Métricas (Barras)
    metrics_dict = {
        'PPO': ppo_metrics, 'DDPG': ddpg_metrics, 
        'A2C': a2c_metrics, 'Ensemble': ensemble_metrics
    }
    fig_metrics = visualizer.plot_metrics_comparison(metrics_dict)
    results_mgr.save_plot(fig_metrics, 'metrics_comparison')
    
    logger.info("✓ Gráficos gerados e salvos em results/plots/")
    
    return complete_results


def optuna_pipeline(
    df: pd.DataFrame,
    assets: Optional[List[str]] = None,
    agent_type: str = "PPO",
    n_trials: int = 20,
) -> Dict:
    """
    Optimize hyperparameters using Optuna Bayesian optimization.
    
    Args:
        df: Preprocessed DataFrame with price data
        assets: List of asset tickers
        agent_type: "PPO", "DDPG", or "A2C"
        n_trials: Number of optimization trials
    
    Returns:
        Dictionary with best hyperparameters and trial history
    """
    logger.info("\n" + "="*70)
    logger.info(f"HYPERPARAMETER OPTIMIZATION: {agent_type}")
    logger.info("="*70)
    
    if assets is None:
        assets = PRIMARY_ASSETS
    stock_dim = len(assets)
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    logger.info(f"Train: {len(train_df)} days | Test: {len(test_df)} days")
    
    # Create training environment
    train_env = StockTradingEnv(
        df=train_df,
        stock_dim=stock_dim,
        initial_amount=INITIAL_CAPITAL,
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
    )
    
    # Create test environment
    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        initial_amount=INITIAL_CAPITAL,
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
    )
    
    logger.info("✓ Environments created")
    
    # Initialize optimizer
    logger.info(f"\nInitializing Optuna optimizer for {agent_type}...")
    optimizer = HyperparameterOptimizer(
        agent_type=agent_type,
        env_fn=lambda: train_env,
        direction="maximize",
        sampler="tpe",
        seed=42,
    )
    
    # Run optimization
    logger.info(f"Running {n_trials} optimization trials...")
    try:
        optimizer_result = optimizer.optimize(
            n_trials=n_trials,
            timeout=TRAINING_CONFIG.get("timeout_seconds", 600),
            show_progress_bar=False,
        )
        
        best_params = optimizer_result.get('best_params', {})
        
        logger.info("\n" + "="*70)
        logger.info(f"OPTIMIZATION RESULTS: {agent_type}")
        logger.info("="*70)
        logger.info("\nBest hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Train final model with best hyperparameters
        logger.info("\nTraining final model with best hyperparameters...")
        if agent_type == "PPO":
            agent = PPOAgent(env=train_env, **best_params)
        elif agent_type == "DDPG":
            agent = DDPGAgent(env=train_env, **best_params)
        else:  # A2C
            agent = A2CAgent(env=train_env, **best_params)
        
        agent.train(total_timesteps=50_000)
        
        # Evaluate on test set
        logger.info("Evaluating final model...")
        metrics = agent.evaluate(num_episodes=5, env=test_env)
        
        logger.info(f"\nFinal Model Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Save optimization results
        logger.info("\nSaving Optuna optimization results...")
        results_mgr = ResultsManager(RESULTS)
        
        complete_results = {
            'agent_type': agent_type,
            'n_trials': n_trials,
            'best_params': best_params,
            'final_metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save as JSON
        results_mgr.save_metrics(complete_results, f'optuna_{agent_type}_optimization')
        
        # Save best model
        results_mgr.save_model(agent.model, f'{agent_type.lower()}_optuna', metrics)
        
        logger.info("✓ Optimization results and best model saved to results/")
        
        return complete_results
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DRL Stock Trading Agent - Main Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['simple-pipeline', 'rolling-ensemble', 'optuna-optimize'],
        default='rolling-ensemble',
        help='Pipeline mode to execute'
    )
    parser.add_argument(
        '--agent-type',
        choices=['PPO', 'DDPG', 'A2C'],
        default='PPO',
        help='Agent type for optimization (used with optuna-optimize mode)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of Optuna optimization trials (used with optuna-optimize mode)'
    )
    parser.add_argument(
        '--assets',
        nargs='+',
        default=PRIMARY_ASSETS,
        help='Assets to trade'
    )
    parser.add_argument(
        '--pinn-features',
        action='store_true',
        default=False,
        help='Enable PINN features in trading environment'
    )
    parser.add_argument(
        '--ab-testing',
        action='store_true',
        default=False,
        help='Run A/B testing (with vs without PINN features)'
    )
    parser.add_argument(
        '--pinn-finetune',
        action='store_true',
        default=False,
        help='Fine-tune PINN model during training (if available)'
    )
    
    args = parser.parse_args()
    
    # ⭐ REPRODUCIBILITY: Set random seeds FIRST (before any random operations)
    set_all_seeds(seed=42, verbose=True)
    assert_reproducible()
    
    # Validate PINN args
    pinn_features_enabled = args.pinn_features and PINN_AVAILABLE
    ab_testing_enabled = args.ab_testing and PINN_AVAILABLE and PINN_ENABLED
    
    if args.pinn_features and not PINN_AVAILABLE:
        logger.warning("PINN features requested but not available")
        pinn_features_enabled = False
    
    if args.ab_testing and not PINN_AVAILABLE:
        logger.warning("A/B testing requested but PINN not available")
        ab_testing_enabled = False
    
    logger.info("\n" + "="*70)
    logger.info("DRL Stock Trading Agent - Main Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Assets: {args.assets}")
    if pinn_features_enabled:
        logger.info("PINN Features: ENABLED")
    if ab_testing_enabled:
        logger.info("A/B Testing: ENABLED")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*70)
    
    try:
        # Setup
        setup_directories()
        
        # Load and preprocess data
        df = load_and_preprocess_data(args.assets)
        
        # ✓ Validate input safety before expensive RL training
        logger.info("\nValidating inputs...")
        validate_input_safety(
            df=df,
            stock_dim=len(args.assets),
            initial_amount=INITIAL_CAPITAL,
            required_columns=['time']  # Only require 'time' column; price column names vary
        )
        logger.info("✓ Input validation passed!")
        
        # Initialize PINN if enabled
        pinn_engine = None
        if pinn_features_enabled:
            logger.info("\nInitializing PINN inference engine...")
            try:
                pinn_engine = PINNInferenceEngine(
                    checkpoint_path=str(PINN_CHECKPOINT_PATH),
                    data_stats_path=str(PINN_DATA_STATS_PATH)
                )
                logger.info("✓ PINN engine initialized")
            except Exception as e:
                logger.warning(f"Could not initialize PINN: {e}")
                pinn_engine = None
                pinn_features_enabled = False
        
        # Execute pipeline
        if args.mode == 'simple-pipeline':
            results = simple_pipeline(df, assets=args.assets)
        elif args.mode == 'optuna-optimize':
            results = optuna_pipeline(
                df,
                assets=args.assets,
                agent_type=args.agent_type,
                n_trials=args.n_trials,
            )
        else:  # rolling-ensemble
            results = rolling_window_ensemble(
                df,
                pinn_engine=pinn_engine,
                pinn_features_enabled=pinn_features_enabled,
                ab_testing_enabled=ab_testing_enabled
            )
        
        logger.info("\n" + "="*70)
        logger.info("Pipeline Complete!")
        logger.info(f"End time: {datetime.now()}")
        logger.info("="*70)
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

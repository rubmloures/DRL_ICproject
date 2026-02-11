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

# Optional torch for PINN support
try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).parent))

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

# Setup logging - MUST BE BEFORE PINN IMPORT
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


def setup_directories() -> None:
    """Create necessary output directories."""
    for directory in [DATA_PROCESSED, TRAINED_MODELS, RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f" {directory}")


def load_raw_options_data(assets: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load raw options data (multiple rows per asset per day).
    
    Returns a dict mapping each asset to its options chain with proper indexing:
    - Index: (date, asset)
    - Columns: strike, premium, spot_price, delta, gamma, theta, vega, rho, days_to_maturity, ...
    
    Parameters
    ----------
    assets : list, optional
        Assets to load (default: PRIMARY_ASSETS)
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        DataFrame per asset with multi-level index (date, asset) for PINN
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Loading Raw Options Data (For PINN Inference)")
    logger.info("="*70)
    
    if assets is None:
        assets = PRIMARY_ASSETS
    
    loader = DataLoader(data_path=DATA_RAW)
    options_data = {}
    
    for asset_name in assets:
        logger.info(f"Loading {asset_name} options chain...")
        try:
            df = loader.load_asset(asset_name)
            
            # Keep all rows (multiple strikes per day)
            # Ensure date column exists
            date_col = next((c for c in ['time', 'date', 'data'] if c in df.columns), None)
            if not date_col:
                logger.warning(f"  No date column in {asset_name}. Skipping.")
                continue
            
            # Standardize to 'date'
            df = df.rename(columns={date_col: 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            df['asset'] = asset_name
            
            # Set multi-level index: (date, asset)
            df = df.set_index(['date', 'asset'])
            df = df.sort_index()
            
            options_data[asset_name] = df
            logger.info(f"  ✓ Loaded {len(df)} option rows for {asset_name}")
            
        except Exception as e:
            logger.warning(f"  Error loading {asset_name}: {e}")
            continue
    
    return options_data


def prepare_pinn_dataset(options_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Prepare dataset for PINN inference.
    
    Combines all options data into single DataFrame with (date, asset) multi-index.
    This ensures that when we create 30-day sliding windows:
    - We get windows of 30 days for EACH asset independently
    - NOT confused with 30 different assets on same day
    
    Parameters
    ----------
    options_data : Dict[str, pd.DataFrame]
        Options data per asset from load_raw_options_data()
    
    Returns
    -------
    pd.DataFrame
        Combined options data with multi-index (date, asset)
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1b: Preparing PINN Dataset (Multi-Index by Date & Asset)")
    logger.info("="*70)
    
    # Concatenate all assets
    pinn_df = pd.concat(options_data.values(), axis=0)
    pinn_df = pinn_df.sort_index()  # Sort by (date, asset)
    
    logger.info(f"✓ PINN Dataset ready: {len(pinn_df)} total option records")
    logger.info(f"  Unique dates: {pinn_df.index.get_level_values(0).nunique()}")
    logger.info(f"  Unique assets: {pinn_df.index.get_level_values(1).nunique()}")
    
    return pinn_df


def prepare_drl_dataset(options_data: Dict[str, pd.DataFrame], processor: DataProcessor) -> pd.DataFrame:
    """
    Prepare dataset for DRL environment.
    
    From options chains, extract:
    1. Daily spot price (OHLCV: open, high, low, close, volume)
    2. Technical indicators (SMA, RSI, MACD, etc.)
    3. Greeks mean/std per day (delta, gamma, theta, vega, rho)
    
    Result: One row per asset per day, columns prefixed by stock_i_*
    
    Parameters
    ----------
    options_data : Dict[str, pd.DataFrame]
        Options data per asset from load_raw_options_data()
    processor : DataProcessor
        Data processor for technical indicators
    
    Returns
    -------
    pd.DataFrame
        Wide-format daily data: index=date, columns=[stock_0_close, stock_1_close, ...]
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Preparing DRL Dataset (Daily OHLCV + Technicals + Greeks)")
    logger.info("="*70)
    
    combined_drl = pd.DataFrame()
    
    for i, (asset_name, df_options) in enumerate(options_data.items()):
        logger.info(f"\nProcessing Asset {i+1}/{len(options_data)}: {asset_name}")
        
        # Reset index to access date and asset columns
        df = df_options.reset_index()
        
        # Aggregate to daily (mean/std of spot_price, Greeks, etc.)
        daily_groups = []
        for date, group in df.groupby('date'):
            daily_record = {'date': date}
            
            # Spot price (OHLCV)
            if 'spot_price' in group.columns:
                daily_record['open'] = group['spot_price'].iloc[0] if len(group) > 0 else np.nan
                daily_record['high'] = group['spot_price'].max()
                daily_record['low'] = group['spot_price'].min()
                daily_record['close'] = group['spot_price'].iloc[-1] if len(group) > 0 else np.nan
            
            # Volume
            if 'acao_vol_fin' in group.columns:
                daily_record['volume'] = group['acao_vol_fin'].sum()
            else:
                daily_record['volume'] = len(group)  # Fallback: number of options
            
            # Greeks (mean and std)
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                if greek in group.columns:
                    daily_record[f'{greek}_mean'] = group[greek].mean()
                    daily_record[f'{greek}_std'] = group[greek].std()
            
            daily_groups.append(daily_record)
        
        df_daily = pd.DataFrame(daily_groups)
        logger.info(f"  Aggregated {len(df)} option rows → {len(df_daily)} daily records")
        
        # Add technical indicators
        df_daily = processor.clean_data(df_daily)
        df_daily = processor.add_technical_indicators(df_daily)
        
        # Rename columns with asset prefix
        prefix = f"stock_{i}_"
        rename_map = {c: f"{prefix}{c}" for c in df_daily.columns if c != 'date'}
        df_daily = df_daily.rename(columns=rename_map)
        
        # Merge into combined
        if combined_drl.empty:
            combined_drl = df_daily
        else:
            combined_drl = pd.merge(combined_drl, df_daily, on='date', how='inner')
    
    # Sort and set date as index
    combined_drl = combined_drl.sort_values('date').reset_index(drop=True)
    combined_drl['date'] = pd.to_datetime(combined_drl['date'])
    combined_drl['data'] = combined_drl['date']  # Alias
    combined_drl['time'] = combined_drl['date']  # Alias
    
    logger.info(f"\n✓ DRL Dataset ready: {len(combined_drl)} trading days")
    logger.info(f"  Columns: {len(combined_drl.columns)}")
    
    return combined_drl, options_data


def load_and_preprocess_data(assets: List[str] = None) -> pd.DataFrame:
    """
    DEPRECATED: Use load_raw_options_data() + prepare_drl_dataset() instead.
    
    Kept for backward compatibility.
    
    Load raw data, process individually, and merge into a Wide-Format DataFrame.
    
    Transformation:
    1. Loads each asset (PETR4, VALE3...)
    2. Filters Options Data -> Underlying Spot Price (1 row per day)
    3. Adds Technical Indicators per asset
    4. Merges into a single DataFrame: index=Date, columns=[stock_0_close, stock_1_close...]
    
    Parameters
    ----------
    assets : list, optional
        Assets to load (default: PRIMARY_ASSETS)
    
    Returns
    -------
    pd.DataFrame
        Wide-format processed data suitable for multi-agent environment.
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Data Loading and Preprocessing (Wide-Format)")
    logger.info("="*70)
    
    if assets is None:
        assets = PRIMARY_ASSETS
    
    loader = DataLoader(data_path=DATA_RAW)
    processor = DataProcessor()
    
    combined_df = pd.DataFrame()
    
    for i, asset_name in enumerate(assets):
        logger.info(f"Processing Asset {i+1}/{len(assets)}: {asset_name}...")
        
        try:
            # 1. Load individual asset
            df = loader.load_asset(asset_name)
            
            # 2. Handle Options Data (Reduce 600+ rows/day to 1 row/day)
            # Check for characteristic options columns
            if 'strike' in df.columns and 'type' in df.columns and 'spot_price' in df.columns:
                logger.info(f"  -> Detected Options Chain for {asset_name}. Aggregating to Daily Market View...")
                
                # Use DataProcessor's aggregation to preserve Greeks (mean/std) and extract Spot Price
                # This ensures PINN gets 'delta_mean', 'gamma_mean' etc., while DRL gets 'spot_price_mean' (close)
                df_daily = processor.aggregate_daily_options(df)
                
                # Remap columns to standard OHLCV for Technical Indicators
                
                # 1. Close Price: Use 'spot_price_mean' (aggregated) or 'spot_price' if flat
                if 'spot_price_mean' in df_daily.columns:
                    df_daily['close'] = df_daily['spot_price_mean']
                elif 'spot_price' in df_daily.columns:
                    df_daily['close'] = df_daily['spot_price']
                
                # 2. Volume: Use 'acao_vol_fin_mean' or default
                if 'acao_vol_fin_mean' in df_daily.columns:
                    df_daily['volume'] = df_daily['acao_vol_fin_mean']
                else:
                    df_daily['volume'] = 1.0
                
                # Keep original data primarily
                df = df_daily
                logger.info(f"  -> Reduced to {len(df)} daily records (aggregated).")
            
            # 3. Clean and Add Features (Per Asset)
            df = processor.clean_data(df)
            df = processor.add_technical_indicators(df)
            
            # 4. Rename Columns with Prefix (stock_0_, stock_1_...)
            # This is CRITICAL for the Environment to distinguish assets
            prefix = f"stock_{i}_"
            
            # Identify the date column to preserve it from prefixing
            date_col = next((c for c in ['date', 'time', 'data'] if c in df.columns), None)
            if not date_col:
                raise ValueError("Date column lost during processing")
            
            # Rename all except date
            rename_map = {c: f"{prefix}{c}" for c in df.columns if c != date_col}
            df = df.rename(columns=rename_map)
            
            # Standardize date column name for merging
            df = df.rename(columns={date_col: 'date'})
            
            # 5. Merge into Combined DataFrame
            if combined_df.empty:
                combined_df = df
            else:
                # Merge on Date (Inner Join to ensure alignment)
                combined_df = pd.merge(combined_df, df, on='date', how='inner')
        
        except FileNotFoundError:
            logger.error(f"  x File for {asset_name} not found. Skipping.")
            continue
        except Exception as e:
            logger.error(f"  x Error processing {asset_name}: {e}")
            raise
            
    # Final check
    if combined_df.empty:
        raise ValueError("No data loaded. Aborting.")
    
    # Sort by date
    combined_df = combined_df.sort_values('date').reset_index(drop=True)
    
    # Determine the 'data' (date) column for RollingWindowStrategy
    # It expects 'data' or 'date'. We have 'date'.
    combined_df['data'] = combined_df['date'] # Alias for compatibility
    combined_df['time'] = combined_df['date'] # Alias for validation compatibility
    
    logger.info(f" Final Wide-Format Dataset: {len(combined_df)} trading days, {len(combined_df.columns)} columns")
    logger.info(f"  Assets processed: {len(assets)}")
    
    return combined_df


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
    
    logger.info(" Environments created")
    
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
    
    logger.info(" Models and metrics saved to results/")
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
    
    logger.info(" Gráficos gerados e salvos em results/plots/")
    return complete_results


def rolling_window_ensemble(
    df: pd.DataFrame,
    pinn_engine: Optional[Any] = None,
    pinn_features_enabled: bool = False,
    ab_testing_enabled: bool = False,
    results_prefix: str = ""
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
        
        # Evaluate ensemble performance (Reward)
        ensemble_metrics = ensemble.evaluate(n_episodes=3, env=test_env)
        
        # ---------------------------------------------------------------------
        # Detailed Evaluation for Metrics (Sharpe, Return, Drawdown)
        # ---------------------------------------------------------------------
        # Run one deterministic episode to get daily portfolio values
        obs, _ = test_env.reset()
        done = False
        while not done:
            action, _ = ensemble.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated
            
        # Extract portfolio history
        portfolio_values = pd.Series(test_env.asset_memory)
        daily_returns = portfolio_values.pct_change().dropna()
        
        # Calculate Metrics
        if len(daily_returns) > 1:
            # Sharpe (Annualized) - assuming 252 days
            # If window is short, this is an approximation
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 1e-9 else 0.0
            
            # Cumulative Return
            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0
            
            # Annualized Return
            days = len(portfolio_values)
            annual_return = ((1 + total_return) ** (252 / days)) - 1.0 if days > 0 else 0.0
            
            # Max Drawdown
            cumulative = (1 + daily_returns).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_drawdown = drawdown.min()
        else:
            sharpe = 0.0
            total_return = 0.0
            annual_return = 0.0
            max_drawdown = 0.0
            
        # Store results
        result = {
            'window_idx': window_idx,
            'ppo_reward': ppo_metrics['mean_reward'],
            'ddpg_reward': ddpg_metrics['mean_reward'],
            'a2c_reward': a2c_metrics['mean_reward'],
            'ensemble_reward': ensemble_metrics['mean_reward'], # Still keep reward
            
            # Calculated Metrics
            'ensemble_sharpe': sharpe,
            'ensemble_total_return': total_return,
            'ensemble_annual_return': annual_return,
            'ensemble_max_drawdown': max_drawdown,
            
            'start_date': date_range['test_start'].date(),
            'end_date': date_range['test_end'].date(),
        }
        window_results.append(result)
        
        logger.info(f"Ensemble Metrics: Reward={ensemble_metrics['mean_reward']:.4f}, "
                   f"Sharpe={sharpe:.4f}, Ann.Ret={annual_return:.2%}, MaxDD={max_drawdown:.2%}")

        # Save Models for this window
        window_save_dir = RESULTS / "models" / f"window_{window_idx}"
        if results_prefix:
            window_save_dir = RESULTS / "models" / f"{results_prefix}window_{window_idx}"
        
        window_save_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            ensemble.save(window_save_dir / "ensemble_agent.zip")
            ppo.save(window_save_dir / "ppo_agent.zip")
            ddpg.save(window_save_dir / "ddpg_agent.zip")
            a2c.save(window_save_dir / "a2c_agent.zip")
            logger.info(f"Saved models to {window_save_dir}")
        except Exception as e:
            logger.error(f"Failed to save models for window {window_idx}: {e}")
        
        # Limit to 3 windows REMOVED for production
        # if window_idx >= 2: ...
    
    # Aggregate results
    logger.info("\n" + "="*70)
    logger.info("Aggregated Results Across All Windows")
    logger.info("="*70)
    
    aggregated = rolling.get_metrics_across_windows(window_results)
    
    logger.info(f"\n{'Algorithm':<12} {'Avg Reward':<14} {'Std':<14}")
    logger.info("-"*40)
    
    if 'avg_ppo_reward' in aggregated:
        logger.info(f"{'PPO':<12} {aggregated['avg_ppo_reward']:>13.4f} "
                   f"{aggregated['std_ppo_reward']:>13.4f}")
    if 'avg_ddpg_reward' in aggregated:
        logger.info(f"{'DDPG':<12} {aggregated['avg_ddpg_reward']:>13.4f} "
                   f"{aggregated['std_ddpg_reward']:>13.4f}")
    if 'avg_a2c_reward' in aggregated:
        logger.info(f"{'A2C':<12} {aggregated['avg_a2c_reward']:>13.4f} "
                   f"{aggregated['std_a2c_reward']:>13.4f}")
    if 'avg_ensemble_reward' in aggregated:
        logger.info(f"{'ENSEMBLE':<12} {aggregated['avg_ensemble_reward']:>13.4f} "
                   f"{aggregated['std_ensemble_reward']:>13.4f}")
    
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
    results_mgr.save_metrics(complete_results, f'{results_prefix}rolling_ensemble_metrics')
    
    # Save as CSV for easier analysis
    df_results = pd.DataFrame(window_results)
    results_mgr.save_metrics_dataframe(df_results, f'{results_prefix}rolling_ensemble_windows')
    
    logger.info(" Rolling window results saved to results/")
    
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
    
    logger.info(" Gráficos gerados e salvos em results/plots/")
    
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
    
    logger.info(" Environments created")
    
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
        
        logger.info(" Optimization results and best model saved to results/")
        
        return complete_results
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def enrich_with_pinn_features(df: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
    """
    Enrich DataFrame with Physics-Informed (PINN) features via Batch Inference.
       
    The PINN acts as a Market Regime Sensor that:
    1. Takes sliding windows of real price data (30 days)
    2. Calibrates Heston parameters (κ, θ, ξ, ρ) describing market physics
    3. Returns these latent parameters as features for the DRL agent
    
    This replaces real-time inference in the environment, speeding up training by ~100x.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide-format processed data with columns like stock_0_close, stock_1_close, etc.
    assets : List[str]
        List of asset tickers
        
    Returns
    -------
    pd.DataFrame
        DataFrame enriched with columns: pinn_kappa, pinn_theta, pinn_xi, pinn_rho
    """
    logger.info("\n" + "="*70)
    logger.info("STEP 1.5: PINN Feature Enrichment (Market Regime Sensor)")
    logger.info("="*70)
    logger.info("\nIntegration Architecture:")
    logger.info("   REAL Data (30-day window) → PINN LSTM → Heston Params (κ,θ,ξ,ρ)")
    logger.info("   These physics parameters enrich DRL agent observations")
    
    try:
        from src.pinn.inference_wrapper import PINNInferenceEngine
        from src.data.pinn_data_preprocessor import PINNDataPreprocessor
    except ImportError as e:
        logger.warning(f"PINN modules not found: {e}. Skipping enrichment.")
        return df

    # ========================================================================
    # Path Resolution: Find trained PINN model
    # ========================================================================
    
    # Primary path: PINN project
    checkpoint_path = PROJECT_ROOT / "src" / "pinn" / "weights" /"best_model_weights.pth"
    stats_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "data_stats.json"
    
    # Fallback paths
    if not checkpoint_path.exists():
        checkpoint_path = PROJECT_ROOT / "src" / "pinn" / "weights" /"best_model_weights.pth"
    if not stats_path.exists():
        stats_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "data_stats.json"
        
    if not checkpoint_path.exists() or not stats_path.exists():
        logger.warning(f"⚠️  PINN model files not found:")
        logger.warning(f"    Checkpoint: {checkpoint_path}")
        logger.warning(f"    Stats: {stats_path}")
        logger.warning("    Skipping PINN enrichment.")
        return df

    logger.info(f"\n Found PINN model: {checkpoint_path.name}")
    
    # ========================================================================
    # Initialize PINN Engine
    # ========================================================================
    logger.info("Initializing PINN Inference Engine...")
    try:
        # Determine device (torch may be None if not installed)
        device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        engine = PINNInferenceEngine(
            checkpoint_path=str(checkpoint_path),
            data_stats_path=str(stats_path),
            device=device,
            enable_validation=False
        )
        
        preprocessor = PINNDataPreprocessor(
            data_stats_path=str(stats_path),
            verbose=False
        )
        logger.info(" PINN Engine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize PINN engine: {e}")
        return df
    
    # ========================================================================
    # Extract Real Price Data (Primary Asset)
    # ========================================================================
    primary_prefix = "stock_0_"
    
    if f'{primary_prefix}close' not in df.columns:
        logger.warning(f"Primary asset columns not found. Available: {df.columns.tolist()}")
        return df
    
    logger.info("\n[1/3] Preparing real market data...")
    
    # Extract real OHLCV data from primary asset
    price_data = pd.DataFrame({
        'date': df['date'],
        'open': df.get(f'{primary_prefix}open', df[f'{primary_prefix}close']),
        'high': df.get(f'{primary_prefix}high', df[f'{primary_prefix}close'] * 1.01),
        'low': df.get(f'{primary_prefix}low', df[f'{primary_prefix}close'] * 0.99),
        'close': df[f'{primary_prefix}close'],
        'volume': df.get(f'{primary_prefix}volume', 1.0),
    })
    
    logger.info(f"    Extracted {len(price_data)} days of real market data")
    logger.info(f"     Date range: {price_data['date'].min().date()} to {price_data['date'].max().date()}")
    logger.info(f"     Price range: ${price_data['close'].min():.2f} - ${price_data['close'].max():.2f}")
    
    # ========================================================================
    # Create Sliding Windows (30-day observations for PINN)
    # ========================================================================
    logger.info("\n[2/3] Creating 30-day sliding windows for PINN...")
    
    WINDOW_SIZE = 30  # Standard window for LSTM in PINN
    
    try:
        # Adapt price_data for PINN preprocessor expectations
        # The preprocessor expects: spot_price, strike, days_to_maturity, r_rate
        price_data_adapted = price_data.copy()
        price_data_adapted['spot_price'] = price_data_adapted['close']
        
        # For PINN regime sensing on pure price data:
        # - Strike = ATM (current price)
        # - Days to maturity = 30 days (standard window)
        # - Risk-free rate = current Brazil SELIC rate (~10.5%)
        price_data_adapted['strike'] = price_data_adapted['close']
        price_data_adapted['days_to_maturity'] = WINDOW_SIZE
        price_data_adapted['r_rate'] = 0.105  # Brazil SELIC rate
        price_data_adapted['Dividend_Yield'] = 0.0
        price_data_adapted['symbol'] = assets[0] if assets else 'ASSET_0'
        
        # Use preprocessor to create LSTM features from real data
        x_seq, x_phy, metadata = preprocessor.calculate_lstm_features(
            price_data_adapted,
            window_size=WINDOW_SIZE
        )
        
        if len(x_seq) == 0:
            logger.warning(f"⚠️  No sequences generated. Dataset might be too short (<{WINDOW_SIZE} days).")
            return df
        
        logger.info(f"    Generated {len(x_seq)} sliding windows")
        logger.info(f"     Effective coverage: Days {WINDOW_SIZE} to {len(price_data)}")
        
    except Exception as e:
        logger.error(f"Error preparing sliding windows: {e}")
        logger.error("   Ensure price_data has required columns for PINN preprocessing")
        return df
    
    # ========================================================================
    # Run PINN Batch Inference (Extract Heston Parameters)
    # ========================================================================
    logger.info("\n[3/3] Running PINN batch inference...")
    logger.info(f"   Inferring: κ (kappa), θ (theta), ξ (xi), ρ (rho)")
    
    try:
        results = engine.infer_heston_params(
            x_seq,
            x_phy,
            return_price=False
        )
        
        logger.info(f"    Inference complete. Extracted parameters from {len(results['kappa'])} windows")
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        return df
    
    # ========================================================================
    # Create Results DataFrame and Merge
    # ========================================================================
    cols = ['pinn_kappa', 'pinn_theta', 'pinn_xi', 'pinn_rho']
    
    # Note: remove pinn_nu from list (not all versions may return it)
    try:
        res_df = pd.DataFrame({
            'date': [m[0] for m in metadata],
            'pinn_kappa': results['kappa'].flatten() if 'kappa' in results else np.zeros(len(metadata)),
            'pinn_theta': results['theta'].flatten() if 'theta' in results else np.zeros(len(metadata)),
            'pinn_xi': results['xi'].flatten() if 'xi' in results else np.zeros(len(metadata)),
            'pinn_rho': results['rho'].flatten() if 'rho' in results else np.zeros(len(metadata)),
        })
    except Exception as e:
        logger.error(f"Error creating results dataframe: {e}")
        logger.error(f"Results keys available: {list(results.keys())}")
        return df
    
    # Convert date column
    res_df['date'] = pd.to_datetime(res_df['date'])
    
    # Merge into main DataFrame with forward fill for warmup period
    df = df.drop(columns=[c for c in cols if c in df.columns], errors='ignore')
    df = pd.merge(df, res_df, on='date', how='left')
    
    # Fill NaNs: forward fill (warmup) then backward fill (end), finally zeros
    for col in cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill().fillna(0.0)
    
    # ========================================================================
    # Summary & Validation
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PINN Feature Enrichment Complete")
    logger.info("="*70)
    
    logger.info("\nHeston Market Regime Parameters:")
    for col in cols:
        if col in df.columns:
            valid_vals = df[col][df[col] != 0.0]
            if len(valid_vals) > 0:
                logger.info(f"   {col:12s}: {valid_vals.mean():8.4f} ± {valid_vals.std():8.4f} "
                           f"(n={len(valid_vals):5d})")
    
    logger.info(f"\n DataFrame enriched with {len(cols)} physics-informed features")
    logger.info(f"Final shape: {df.shape}")
    logger.info("\n These features enable DRL agent to recognize market regimes:")
    logger.info("   • κ (kappa): Volatility Mean-Reversion Speed")
    logger.info("   • θ (theta): Long-term Volatility Level") 
    logger.info("   • ξ (xi):    Volatility of Volatility (Tail Risk)")
    logger.info("   • ρ (rho):   Price-Vol Correlation (Leverage Effects)")
    logger.info("="*70 + "\n")
    
    return df


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
    
    #  REPRODUCIBILITY: Set random seeds FIRST (before any random operations)
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
        
        # =====================================================================
        # NEW PIPELINE: Separate PINN and DRL datasets
        # =====================================================================
        
        # 1. Load raw options data (multiple rows per asset per day)
        options_data = load_raw_options_data(args.assets)
        
        if not options_data:
            raise ValueError("No data loaded. Aborting.")
        
        # 2. Prepare DRL dataset (daily OHLCV + indicators)
        logger.info("\nPreparing DRL observation dataset...")
        processor = DataProcessor()
        df_drl, options_data_dict = prepare_drl_dataset(options_data, processor)
        
        # 3. Prepare PINN dataset (multi-index by date & asset)
        logger.info("\nPreparing PINN inference dataset...")
        df_pinn = prepare_pinn_dataset(options_data_dict)
        
        # 4. Run PINN inference if enabled
        pinn_features = None
        if pinn_features_enabled:
            logger.info("\nRunning PINN batch inference...")
            try:
                # TODO: Implement batch inference on df_pinn
                # For now, use legacy function
                df_drl = enrich_with_pinn_features(df_drl, args.assets)
            except Exception as e:
                logger.warning(f"PINN inference failed: {e}. Proceeding without features.")
                pinn_features_enabled = False
        
        # ✓ Validate input safety before expensive RL training
        logger.info("\nValidating inputs...")
        validate_input_safety(
            df=df_drl,
            stock_dim=len(args.assets),
            initial_amount=INITIAL_CAPITAL,
            required_columns=['time']  # Only require 'time' column; price column names vary
        )
        logger.info("✓ Input validation passed!")
        
        # Note: PINN features already merged into df_drl above
        
        # Execute pipeline
        if args.mode == 'simple-pipeline':
            results = simple_pipeline(df_drl, assets=args.assets)
        elif args.mode == 'optuna-optimize':
            results = optuna_pipeline(
                df_drl,
                assets=args.assets,
                agent_type=args.agent_type,
                n_trials=args.n_trials,
            )
        else:  # rolling-ensemble
            if ab_testing_enabled:
                logger.info("\n" + "="*50)
                logger.info("🔰 EXECUTING A/B TEST: Baseline vs PINN-Enhanced")
                logger.info("="*50)
                
                # --- Group B: Baseline (Control) ---
                logger.info("\n>>>Running Group B: Baseline (No PINN)...")
                results_b = rolling_window_ensemble(
                    df_drl,
                    pinn_engine=None,
                    pinn_features_enabled=False,
                    ab_testing_enabled=True,
                    results_prefix="group_B_baseline_"
                )
                
                # --- Group A: Experiment (Treatment) ---
                logger.info("\n>>> Running Group A: Experiment (With PINN)...")
                
                results_a = rolling_window_ensemble(
                    df_drl,
                    pinn_engine=None,
                    pinn_features_enabled=True,
                    ab_testing_enabled=True,
                    results_prefix="group_A_pinn_"
                )
                
                # Retrieve aggregated summary for logging
                summary_b = results_b.get('aggregated', {})
                summary_a = results_a.get('aggregated', {})
                
                logger.info("\n" + "="*50)
                logger.info("A/B TEST QUICK SUMMARY")
                logger.info("="*50)
                logger.info(f"{'Metric':<20} | {'Baseline (B)':<15} | {'PINN (A)':<15} | {'Diff':<10}")
                logger.info("-" * 65)
                for metric in ['avg_ensemble_reward', 'avg_ensemble_annual_return']:
                    val_b = summary_b.get(metric, 0.0)
                    val_a = summary_a.get(metric, 0.0)
                    diff = val_a - val_b
                    logger.info(f"{metric:<20} | {val_b:>15.4f} | {val_a:>15.4f} | {diff:>+10.4f}")
                logger.info("="*50)
                
                results = {'group_a': results_a, 'group_b': results_b}
                
            else:
                # Standard single run
                logger.info("\nExecuting Rolling Window Ensemble...")
                results = rolling_window_ensemble(
                    df_drl,
                    pinn_engine=None,
                    pinn_features_enabled=pinn_features_enabled,
                    ab_testing_enabled=False
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

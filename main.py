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
import json
import glob

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
    TRAIN_START, VAL_END
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
from src.agents.unified_callbacks import UnifiedRichDashboard, StepAuditCallback, TrainingLossAuditCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Setup logging - MUST BE BEFORE PINN IMPORT
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_output.txt", mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
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

# Diretório para logs do TensorBoard
TENSORBOARD_LOG_DIR = RESULTS / "tensorboard_logs"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_directories() -> None:
    """Create necessary output directories."""
    for directory in [DATA_PROCESSED, TRAINED_MODELS, RESULTS]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f" Prepared directory: {directory}")

def load_optimized_hyperparameters(agent_type: str, current_params: Dict) -> Dict:
    """
    Search for the latest Optuna optimization results and update parameters.
    
    Args:
        agent_type: 'PPO', 'DDPG', or 'A2C'
        current_params: Default parameters to be updated
    
    Returns:
        Updated parameters dictionary
    """
    results_mgr = ResultsManager(RESULTS)
    # Search for optuna_{agent_type}_optimization files
    pattern = f"optuna_{agent_type}_optimization"
    metrics_files = sorted(list(results_mgr.metrics_dir.glob(f"{pattern}*.json")), reverse=True)
    
    if not metrics_files:
        logger.info(f" No optimized hyperparameters found for {agent_type}. Using defaults.")
        return current_params
    
    latest_file = metrics_files[0]
    logger.info(f" Loading optimized hyperparameters for {agent_type} from {latest_file.name}")
    
    try:
        with open(latest_file, 'r') as f:
            opt_data = json.load(f)
            best_params = opt_data.get('best_params', {})
            
            # Map Optuna keys to Agent keys if necessary
            mapping = {'lr': 'learning_rate'}
            for opt_key, val in best_params.items():
                agent_key = mapping.get(opt_key, opt_key)
                # Handle nested dicts (policy_kwargs)
                if isinstance(val, dict) and agent_key in current_params:
                    current_params[agent_key].update(val)
                else:
                    current_params[agent_key] = val
                    
        return current_params
    except Exception as e:
        logger.warning(f" Failed to load optimized parameters: {e}. Using defaults.")
        return current_params


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
            
            # Spot price (OHLCV) from true stock values if available
            if 'acao_open' in group.columns and 'acao_close_ajustado' in group.columns:
                daily_record['open'] = group['acao_open'].iloc[0]
                daily_record['high'] = group['acao_high'].iloc[0]
                daily_record['low'] = group['acao_low'].iloc[0]
                daily_record['close'] = group['acao_close_ajustado'].iloc[0]
                daily_record['volume'] = group['acao_vol_fin'].iloc[0] if 'acao_vol_fin' in group.columns else len(group)
            elif 'spot_price' in group.columns:
                daily_record['open'] = group['spot_price'].iloc[0] if len(group) > 0 else np.nan
                daily_record['high'] = group['spot_price'].max()
                daily_record['low'] = group['spot_price'].min()
                daily_record['close'] = group['spot_price'].iloc[-1] if len(group) > 0 else np.nan
                daily_record['volume'] = len(group)
            
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
    
    # Audit export
    audit_path = RESULTS / "DRL_observation_dataset_audit.csv"
    combined_drl.to_csv(audit_path, index=False)
    logger.info(f"  Dataset exported for audit: {audit_path}")
    
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
    train_env = Monitor(train_env, filename=str(log_dir / "train_monitor"))

    test_env = StockTradingEnv(
        df=test_df,
        stock_dim=stock_dim,
        initial_amount=INITIAL_CAPITAL,
        buy_cost_pct=TRANSACTION_COST,
        sell_cost_pct=TRANSACTION_COST,
    )
    test_env = Monitor(test_env, filename=str(log_dir / "test_monitor"))
    
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
    
    # ==========================================
    # 4. NOVOS PLOTS: Análise de Sinais Intermediários
    # (Heston vs Preço, Volatilidade vs Preço)
    # ==========================================
    try:
        import matplotlib.pyplot as plt
        
        # Obter o preço do primeiro ativo (ex: PETR4) para sobrepor
        # O histórico real foi de len(account_memory)-1 dias
        n_days = len(account_memory) - 1
        asset_prices = [test_env.df.iloc[day][f'stock_0_close'] for day in range(n_days)]
        
        if hasattr(test_env, 'heston_memory') and len(test_env.heston_memory) > 0:
            # Plota Heston Nu vs Preço Real
            nu_series = [h.get('nu', 0) for h in test_env.heston_memory]
            xi_series = [h.get('xi', 0) for h in test_env.heston_memory]
            
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(asset_prices, color='blue', label='Preço (PETR4)')
            ax1.set_ylabel('Preço BRL', color='blue')
            ax2 = ax1.twinx()
            ax2.plot(nu_series, color='red', alpha=0.5, label='Heston Nu (Mean Vol)')
            ax2.plot(xi_series, color='green', alpha=0.5, label='Heston Xi (Vol of Vol)')
            ax2.set_ylabel('Heston Params', color='black')
            fig.suptitle('Heston Parameters vs Real Price (Regime Detection Audit)')
            fig.legend(loc='upper right')
            results_mgr.save_plot(fig, 'heston_vs_price_audit')
            plt.close(fig)
        
        if hasattr(test_env, 'volatility_memory') and len(test_env.volatility_memory) > 0:
            # Plota Realized Volatility vs Preço Real
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(asset_prices, color='blue', label='Preço (PETR4)')
            ax1.set_ylabel('Preço BRL', color='blue')
            ax2 = ax1.twinx()
            ax2.plot(test_env.volatility_memory, color='purple', alpha=0.6, label='Current Volatility')
            ax2.set_ylabel('Volatility', color='purple')
            fig.suptitle('Market Volatility vs Real Price')
            fig.legend(loc='upper right')
            results_mgr.save_plot(fig, 'volatility_vs_price_audit')
            plt.close(fig)
            
    except Exception as e:
        logger.error(f"Erro ao gerar plots intermediários de auditoria: {e}")

    logger.info(" Gráficos gerados e salvos em results/plots/")
    return complete_results


def rolling_window_ensemble(
    df: pd.DataFrame,
    assets: Optional[List[str]] = None,
    pinn_engine: Optional[Any] = None,
    pinn_features_enabled: bool = False,
    ab_testing_enabled: bool = False,
    results_prefix: str = "",
    minimal_plots: bool = False,
    monitor_queue: Optional[Any] = None
) -> Dict:
    """
    Production pipeline with rolling window cross-validation.
    
    Uses 14 weeks training + 4 weeks testing with sliding windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed data with technical indicators
    assets : list, optional
        List of asset tickers to trade. If None, uses PRIMARY_ASSETS from config.
        Controls stock_dim — MUST match the number of stocks loaded in df.
    pinn_engine : PINNInferenceEngine, optional
        PINN inference engine for Heston parameter extraction
    pinn_features_enabled : bool
        Whether to include PINN features in environment
    ab_testing_enabled : bool
        Whether to run A/B testing (with/without PINN)
    """
    # Resolve actual assets to use — default to config but respect caller's override
    if assets is None:
        assets = PRIMARY_ASSETS
    stock_dim = len(assets)

    logger.info("\n" + "="*70)
    logger.info("ROLLING WINDOW ENSEMBLE STRATEGY (Production)") 
    if pinn_features_enabled:
        logger.info("+ PINN Features: ENABLED")
    if ab_testing_enabled:
        logger.info("+ A/B Testing: ENABLED")
    logger.info("="*70)
    
    logger.info(f"Applying Temporal Filter: {TRAIN_START} to {VAL_END}")
    logger.info(f"  Trading {stock_dim} asset(s): {assets}")
    mask = (df['date'] >= pd.to_datetime(TRAIN_START)) & (df['date'] <= pd.to_datetime(VAL_END))
    original_len = len(df)
    df = df[mask].copy().sort_values('date').reset_index(drop=True)
    # CRITICAL: Sync the 'data' column used by RollingWindowStrategy after date-based filter
    # RollingWindowStrategy reads dates from 'data' column for reporting.
    # If 'data' column is stale (e.g. from before filter), windows will show wrong dates.
    df['data'] = df['date']
    df['time'] = df['date']
    logger.info(f" Data filtered: {original_len} -> {len(df)} rows | Date range: "
               f"{df['date'].iloc[0].date() if len(df) > 0 else 'N/A'} to "
               f"{df['date'].iloc[-1].date() if len(df) > 0 else 'N/A'}")
    
    if len(df) == 0:
        logger.error("No data remaining after temporal filter! Check config dates vs data dates.")
        return {}

    # Create rolling window strategy
    rolling = RollingWindowStrategy(
        df=df,
        train_weeks=ROLLING_WINDOW_CONFIG['train_weeks'],
        test_weeks=ROLLING_WINDOW_CONFIG['test_weeks'],
        overlap_weeks=ROLLING_WINDOW_CONFIG['overlap_weeks'],
        purge_days=ROLLING_WINDOW_CONFIG.get('purge_days', 0),
        purge_kfold_days=ROLLING_WINDOW_CONFIG.get('purge_kfold_days', None),
    )
    
    window_results = []
    
    # Initialize agents outside the loop for Warm-start
    ppo = None
    ddpg = None
    a2c = None
    
    first_train_env = None # Capture for final audit visualization

    # Load optimized parameters once if not already done
    global PPO_PARAMS, DDPG_PARAMS, A2C_PARAMS
    PPO_PARAMS = load_optimized_hyperparameters('PPO', PPO_PARAMS)
    DDPG_PARAMS = load_optimized_hyperparameters('DDPG', DDPG_PARAMS)
    A2C_PARAMS = load_optimized_hyperparameters('A2C', A2C_PARAMS)

    for train_df, test_df, window_idx, date_range in rolling.generate_rolling_windows():
        logger.info(f"\n{'='*70}")
        logger.info(f"Window {window_idx}: {date_range['train_start'].date()} "
                   f"→ {date_range['test_end'].date()}")
        logger.info(f"{'='*70}")
        
        # Diretório para salvar os logs de cada janela
        log_dir = RESULTS / "logs" / f"window_{window_idx}"
        log_dir.mkdir(parents=True, exist_ok=True)

        # --- STEP 1: Fit K-Means Regime Detector BEFORE creating environments ---
        # K-Means uses PINN features from training data to learn dynamic regime boundaries.
        # This must happen BEFORE train_env creation so the fitted detector can be passed.
        # We only fit K-Means if PINN features are enabled by the user.
        regime_detector_fitted = False
        if pinn_features_enabled:
            dummy_env_for_detector = StockTradingEnv(
                df=train_df,
                stock_dim=stock_dim,
                initial_amount=INITIAL_CAPITAL,
                buy_cost_pct=TRANSACTION_COST,
                sell_cost_pct=TRANSACTION_COST,
            )
            
            pinn_cols_canonical = ['pinn_nu', 'pinn_xi', 'pinn_rho']
            base_pinn_df = pd.DataFrame()
            
            # Priority 1: non-prefixed columns (set as alias for asset 0 in enrich_with_pinn_features)
            if all(col in train_df.columns for col in pinn_cols_canonical):
                base_pinn_df = train_df[pinn_cols_canonical].rename(
                    columns={'pinn_nu': 'nu', 'pinn_xi': 'xi', 'pinn_rho': 'rho'}
                )
            else:
                # Priority 2: prefixed columns for stock_0 (first asset as proxy)
                prefixed_pinn_cols = [f'stock_0_pinn_nu', f'stock_0_pinn_xi', f'stock_0_pinn_rho']
                if all(col in train_df.columns for col in prefixed_pinn_cols):
                    base_pinn_df = train_df[prefixed_pinn_cols].rename(
                        columns={'stock_0_pinn_nu': 'nu', 'stock_0_pinn_xi': 'xi', 'stock_0_pinn_rho': 'rho'}
                    )
            
            if not base_pinn_df.empty:
                # Drop NaNs before fitting
                clean_df = base_pinn_df.dropna()
                
                # SAFETY CHECK: K-Means fails silently when all data is constant/identical
                # (e.g. all zeros when PINN enrichment failed). Check variance before fitting.
                col_variances = clean_df.var()
                has_variance = (col_variances > 1e-10).any()
                
                if not has_variance:
                    logger.warning(
                        f" Window {window_idx}: PINN columns are constant/zero (PINN enrichment failed?). "
                        f"Skipping K-Means — regime detector will use static thresholds as fallback."
                    )
                elif len(clean_df) < 4:  # Minimum 4 samples for 4 clusters
                    logger.warning(f" Not enough clean PINN samples for K-Means ({len(clean_df)} < 4)")
                else:
                    logger.info(f" Fitting K-Means Regime Detector for Window {window_idx} ({len(clean_df)} samples)...")
                    dummy_env_for_detector.regime_detector.fit_kmeans(clean_df.astype(np.float64))
                    if dummy_env_for_detector.regime_detector.kmeans is not None and \
                       not np.isnan(dummy_env_for_detector.regime_detector.kmeans.cluster_centers_).any():
                        logger.info(f" ✅ K-Means fitted successfully. "
                                   f"Clusters: {dummy_env_for_detector.regime_detector.kmeans.cluster_centers_.shape[0]}")
                        regime_detector_fitted = True
                    else:
                        logger.error("🚨 K-Means fitting failed: Centroids contain NaNs!")
            else:
                logger.warning(f" Window {window_idx}: No PINN columns found in train_df. "
                              f"K-Means will use static thresholds. Run with --pinn-features to enable.")

        # --- STEP 2: Create environments with the pre-fitted regime detector ---
        train_env = StockTradingEnv(
            df=train_df,
            stock_dim=stock_dim,
            initial_amount=INITIAL_CAPITAL,
            buy_cost_pct=TRANSACTION_COST,
            sell_cost_pct=TRANSACTION_COST,
            pinn_engine=pinn_engine,
            include_pinn_features=pinn_features_enabled,
            print_verbosity=1000
        )
        
        # Save first train_env for the training sample plot at the end
        if first_train_env is None:
            first_train_env = train_env
        
        # Transfer the pre-fitted K-Means detector to the actual training environment
        if regime_detector_fitted:
            train_env.regime_detector = dummy_env_for_detector.regime_detector
            logger.info(f" ✅ Pre-fitted K-Means Regime Detector transferred to train_env")

        test_env = StockTradingEnv(
            df=test_df,
            stock_dim=stock_dim,
            initial_amount=INITIAL_CAPITAL,
            buy_cost_pct=TRANSACTION_COST,
            sell_cost_pct=TRANSACTION_COST,
            pinn_engine=pinn_engine,
            include_pinn_features=pinn_features_enabled,
        )
        
        # Critical Fix: Transfer fitted Regime Detector from Train to Test env
        # This ensures the Test environment uses the SAME regime boundaries learned during Training.
        # Note: Transfer whenever K-Means was successfully fitted, regardless of pinn_features_enabled.
        if regime_detector_fitted and hasattr(train_env, 'regime_detector'):
            test_env.regime_detector = train_env.regime_detector
            logger.info(f" 🔄 Transferred fitted K-Means Detector to Test Environment (Shared Knowledge)")

        # Steps configuration: full for first window, reduced for subsequent (Warm-start)
        base_steps = TRAINING_CONFIG.get("total_timesteps", 100_000)
        total_steps = base_steps if window_idx == 0 else int(base_steps * 0.3) # 30% for adaptation

        # Train/Update agents
        if window_idx == 0:
            logger.info("Initializing agents for first window...")
            ppo = PPOAgent(env=train_env, tensorboard_log=str(TENSORBOARD_LOG_DIR), **PPO_PARAMS)
            ddpg = DDPGAgent(env=train_env, tensorboard_log=str(TENSORBOARD_LOG_DIR), **DDPG_PARAMS)
            a2c = A2CAgent(env=train_env, tensorboard_log=str(TENSORBOARD_LOG_DIR), **A2C_PARAMS)
        else:
            logger.info(f"Warm-start: Updating agents environments for window {window_idx}...")
            ppo.model.set_env(train_env)
            ddpg.model.set_env(train_env)
            a2c.model.set_env(train_env)

        logger.info("Configurando SB3 Loggers (CSV/Tensorboard)...")
        sb3_log_dir = log_dir / "sb3_logs"
        
        ppo_logger = configure(str(sb3_log_dir / "ppo"), ["csv", "tensorboard"])
        ppo.model.set_logger(ppo_logger)
        
        ddpg_logger = configure(str(sb3_log_dir / "ddpg"), ["csv", "tensorboard"])
        ddpg.model.set_logger(ddpg_logger)
        
        a2c_logger = configure(str(sb3_log_dir / "a2c"), ["csv", "tensorboard"])
        a2c.model.set_logger(a2c_logger)

        logger.info(f"Training agents for {total_steps} timesteps (Adaptation)...")
        
        # Instantiate Curriculum Callback
        from src.agents.curriculum import CurriculumCallback
        curriculum_cb = CurriculumCallback(total_timesteps=total_steps)
        
        # --- NEW: Unified Dashboards and Audits ---
        # Normalize name for parallel dashboard matching (uppercase, no parallel_ prefix)
        label = results_prefix.upper().replace("PARALLEL_", "").replace("_", "") if "parallel" in results_prefix.lower() else (results_prefix if results_prefix else "Ensemble")
        
        ppo_dash = UnifiedRichDashboard(name=label, total_timesteps=total_steps, window_idx=window_idx, queue=monitor_queue)
        ddpg_dash = UnifiedRichDashboard(name=label, total_timesteps=total_steps, window_idx=window_idx, queue=monitor_queue)
        a2c_dash = UnifiedRichDashboard(name=label, total_timesteps=total_steps, window_idx=window_idx, queue=monitor_queue)
        
        audit_cb = StepAuditCallback(filename=f"audit_ensemble_w{window_idx}.csv")
        loss_audit_cb = TrainingLossAuditCallback(filename=f"loss_audit_ensemble_w{window_idx}.csv")
        
        ppo.train(total_timesteps=total_steps, custom_callbacks=[curriculum_cb, ppo_dash, audit_cb, loss_audit_cb], verbose=0)
        ddpg.train(total_timesteps=total_steps, custom_callbacks=[curriculum_cb, ddpg_dash, audit_cb, loss_audit_cb], verbose=0)
        a2c.train(total_timesteps=total_steps, custom_callbacks=[curriculum_cb, a2c_dash, audit_cb, loss_audit_cb], verbose=0)
        
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
        
        # Detailed Evaluation for Metrics (Sharpe, Return, Drawdown)
        # Run one deterministic episode to get daily portfolio values
        obs, _ = test_env.reset()
        done = False

        # Executar episódio completo
        while not done:
            action, _ = ensemble.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = test_env.step(action)
            done = terminated or truncated

        # Extrair históricos do ambiente
        account_memory = getattr(test_env, 'asset_memory', [INITIAL_CAPITAL])
        dates = test_env.df['date'].values[:len(account_memory)].tolist() if hasattr(test_env, 'df') and 'date' in test_env.df.columns else list(range(len(account_memory)))

        # Calcular benchmark (preço do primeiro ativo)
        if hasattr(test_env, 'df') and f'stock_0_close' in test_env.df.columns:
            benchmark_prices = test_env.df[f'stock_0_close'].values[:len(account_memory)].tolist()
        else:
            # Fallback: usar o próprio portfólio ou valores constantes
            benchmark_prices = [INITIAL_CAPITAL] * len(account_memory)

        # Calcular retornos diários
        portfolio_values = pd.Series(account_memory)
        daily_returns = portfolio_values.pct_change().dropna()

        # Calcular Métricas
        if len(daily_returns) > 1:
            # Sharpe (Annualized) - assumindo 252 dias
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
            'ensemble_reward': ensemble_metrics['mean_reward'],

            # Calculated Metrics
            'ensemble_sharpe': sharpe,
            'ensemble_total_return': total_return,
            'ensemble_annual_return': annual_return,
            'ensemble_max_drawdown': max_drawdown,

            'start_date': date_range['test_start'].date(),
            'end_date': date_range['test_end'].date(),

            # Full history for global plotting
            'portfolio_history': list(account_memory),
            'benchmark_history': list(benchmark_prices),
            'dates_history': [str(d) for d in dates[:len(account_memory)]]
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
    # Drop large history lists before saving CSV
    df_csv = df_results.drop(columns=['portfolio_history', 'benchmark_history', 'dates_history'], errors='ignore')
    results_mgr.save_metrics_dataframe(df_csv, f'{results_prefix}rolling_ensemble_windows')
    
    logger.info(" Rolling window results saved to results/")
    
    # Visualização Global
    logger.info("\nGerando visualizações globais (Toda a Validação)...")
    
    # Reconstruct full validation curve from individual windows
    full_portfolio = [INITIAL_CAPITAL]
    full_benchmark = [1.0]
    full_dates = []
    
    # Initial date for benchmark logic
    if window_results:
        first_bench_price = window_results[0]['benchmark_history'][0]
    
    for res in window_results:
        # Avoid including initial step value repeatedly if possible
        window_port = res['portfolio_history'][1:]
        window_bench = res['benchmark_history'][1:]
        window_dates = res['dates_history'][1:]
        
        # Adjust portfolio value to be continuous (chaining)
        last_val = full_portfolio[-1]
        returns = np.array(window_port) / res['portfolio_history'][0]
        full_portfolio.extend((returns * last_val).tolist())
        
        # Benchmark concatenated (normalized to start at 1.0)
        full_benchmark.extend((np.array(window_bench) / first_bench_price).tolist())
        full_dates.extend(window_dates)

    # Ensure sizes match
    min_len = min(len(full_portfolio), len(full_benchmark), len(full_dates))
    full_portfolio = full_portfolio[:min_len]
    full_benchmark = full_benchmark[:min_len]
    full_dates = full_dates[:min_len]
    
    df_visualizacao = pd.DataFrame({
        'date': full_dates,
        'account_value': full_portfolio
    })
    
    visualizer = TradingVisualizer()
    results_mgr = ResultsManager(RESULTS)

    if minimal_plots:
        logger.info("  [Minimal Plots Mode] Generating only Strategy vs Benchmark comparison.")
        try:
            # Em modo minimalista, preservamos apenas o comparativo real DRL vs Benchmark
            strategy_name = "DRL Ensemble (Baseline)"
            fig_bench = visualizer.plot_strategy_vs_benchmark(
                full_portfolio, 
                (np.array(full_benchmark) * INITIAL_CAPITAL).tolist(), 
                full_dates
            )
            # Customizar legenda para não mostrar "+ PINN" se for baseline
            fig_bench.data[0].name = strategy_name
            results_mgr.save_plot(fig_bench, f'{results_prefix}strategy_vs_benchmark')
        except Exception as e:
            logger.warning(f"Erro ao gerar plot minimalista: {e}")
    else:
        # 3. Gerar e Salvar os Gráficos Globais (Validação Completa - Modo Full)
        # Gráfico de Valor do Portfólio Global
        fig_portfolio = visualizer.plot_portfolio_value(df_visualizacao, title="Evolução do Patrimônio - Validação Global")
        results_mgr.save_plot(fig_portfolio, f'{results_prefix}equity_curve')
        
        # Gráfico de Drawdown Global
        returns_global = df_visualizacao['account_value'].pct_change().dropna()
        fig_drawdown = visualizer.plot_drawdown(returns_global)
        results_mgr.save_plot(fig_drawdown, f'{results_prefix}drawdown_underwater')
        
        # Comparativo de Métricas (Barras) usando as MÉDIAS de todas as janelas
        aggregated = complete_results.get('aggregated', {})
        
        metrics_dict = {
            'PPO': {
                'mean_reward': aggregated.get('avg_ppo_reward', 0.0),
            },
            'DDPG': {
                'mean_reward': aggregated.get('avg_ddpg_reward', 0.0),
            },
            'A2C': {
                'mean_reward': aggregated.get('avg_a2c_reward', 0.0),
            },
            'Ensemble': {
                'mean_reward': aggregated.get('avg_ensemble_reward', 0.0),
                'sharpe_ratio': aggregated.get('avg_ensemble_sharpe', 0.0),
                'annual_return': aggregated.get('avg_ensemble_annual_return', 0.0),
                'max_drawdown': aggregated.get('avg_ensemble_max_drawdown', 0.0),
            }
        }
        
        fig_metrics = visualizer.plot_metrics_comparison(metrics_dict)
        results_mgr.save_plot(fig_metrics, f'{results_prefix}metrics_comparison')
        
        # NOVOS GRÁFICOS (Eficácia de Regimes e Heston)
        audit_files = glob.glob(str(RESULTS / "logs" / "audit" / "audit_ensemble_w*.csv"))
        if audit_files:
            combined_audit = pd.concat([pd.read_csv(f) for f in audit_files]).sort_values("step")
            
            # 1. Eficácia de Regimes (Global)
            fig_regime_eff = visualizer.plot_regime_efficacy(combined_audit, title="Eficácia da Detecção de Regimes DRL/PINN - Global")
            results_mgr.save_plot(fig_regime_eff, f'{results_prefix}regime_efficacy_global')
            
            # 2. Heston vs Resultados
            fig_heston = visualizer.plot_heston_v_results(combined_audit, title="Parâmetros Heston (PINN) vs Performance do Agente")
            results_mgr.save_plot(fig_heston, f'{results_prefix}heston_vs_returns')
            
            # 3. Dispersão de Clusters de Regime (NEW)
            fig_clusters = visualizer.plot_regime_clusters(combined_audit, title="Clusters de Regime Detectados (Espaço Heston)")
            results_mgr.save_plot(fig_clusters, f'{results_prefix}regime_clusters_pinn')
            
            # 4. Intensidade de Ação por Regime
            fig_action_regime = visualizer.plot_regime_vs_actions(combined_audit, title="Intensidade de Ação (Agnóstia de Risco) por Regime")
            results_mgr.save_plot(fig_action_regime, f'{results_prefix}action_intensity_regime')
        
        try:
            # Gráfico Estratégia vs Benchmark Global
            fig_bench = visualizer.plot_strategy_vs_benchmark(full_portfolio, (np.array(full_benchmark) * INITIAL_CAPITAL).tolist(), full_dates)
            results_mgr.save_plot(fig_bench, f'{results_prefix}strategy_vs_benchmark')
            
            # Gráfico de Volatilidade Móvel Global
            fig_vol = visualizer.plot_rolling_volatility(full_portfolio)
            results_mgr.save_plot(fig_vol, f'{results_prefix}rolling_volatility')
            
            # Performance durante o Treinamento
            if first_train_env is not None:
                obs, _ = first_train_env.reset()
                done = False
                while not done:
                    action, _ = ensemble.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = first_train_env.step(action)
                    done = terminated or truncated
                    
                train_dates = first_train_env.df['date'].values[:len(first_train_env.asset_memory)]
                fig_train = visualizer.plot_portfolio_value(
                    pd.DataFrame({
                        'date': train_dates,
                        'account_value': first_train_env.asset_memory
                    }), 
                    title=f"Performance em Dados de Treino (Sample Window 0)"
                )
                results_mgr.save_plot(fig_train, f'{results_prefix}training_performance_sample')

            train_audit_file = RESULTS / "logs" / "audit" / "audit_ensemble_w0.csv"
            if train_audit_file.exists():
                train_audit = pd.read_csv(train_audit_file)
                fig_reg_train = visualizer.plot_regime_efficacy(train_audit, title="Eficácia de Regimes (Treinamento - W0)")
                results_mgr.save_plot(fig_reg_train, f'{results_prefix}regime_efficacy_train')
                
                fig_heston_train = visualizer.plot_heston_v_results(train_audit, title="Heston (PINN) Evolution during Training (W0)")
                results_mgr.save_plot(fig_heston_train, f'{results_prefix}heston_evolution_train')
        except Exception as e:
            logger.warning(f"Erro ao gerar gráficos detalhados: {e}")
        
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
    
    # Instantiate Curriculum Callback for final model training
    from src.agents.curriculum import CurriculumCallback
    curriculum_cb = CurriculumCallback(total_timesteps=50_000)
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
        
        agent.train(total_timesteps=50_000, custom_callbacks=[curriculum_cb])
        
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


def enrich_with_pinn_features(
    df: pd.DataFrame,
    assets: List[str],
    checkpoint_override: Optional[str] = None,
    stats_override: Optional[str] = None,
) -> pd.DataFrame:
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
    checkpoint_override : str | None
        If provided, ALL assets use this checkpoint instead of per-asset auto-detection.
        Typically set by the interactive menu (generalist or specialist selection).
    stats_override : str | None
        If provided, ALL assets use this data_stats.json instead of per-asset auto-detection.
        
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
    if checkpoint_override:
        logger.info(f"   Weight Mode: OVERRIDE → {checkpoint_override}")
    else:
        logger.info("   Weight Mode: AUTO-DETECT (specialist > generalist fallback)")
    
    try:
        from src.pinn.inference_wrapper import PINNInferenceEngine
        from src.data.pinn_data_preprocessor import PINNDataPreprocessor
    except ImportError as e:
        logger.warning(f"PINN modules not found: {e}. Skipping enrichment.")
        return df

    # Determine device
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    
    # Prepare enriched dataframe
    enriched_df = df.copy()
    
    # Loop over ALL assets to enrich
    for i, asset_name in enumerate(assets):
        logger.info(f"\n[PINN] Processing Asset {i+1}/{len(assets)}: {asset_name}")
        
        # 1. Resolve weight paths
        #    Priority: checkpoint_override > specialist dir > generalist fallback
        if checkpoint_override:
            # Menu-selected mode: use overridden path for ALL assets
            checkpoint_path = Path(checkpoint_override)
            stats_path = Path(stats_override) if stats_override else \
                         checkpoint_path.parent / "data_stats.json"
            logger.info(f"       Weights: {checkpoint_path.parent.name}/{checkpoint_path.name} [override]")
        else:
            # Auto-detect: prefer specialist then generalist
            checkpoint_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "best_model_weights.pth"
            stats_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "data_stats.json"
            
            asset_prefix = asset_name[:4]  # e.g. PETR
            specialist_dir = PROJECT_ROOT / "src" / "pinn" / "weights" / asset_prefix
            
            if specialist_dir.exists() and (specialist_dir / "best_model_weights.pth").exists():
                checkpoint_path = specialist_dir / "best_model_weights.pth"
                stats_path = specialist_dir / "data_stats.json"
                logger.info(f"       Using Specialist Weights: {specialist_dir.name}")
            else:
                logger.info(f"       Using Generalist Weights (no specialist for {asset_prefix})")
        
        if not checkpoint_path.exists():
            logger.warning(f"       No weights found at {checkpoint_path}. Skipping.")
            continue

        try:
            # 2. Initialize Engine specifically for this asset
            engine = PINNInferenceEngine(
                checkpoint_path=str(checkpoint_path),
                data_stats_path=str(stats_path),
                device=device,
                enable_validation=False,
                asset_name=asset_name, # Critical: Matches MARL logic
                verbose=False 
            )
            
            preprocessor = PINNDataPreprocessor(
                data_stats_path=str(stats_path),
                verbose=False
            )
            
            # 3. Extract Data for stock_i_
            col_prefix = f"stock_{i}_"
            if f'{col_prefix}close' not in df.columns:
                logger.warning(f"       Columns for {col_prefix} not found. Skipping.")
                continue
                
            price_data = pd.DataFrame({
                'date': df['date'],
                'close': df[f'{col_prefix}close'],
                'volume': df.get(f'{col_prefix}volume', 1.0),
            })
            
            # Adapt for Preprocessor
            WINDOW_SIZE = 30
            price_data_adapted = price_data.copy()
            price_data_adapted['spot_price'] = price_data_adapted['close']
            price_data_adapted['strike'] = price_data_adapted['close']
            price_data_adapted['days_to_maturity'] = WINDOW_SIZE
            price_data_adapted['r_rate'] = 0.105
            price_data_adapted['Dividend_Yield'] = 0.0
            price_data_adapted['symbol'] = asset_name
            
            # 4. Generate Sequences & Infer
            x_seq, x_phy, metadata = preprocessor.calculate_lstm_features(
                price_data_adapted, window_size=WINDOW_SIZE
            )
            
            if len(x_seq) == 0:
                logger.warning("       Not enough data for inference windows.")
                continue
            
            # Mini-batch inference to prevent OOM with large datasets.
            # Processing thousands of windows at once (e.g. 1441 for 1470 days) can
            # exhaust CPU RAM. We chunk into PINN_BATCH_SIZE_INFERENCE sized batches.
            PINN_INFER_BATCH = 256  # Safe batch size for CPU inference
            n_windows_total = len(x_seq)
            all_results = {k: [] for k in ['nu', 'theta', 'kappa', 'xi', 'rho']}
            
            logger.info(f"       Inferring {n_windows_total} windows in batches of {PINN_INFER_BATCH}...")
            for batch_start in range(0, n_windows_total, PINN_INFER_BATCH):
                batch_end = min(batch_start + PINN_INFER_BATCH, n_windows_total)
                x_seq_batch = x_seq[batch_start:batch_end]
                x_phy_batch = x_phy[batch_start:batch_end]
                
                try:
                    batch_results = engine.infer_heston_params(
                        x_seq_batch, x_phy_batch, return_price=False
                    )
                    for k in all_results:
                        all_results[k].append(batch_results[k])
                except Exception as batch_err:
                    logger.warning(f"       Batch [{batch_start}:{batch_end}] failed: {batch_err}. Filling with zeros.")
                    n_batch = batch_end - batch_start
                    for k in all_results:
                        all_results[k].append(np.zeros((n_batch, 1)))
            
            # Concatenate batches
            results = {k: np.vstack(v) for k, v in all_results.items() if v}
            if not results:
                logger.warning(f"       All batches failed for {asset_name}. Skipping.")
                continue
            
            # 5. Create Result DF
            res_df = pd.DataFrame({
                'date': [m[0] for m in metadata],
                f'{col_prefix}pinn_kappa': results['kappa'].flatten(),
                f'{col_prefix}pinn_theta': results['theta'].flatten(),
                f'{col_prefix}pinn_xi': results['xi'].flatten(),
                f'{col_prefix}pinn_rho': results['rho'].flatten(),
                f'{col_prefix}pinn_nu': results['nu'].flatten(),
            })
            
            res_df['date'] = pd.to_datetime(res_df['date'])
            
            # 5b. AGGREGATE BY DATE
            # res_df currently has one row per option/window. 
            # DRL needs one row per day. We take the mean.
            logger.debug(f"       Aggregating {len(res_df)} inference windows to daily resolution...")
            res_df = res_df.groupby('date').mean().reset_index()
            
            # 5c. Global Consolidation
            # K-Means Regime Detector uses 'pinn_nu', 'pinn_xi', 'pinn_rho' (no prefix).
            # Strategy: Maintain a global average across all processed assets for a portfolio-level signal.
            for param in ['kappa', 'theta', 'xi', 'rho', 'nu']:
                generic_col = f'pinn_{param}'
                asset_col = f'{col_prefix}pinn_{param}'
                
                if generic_col in enriched_df.columns:
                    # Rolling mean: (Existing Global Value + Current Asset Value) / 2
                    existing_data = enriched_df.set_index('date')[generic_col]
                    # Map existing data back to res_df dates
                    res_df[generic_col] = (res_df['date'].map(existing_data).fillna(res_df[asset_col]) + res_df[asset_col]) / 2.0
                else:
                    # First asset processed establishes the baseline
                    res_df[generic_col] = res_df[asset_col]
            
            # 6. Merge
            # Identify columns to merge (exclude date)
            cols_to_merge = [c for c in res_df.columns if c != 'date']
            
            # Drop existing if any (to avoid duplicates/suffixes)
            enriched_df = enriched_df.drop(columns=[c for c in cols_to_merge if c in enriched_df.columns], errors='ignore')
            enriched_df = pd.merge(enriched_df, res_df, on='date', how='left')
            
            # Fill NaNs
            for col in cols_to_merge:
                enriched_df[col] = enriched_df[col].ffill().bfill().fillna(0.0).astype(np.float64)
                
            logger.info(f"       ✓ Inferred {len(results['kappa'])} windows for {asset_name}")
            
        except Exception as e:
            logger.error(f"       Error inferring for {asset_name}: {e}")
            # Continue to next asset instead of breaking pipeline
            continue
            
    # ========================================================================
    # Summary & Validation
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PINN Feature Enrichment Complete")
    logger.info("="*70)
    
    # Validation Log based on Asset 0 (Primary)
    check_cols = ['pinn_kappa', 'pinn_xi', 'pinn_rho']
    logger.info("\nPrimary Asset Regime Parameters (Sample):")
    for col in check_cols:
        if col in enriched_df.columns:
            valid_vals = enriched_df[col][enriched_df[col] != 0.0]
            if len(valid_vals) > 0:
                logger.info(f"   {col:12s}: {valid_vals.mean():8.4f} ± {valid_vals.std():8.4f} "
                           f"(n={len(valid_vals):5d})")
    
    logger.info(f"\n DataFrame enriched. Final shape: {enriched_df.shape}")
    logger.info("="*70 + "\n")
    
    return enriched_df


def interactive_asset_selector() -> dict:
    """
    Interactive Rich terminal menu for asset and PINN weight selection.
    
    Returns
    -------
    dict with keys:
        'assets': List[str] - assets to trade
        'pinn_mode': str     - 'generalist' | 'specialist' | 'none'
        'pinn_checkpoint': str | None - path to checkpoint
        'pinn_stats': str | None      - path to data_stats.json
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from pathlib import Path
    
    c = Console()
    weights_root = PROJECT_ROOT / "src" / "pinn" / "weights"
    generalist_ckpt  = weights_root / "best_model_weights.pth"
    generalist_stats = weights_root / "data_stats.json"
    
    # Discover available specialist assets
    specialists = {}
    for d in sorted(weights_root.iterdir()):
        if d.is_dir():
            ckpt  = d / "best_model_weights.pth"
            stats = d / "data_stats.json"
            if ckpt.exists() and stats.exists():
                # specialist directory name maps to asset prefix (e.g. PETR → PETR4)
                prefix = d.name.upper()  # e.g. PETR, VALE, ABEV
                specialists[prefix] = {'ckpt': str(ckpt), 'stats': str(stats)}
    
    c.print()
    c.print(Panel(
        "[bold cyan]Rolling Ensemble — Asset & PINN Weight Selection[/bold cyan]\n"
        "[dim]Heston parameters calibrated by PINN drive the K-Means Regime Detector.[/dim]",
        border_style="blue", box=box.DOUBLE
    ))
    
    # ---- Option table ----
    opt_tbl = Table(show_header=True, box=box.SIMPLE_HEAVY, expand=False)
    opt_tbl.add_column("#", style="bold yellow", width=3)
    opt_tbl.add_column("Mode", style="bold white", width=35)
    opt_tbl.add_column("Assets", style="cyan")
    opt_tbl.add_column("PINN Weights", style="magenta")
    
    opt_tbl.add_row(
        "1",
        "Use Defaults",
        "PETR4, VALE3",
        f"Generalist  ({generalist_ckpt.name})" if generalist_ckpt.exists() else "[red]⚠ not found[/red]"
    )
    opt_tbl.add_row(
        "2",
        "Enter Custom Assets",
        "[dim]type your own...[/dim]",
        f"Generalist  ({generalist_ckpt.name})" if generalist_ckpt.exists() else "[red]⚠ not found[/red]"
    )
    
    # One row per specialist
    for prefix, info in specialists.items():
        # Map prefix to canonical asset name (add suffix if not present)
        # e.g. PETR → PETR4, VALE → VALE3, ABEV → ABEV3
        suffix_map = {'PETR': 'PETR4', 'VALE': 'VALE3', 'ABEV': 'ABEV3', 'BBAS': 'BBAS3',
                      'CSNA': 'CSNA3', 'MGLU': 'MGLU3', 'B3SA': 'B3SA3', 'BOVA': 'BOVA11'}
        canon = suffix_map.get(prefix, prefix)
        opt_tbl.add_row(
            "3" if prefix == list(specialists.keys())[0] else " ",
            f"Individual Specialist: {canon}",
            canon,
            f"Specialist  ({prefix}/best_model_weights.pth)"
        )
    
    c.print(opt_tbl)
    c.print()
    
    # ---- Get choice ----
    valid_choices = {"1", "2"}
    specialist_list = list(specialists.keys())
    if specialist_list:
        valid_choices.add("3")
    
    choice = ""
    while choice not in valid_choices:
        choice = c.input("[bold yellow]Select mode[/bold yellow] (1/2/3): ").strip()
        if choice not in valid_choices:
            c.print(f"[red]Invalid choice. Please enter {'/'.join(sorted(valid_choices))}.[/red]")
    
    # ---- Mode 1: Defaults ----
    if choice == "1":
        assets = ["PETR4", "VALE3"]
        pinn_mode = "generalist" if generalist_ckpt.exists() else "none"
        c.print(f"[green]✓ Defaults selected:[/green] {assets}")
        c.print(f"[green]✓ PINN:[/green] Generalist weights ({generalist_ckpt})")
        return {
            'assets': assets,
            'pinn_mode': pinn_mode,
            'pinn_checkpoint': str(generalist_ckpt) if generalist_ckpt.exists() else None,
            'pinn_stats': str(generalist_stats) if generalist_stats.exists() else None,
        }
    
    # ---- Mode 2: Custom ----
    if choice == "2":
        raw = c.input(
            "[bold yellow]Enter asset tickers separated by spaces[/bold yellow] "
            "(e.g. PETR4 VALE3 BBAS3): "
        ).strip().upper()
        assets = [a.strip() for a in raw.split() if a.strip()]
        if not assets:
            c.print("[red]No assets entered. Falling back to defaults (PETR4, VALE3).[/red]")
            assets = ["PETR4", "VALE3"]
        pinn_mode = "generalist" if generalist_ckpt.exists() else "none"
        c.print(f"[green]✓ Custom assets:[/green] {assets}")
        c.print(f"[green]✓ PINN:[/green] Generalist weights")
        return {
            'assets': assets,
            'pinn_mode': pinn_mode,
            'pinn_checkpoint': str(generalist_ckpt) if generalist_ckpt.exists() else None,
            'pinn_stats': str(generalist_stats) if generalist_stats.exists() else None,
        }
    
    # ---- Mode 3: Specialist ----
    if choice == "3":
        if len(specialist_list) == 1:
            selected_prefix = specialist_list[0]
        else:
            # Sub-menu to pick specialist
            c.print("\n[bold]Available specialists:[/bold]")
            suffix_map = {'PETR': 'PETR4', 'VALE': 'VALE3', 'ABEV': 'ABEV3', 'BBAS': 'BBAS3',
                          'CSNA': 'CSNA3', 'MGLU': 'MGLU3', 'B3SA': 'B3SA3', 'BOVA': 'BOVA11'}
            for idx, prefix in enumerate(specialist_list, start=1):
                canon = suffix_map.get(prefix, prefix)
                c.print(f"  [yellow]{idx}[/yellow]. {canon}  (weights: {prefix}/)")
            
            sub_choice = ""
            while not sub_choice.isdigit() or int(sub_choice) not in range(1, len(specialist_list) + 1):
                sub_choice = c.input("[bold yellow]Select specialist[/bold yellow]: ").strip()
            selected_prefix = specialist_list[int(sub_choice) - 1]
        
        suffix_map = {'PETR': 'PETR4', 'VALE': 'VALE3', 'ABEV': 'ABEV3', 'BBAS': 'BBAS3',
                      'CSNA': 'CSNA3', 'MGLU': 'MGLU3', 'B3SA': 'B3SA3', 'BOVA': 'BOVA11'}
        canon_asset = suffix_map.get(selected_prefix, selected_prefix)
        assets = [canon_asset]
        info = specialists[selected_prefix]
        
        c.print(f"[green]✓ Specialist selected:[/green] {canon_asset}")
        c.print(f"[green]✓ PINN:[/green] Specialist weights ({selected_prefix}/)")
        return {
            'assets': assets,
            'pinn_mode': 'specialist',
            'pinn_checkpoint': info['ckpt'],
            'pinn_stats': info['stats'],
        }
    
    # Fallback (should never reach here)
    return {'assets': PRIMARY_ASSETS, 'pinn_mode': 'none',
            'pinn_checkpoint': None, 'pinn_stats': None}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='DRL Stock Trading Agent - Main Pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['simple-pipeline', 'rolling-ensemble', 'optuna-optimize', 'baseline-comparison'],
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
        default=None,   # None = trigger interactive menu for rolling-ensemble
        help='Assets to trade. If omitted in rolling-ensemble mode, shows interactive selection menu.'
    )
    parser.add_argument(
        '--asset-mode',
        choices=['defaults', 'custom', 'specialist'],
        default=None,
        help='(Non-interactive) asset selection mode: defaults|custom|specialist'
    )
    parser.add_argument(
        '--specialist',
        default=None,
        help='(Non-interactive) specialist prefix, e.g. PETR (requires --asset-mode specialist)'
    )
    parser.add_argument(
        '--pinn-features',
        action='store_true',
        default=False,
        help='Enable PINN features in trading environment (observation space)'
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
    parser.add_argument(
        '--no-menu',
        action='store_true',
        default=False,
        help='Skip interactive menu; use --assets or PRIMARY_ASSETS from config directly'
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # ASSET SELECTION: interactive menu OR non-interactive CLI flags
    # =========================================================================
    # The interactive menu is shown when:
    #   - Mode is rolling-ensemble  AND
    #   - --assets was not explicitly given  AND
    #   - --no-menu was not passed
    pinn_checkpoint_override = None  # path to use instead of auto-detection
    pinn_stats_override = None
    
    is_rolling = args.mode in ('rolling-ensemble', 'baseline-comparison')
    wants_menu = is_rolling and args.assets is None and not args.no_menu
    
    if wants_menu:
        selection = interactive_asset_selector()
        args.assets = selection['assets']
        pinn_mode   = selection['pinn_mode']         # 'generalist' | 'specialist' | 'none'
        pinn_checkpoint_override = selection['pinn_checkpoint']
        pinn_stats_override      = selection['pinn_stats']
        # Automatically enable PINN features if specialist/generalist weights found
        if pinn_mode in ('generalist', 'specialist') and PINN_AVAILABLE:
            args.pinn_features = True
    else:
        # Non-interactive: resolve from CLI flags or defaults
        if args.assets is None:
            args.assets = PRIMARY_ASSETS
        
        if args.asset_mode == 'specialist' and args.specialist:
            weights_root = PROJECT_ROOT / "src" / "pinn" / "weights"
            spec_dir = weights_root / args.specialist.upper()
            pinn_checkpoint_override = str(spec_dir / "best_model_weights.pth") if spec_dir.exists() else None
            pinn_stats_override      = str(spec_dir / "data_stats.json") if spec_dir.exists() else None
        elif args.asset_mode in ('defaults', 'custom', None):
            weights_root = PROJECT_ROOT / "src" / "pinn" / "weights"
            generalist_ckpt  = weights_root / "best_model_weights.pth"
            generalist_stats = weights_root / "data_stats.json"
            pinn_checkpoint_override = str(generalist_ckpt)  if generalist_ckpt.exists()  else None
            pinn_stats_override      = str(generalist_stats) if generalist_stats.exists() else None

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
            
        # 1.5 Temporal Filter (QA MASTER Alignment)
        # We filter here so the audit CSV and all plots only see the target period (2023-2025)
        logger.info(f"\n[Temporal Alignment] Filtering raw data: {TRAIN_START} to {VAL_END}")
        start_dt = pd.to_datetime(TRAIN_START).date()
        end_dt = pd.to_datetime(VAL_END).date()
        
        filtered_options = {}
        for asset, df_opt in options_data.items():
            # index is (date, asset). date level is 0.
            mask = (df_opt.index.get_level_values(0) >= start_dt) & \
                   (df_opt.index.get_level_values(0) <= end_dt)
            df_filtered = df_opt[mask].copy()
            if not df_filtered.empty:
                filtered_options[asset] = df_filtered
                logger.info(f"  {asset}: {len(df_filtered)} rows remaining.")
            else:
                logger.warning(f"  {asset}: No data remaining after filter!")
        
        options_data = filtered_options
        if not options_data:
            raise ValueError("No data remaining after temporal filter! Check config.py dates.")
        
        # 2. Prepare DRL dataset (daily OHLCV + indicators)
        logger.info("\nPreparing DRL observation dataset...")
        processor = DataProcessor()
        df_drl, options_data_dict = prepare_drl_dataset(options_data, processor)
        
        # 3. Prepare PINN dataset (multi-index by date & asset)
        logger.info("\nPreparing PINN inference dataset...")
        df_pinn = prepare_pinn_dataset(options_data_dict)
        
        # 4. Run PINN inference — only if requested or A/B testing is enabled.
        # Reason: The K-Means Regime Detector needs pinn_nu/xi/rho columns 
        # to learn dynamic market regime boundaries, but we should respect the --no-pinn flag.
        if PINN_AVAILABLE and (pinn_features_enabled or ab_testing_enabled):
            # Log which weights will be used
            if pinn_checkpoint_override:
                logger.info(f"\n[PINN] Weight override active: {pinn_checkpoint_override}")
            else:
                logger.info("\n[PINN] Using auto-detected weights (specialist if available, else generalist)")
            logger.info("Running PINN batch inference (required for K-Means Regime Detection)...")
            try:
                df_drl = enrich_with_pinn_features(
                    df_drl,
                    args.assets,
                    checkpoint_override=pinn_checkpoint_override,
                    stats_override=pinn_stats_override,
                )
                logger.info("✅ PINN enrichment complete. Heston columns available for K-Means.")
            except Exception as e:
                logger.warning(f"PINN enrichment failed: {e}. K-Means will use static fallback.")
                if pinn_features_enabled:
                    logger.warning("   --pinn-features requested but enrichment failed. Disabling PINN obs.")
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
        if args.mode == "simple-pipeline":
            results = simple_pipeline(df_drl, assets=args.assets)
        elif args.mode == "optuna-optimize":
            results = optuna_pipeline(
                df_drl,
                assets=args.assets,
                agent_type=args.agent_type,
                n_trials=args.n_trials,
            )
        elif args.mode == "rolling-ensemble":
            if ab_testing_enabled:
                logger.info("\n" + "="*50)
                logger.info("EXECUTING A/B TEST: Baseline vs PINN-Enhanced")
                logger.info("="*50)
                
                logger.info("\n>>>Running Group B: Baseline (No PINN)...")
                results_b = rolling_window_ensemble(
                    df_drl,
                    assets=args.assets,
                    pinn_engine=None,
                    pinn_features_enabled=False,
                    ab_testing_enabled=True,
                    results_prefix="group_B_baseline_"
                )
                
                logger.info("\n>>> Running Group A: Experiment (With PINN)...")
                
                results_a = rolling_window_ensemble(
                    df_drl,
                    assets=args.assets,
                    pinn_engine=None,
                    pinn_features_enabled=True,
                    ab_testing_enabled=True,
                    results_prefix="group_A_pinn_"
                )
                
                summary_b = results_b.get("aggregated", {})
                summary_a = results_a.get("aggregated", {})
                
                logger.info("\n" + "="*50)
                logger.info("A/B TEST QUICK SUMMARY")
                logger.info("="*50)
                logger.info(f"{'Metric':<20} | {'Baseline (B)':<15} | {'PINN (A)':<15} | {'Diff':<10}")
                logger.info("-" * 65)
                for metric in ["avg_ensemble_reward", "avg_ensemble_annual_return"]:
                    val_b = summary_b.get(metric, 0.0)
                    val_a = summary_a.get(metric, 0.0)
                    diff = val_a - val_b
                    logger.info(f"{metric:<20} | {val_b:>15.4f} | {val_a:>15.4f} | {diff:>+10.4f}")
                logger.info("="*50)
                
                results = {"group_a": results_a, "group_b": results_b}
                
            else:
                logger.info("\nExecuting Rolling Window Ensemble...")
                results = rolling_window_ensemble(
                    df_drl,
                    assets=args.assets,
                    pinn_engine=None,
                    pinn_features_enabled=pinn_features_enabled,
                    ab_testing_enabled=False
                )
        elif args.mode == "baseline-comparison":
            logger.info("\n" + "="*50)
            logger.info("EXECUTING BASELINE COMPARISON: DRL vs Benchmark (No PINN)")
            logger.info("="*50)
            
            results = rolling_window_ensemble(
                df_drl,
                assets=args.assets,
                pinn_engine=None,
                pinn_features_enabled=False,
                ab_testing_enabled=False,
                results_prefix="baseline_comparison_",
                minimal_plots=True
            )
        else:
            logger.error(f"Unknown mode: {args.mode}")
            raise ValueError(f"Unknown mode: {args.mode}")
        
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

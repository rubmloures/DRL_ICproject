"""
MARL Specialist Pipeline
=========================
Treinamento Hierarquico CTDE:
  - Fase 1: Agentes Especialistas (PPO por ativo) treinados em isolamento.
  - Fase 2: Coordenador (PortfolioCoordinator) distribui capital baseado
            nas Conviccoes dos Especialistas.

Hiperparametros: Herda de config/hyperparameters.py (PPO_PARAMS).
Callbacks: Rich Live Dashboard por agente.
Auditoria: CSV completo salvo ao final.
"""

import os
import sys
import io
import csv
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Force UTF-8 output so Rich works correctly on Windows
console = Console(highlight=False, force_terminal=True)

from src.data.data_loader import DataLoader
from src.data.data_processor import DataProcessor
from src.data.rolling_window import RollingWindowStrategy
from src.env.single_asset_env import SingleAssetTradingEnv
from src.agents.drl_agents import PPOAgent
from src.agents.coordinator import PortfolioCoordinator
from src.agents.unified_callbacks import UnifiedRichDashboard, StepAuditCallback, TrainingLossAuditCallback
from src.agents.curriculum import CurriculumCallback
from src.evaluation.marl_visualization import plot_marl_routing, plot_marl_convictions, plot_global_equity, plot_kmeans_clusters, plot_bnh_comparison, plot_regime_efficacy

# Optional PINN support imports
PINN_AVAILABLE = False
try:
    import torch
    from src.pinn.inference_wrapper import PINNInferenceEngine
    from src.data.pinn_data_preprocessor import PINNDataPreprocessor
    from config.config import PROJECT_ROOT
except ImportError:
    torch = None
    
from config.config import (
    PRIMARY_ASSETS, INITIAL_CAPITAL, RESULTS, DATA_RAW, 
    ROLLING_WINDOW_CONFIG, TRAIN_START, TRAIN_END, TEST_START, TEST_END, VAL_START, VAL_END
)
from config.hyperparameters import PPO_PARAMS, TRAINING_CONFIG

logging.basicConfig(
    level=logging.WARNING,   # Suppress verbose SB3/TF logs; Rich Dashboard takes over
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("marl_training.log", mode='w', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# MARL Config (overridable)
# ──────────────────────────────────────────────
MARL_CONFIG = {
    "assets": ["PETR4", "VALE3"],
    "train_cutoff": "2023-01-01",
    "tech_indicators": ["SMA_20", "RSI_14"],
    "include_pinn_features": True,
    # Specialist timesteps (set to 100_000+ for production)
    "specialist_timesteps": 50_000,
    # Rolling Window Config (Overrides config.py if needed, or uses it)
    "rolling_train_weeks": ROLLING_WINDOW_CONFIG.get('train_weeks', 52),
    "rolling_test_weeks": ROLLING_WINDOW_CONFIG.get('test_weeks', 12),
    "rolling_overlap": ROLLING_WINDOW_CONFIG.get('overlap_weeks', 0),
    "rolling_purge": ROLLING_WINDOW_CONFIG.get('purge_days', 0),
    "rolling_purge_kfold": ROLLING_WINDOW_CONFIG.get('purge_kfold_days', None),
    # Coordinator
    "coordinator_temperature": 2.0,
    "cash_buffer_penalty": 0.05,
    # Selic daily stub
    "selic_daily": 0.00025,
}

SPECIALIST_PPO_PARAMS = {
    **PPO_PARAMS,
    "verbose": 0,   # Rich Dashboard takes over stdout
    "batch_size": 256, # Increased from 64 to 256 for faster epoch processing on GPU
    "n_steps": 2048,
}


def _banner():
    console.print(Panel(
        Text("MARL Specialist Pipeline :: CTDE Architecture", style="bold white", justify="center"),
        subtitle="Hierarchical Orchestration | Centralized Training | Decentralized Execution",
        style="bold blue"
    ))


def _validate_data_integrity(df, assets):
    """QA Master Data Integrity Callback."""
    console.rule("[bold green]Data Integrity Check (QA Master)[/bold green]")
    from rich.table import Table
    
    table = Table(title="Dataset Summary per Asset")
    table.add_column("Asset", style="cyan")
    table.add_column("Rows", justify="right")
    table.add_column("Start Date", justify="center")
    table.add_column("End Date", justify="center")
    table.add_column("Missing (%)", justify="right")
    table.add_column("PINN Ready", justify="center")

    for asset in assets:
        asset_df = df[df['asset'] == asset]
        if asset_df.empty:
            table.add_row(asset, "0", "-", "-", "-", "[red]NO[/red]")
            continue
            
        start = asset_df['date'].min().strftime('%Y-%m-%d')
        end = asset_df['date'].max().strftime('%Y-%m-%d')
        missing_pct = (asset_df.isna().sum().sum() / (asset_df.size if asset_df.size > 0 else 1)) * 100
        pinn_cols = ['pinn_kappa', 'pinn_theta', 'pinn_xi', 'pinn_rho', 'pinn_nu']
        has_pinn = "[green]YES[/green]" if all(c in asset_df.columns for c in pinn_cols) else "[yellow]PARTIAL[/yellow]"
        
        table.add_row(
            asset, 
            f"{len(asset_df):,}", 
            start, 
            end, 
            f"{missing_pct:.2f}%",
            has_pinn
        )
    
    console.print(table)
    
    # Check if dates align with requested periods
    config_start = pd.to_datetime(TRAIN_START)
    data_start = df['date'].min()
    if data_start > config_start:
        console.print(f"[bold yellow]WARNING:[/bold yellow] Data starts at {data_start.date()}, which is after requested TRAIN_START {TRAIN_START}")
    else:
        console.print(f"[bold green]OK:[/bold green] Data covers requested period starting {TRAIN_START}")


def _load_robust_data(assets):
    """
    Robust Data Pipeline:
    1. Loads Raw Data (Options/Spot) for PINN/Heston inference.
    2. Prepares DRL Dataset (Daily OHLCV + Indicators).
    3. Enriches DRL Dataset with PINN Features (if enabled).
    """
    console.rule("[bold cyan]Step 1: Robust Data Loading & PINN Prep[/bold cyan]")
    
    loader = DataLoader(data_path=DATA_RAW)
    processor = DataProcessor()
    
    # 1. Load Raw Data per Asset
    raw_data_map = {}
    console.print(f"[cyan]>> Loading raw data for: {assets}[/cyan]")
    
    for asset in assets:
        try:
            df = loader.load_asset(asset)
            # Standardize date
            date_col = next((c for c in ['time', 'date', 'data'] if c in df.columns), None)
            if not date_col:
                continue
            df = df.rename(columns={date_col: 'date'})
            df['date'] = pd.to_datetime(df['date'])
            df['asset'] = asset
            raw_data_map[asset] = df
        except Exception as e:
            console.print(f"[red]Failed to load {asset}: {e}[/red]")

    # 2. Prepare DRL Dataset (Daily Aggregation)
    console.print("[cyan]>> Aggregating to Daily OHLCV + Indicators...[/cyan]")
    combined_drl = pd.DataFrame()
    
    for asset, df_raw in raw_data_map.items():
        # Robust aggregation logic ported from main.py
        # Handles Options Chain -> Daily Spot + Greeks Mean/Std
        
        daily_groups = []
        # Ensure date is datetime for grouping
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        
        for date, group in df_raw.groupby('date'):
            daily_record = {'date': date, 'asset': asset}
            
            # 1. Spot Price Extraction (Prioritize adjusted close)
            if 'acao_close_ajustado' in group.columns:
                daily_record['close'] = group['acao_close_ajustado'].iloc[0]
                daily_record['open'] = group['acao_open'].iloc[0] if 'acao_open' in group.columns else daily_record['close']
                daily_record['high'] = group['acao_high'].iloc[0] if 'acao_high' in group.columns else daily_record['close']
                daily_record['low'] = group['acao_low'].iloc[0] if 'acao_low' in group.columns else daily_record['close']
                daily_record['volume'] = group['acao_vol_fin'].iloc[0] if 'acao_vol_fin' in group.columns else 1.0
            elif 'spot_price' in group.columns:
                daily_record['close'] = group['spot_price'].iloc[-1]
                daily_record['open'] = group['spot_price'].iloc[0]
                daily_record['high'] = group['spot_price'].max()
                daily_record['low'] = group['spot_price'].min()
                daily_record['volume'] = len(group)
            else:
                # Fallback for simple CSVs
                daily_record['close'] = group.iloc[-1]['close'] if 'close' in group.columns else 0.0
                daily_record['open'] = group.iloc[0]['open'] if 'open' in group.columns else daily_record['close']
                daily_record['high'] = group['high'].max() if 'high' in group.columns else daily_record['close']
                daily_record['low'] = group['low'].min() if 'low' in group.columns else daily_record['close']
                daily_record['volume'] = group['volume'].sum() if 'volume' in group.columns else 0.0

            # 2. Greeks Aggregation (Mean & Std) - Critical for PINN/Regime context
            for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                if greek in group.columns:
                    daily_record[f'{greek}_mean'] = group[greek].mean()
                    daily_record[f'{greek}_std'] = group[greek].std()
            
            daily_groups.append(daily_record)
            
        df_daily = pd.DataFrame(daily_groups)
        
        # Ensure we have OHLC for indicators (backfill if missing)
        for col in ['open', 'high', 'low']:
            if col not in df_daily.columns:
                df_daily[col] = df_daily['close']

        # Add Tech Indicators
        df_daily = processor.clean_data(df_daily)
        df_daily = processor.add_technical_indicators(df_daily, include_indicators=MARL_CONFIG["tech_indicators"])
        
        combined_drl = pd.concat([combined_drl, df_daily], axis=0)

    # 3. PINN Enrichment (if enabled)
    if MARL_CONFIG["include_pinn_features"]:
        # We pass the DRL df to be enriched. 
        # Ideally, PINN uses raw data, but for this implementation we use the robust enrichment 
        # that handles the sliding window generation internally.
        combined_drl = _enrich_with_pinn_robust(combined_drl, assets)

    # Final formatting
    combined_drl = combined_drl.sort_values(['asset', 'date']).reset_index(drop=True)
    
    console.print(f"[green]OK Data Ready:[/green] {len(combined_drl)} rows total.")
    # Return dataframe and the name of the date column
    _validate_data_integrity(combined_drl, assets)
    return combined_drl, 'date'


def _enrich_with_pinn_robust(df, assets):
    """
    Applies PINN inference to generate Heston parameters (kappa, theta, xi, rho).
    Ported logic from main.py to ensure MARL agents see physics-informed features.
    """
    if not MARL_CONFIG["include_pinn_features"]:
        return df

    console.print("[bold magenta]>> PINN Enrichment: Initializing Inference Engine...[/bold magenta]")
    
    # Process per asset to maintain time-series continuity for LSTM windows
    enriched_dfs = []
    
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    checkpoint_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "best_model_weights.pth"
    stats_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "data_stats.json"

    for asset in assets:
        asset_df = df[df['asset'] == asset].copy().sort_values('date')
        if len(asset_df) < 30:
            enriched_dfs.append(asset_df)
            continue
            
        console.print(f"[magenta]   -> Inferring for {asset} using Specialist PINN weights...[/magenta]")
        
        try:
            # Initialize engine for THIS specific asset 
            # This triggers the specialist routing logic in inference_wrapper.py
            engine = PINNInferenceEngine(
                checkpoint_path=str(checkpoint_path),
                data_stats_path=str(stats_path),
                device=device,
                enable_validation=False,
                asset_name=asset,
                verbose=False
            )
            preprocessor = PINNDataPreprocessor(data_stats_path=engine.data_stats_path, verbose=False)
            
            # Adapt for preprocessor
            temp_df = asset_df.copy()
            close_col = next((c for c in ['close', 'acao_close_ajustado'] if c in temp_df.columns), 'close')
            temp_df['spot_price'] = temp_df[close_col]
            temp_df['strike'] = temp_df[close_col] # ATM assumption for regime sensing
            temp_df['days_to_maturity'] = 30
            temp_df['r_rate'] = 0.105 # Stub Selic
            temp_df['Dividend_Yield'] = 0.0
            temp_df['symbol'] = asset

            x_seq, x_phy, metadata = preprocessor.calculate_lstm_features(temp_df, window_size=30)
            results = engine.infer_heston_params(x_seq, x_phy, return_price=False)
            
            # Create result dataframe
            pinn_res = pd.DataFrame({
                'date': [m[0] for m in metadata],
                'pinn_kappa': results['kappa'].flatten(),
                'pinn_theta': results['theta'].flatten(),
                'pinn_xi': results['xi'].flatten(),
                'pinn_rho': results['rho'].flatten(),
                'pinn_nu': results['nu'].flatten(),
            })
            pinn_res['date'] = pd.to_datetime(pinn_res['date'])
            
            # Merge back to asset_df
            asset_df = pd.merge(asset_df, pinn_res, on='date', how='left')
            asset_df[['pinn_kappa', 'pinn_theta', 'pinn_xi', 'pinn_rho', 'pinn_nu']] = asset_df[['pinn_kappa', 'pinn_theta', 'pinn_xi', 'pinn_rho', 'pinn_nu']].fillna(0.0)
            
        except Exception as e:
            console.print(f"[red]Error inferring for {asset}: {e}[/red]")
        
        enriched_dfs.append(asset_df)

    return pd.concat(enriched_dfs).sort_values(['asset', 'date']).reset_index(drop=True)


def _enrich_with_pinn(df, assets):
    """Deprecated: Use _enrich_with_pinn_robust instead."""
    return _enrich_with_pinn_robust(df, assets)


def _filter_asset(df, asset):
    """Filter and clean data for a single asset."""
    asset_df = df[df['asset'] == asset].copy()
    asset_df.reset_index(drop=True, inplace=True)
    return asset_df


# ──────────────────────────────────────────────
# Phase 1: Train Specialist Agents
# ──────────────────────────────────────────────
def train_specialists(train_df, existing_agents=None, window_idx=0):
    """
    CTDE Phase 1: Train one PPO specialist per asset, in isolation.
    Supports Warm-Start (Fine-tuning) if existing_agents is provided.
    
    Each specialist sees ONLY its own asset data.
    """
    specialists = existing_agents if existing_agents else {}
    
    # Adjust timesteps: Full training for first window, reduced for fine-tuning
    current_timesteps = MARL_CONFIG["specialist_timesteps"] if window_idx == 0 else int(MARL_CONFIG["specialist_timesteps"] * 0.3)

    for asset in MARL_CONFIG["assets"]:
        console.rule(f"[bold yellow]Specialist Training :: {asset}[/bold yellow]")

        asset_train_df = _filter_asset(train_df, asset)

        if len(asset_train_df) < 100:
            console.print(
                f"[bold red]WARNING: Insufficient data for {asset} "
                f"({len(asset_train_df)} rows). Skipping.[/bold red]"
            )
            continue

        # Find close column for normalization check
        close_col = next((c for c in ['close', 'acao_close_ajustado'] if c in asset_train_df.columns), None)
        if close_col:
            close_mean = asset_train_df[close_col].mean()
            close_std = asset_train_df[close_col].std()
        else:
            close_mean = close_std = float('nan')

        sma_nan_pct = asset_train_df.get('SMA_20', pd.Series([np.nan])).isna().mean() * 100

        console.print(
            f"[cyan]{asset}[/cyan] -- "
            f"Training rows: {len(asset_train_df):,} | "
            f"Close mean: {close_mean:.4f} | std: {close_std:.4f} | "
            f"SMA_20 NaN: {sma_nan_pct:.1f}%"
        )

        # Build specialist environment
        env = SingleAssetTradingEnv(
            df=asset_train_df,
            asset_name=asset,
            initial_amount=INITIAL_CAPITAL,
            tech_indicator_list=MARL_CONFIG["tech_indicators"],
            include_pinn_features=MARL_CONFIG["include_pinn_features"]
        )

        # --- NEW: Fit K-Means Regime Detector for this asset/window ---
        if MARL_CONFIG["include_pinn_features"]:
            # Need columns 'nu', 'xi', 'rho' (mapped from 'pinn_nu', etc.)
            kmeans_features = asset_train_df.rename(columns={
                'pinn_nu': 'nu', 'pinn_xi': 'xi', 'pinn_rho': 'rho'
            })
            if all(col in kmeans_features.columns for col in ['nu', 'xi', 'rho']):
                console.print(f"[dim]   Fitting K-Means Regime Detector for {asset}...[/dim]")
                env.regime_detector.fit_kmeans(kmeans_features)
            else:
                console.print(f"[yellow]   Warning: PINN features missing for K-Means fitting on {asset}[/yellow]")

        # Build Unified Rich Dashboard
        rich_callback = UnifiedRichDashboard(
            name=f"Specialist: {asset}",
            total_timesteps=current_timesteps,
            refresh_rate=200,
            window_idx=window_idx
        )
        
        # Build Standard Step Audit (Log)
        audit_dir = os.path.join(RESULTS, "logs", "audit")
        step_logger = StepAuditCallback(
            filename=f"audit_specialist_{asset}_w{window_idx}.csv",
            log_dir=audit_dir
        )

        # Initialize or Update Agent
        if asset in specialists:
            console.print(f"[dim]Warm-starting existing agent for {asset} (Fine-tuning)...[/dim]")
            agent = specialists[asset]
            agent.model.set_env(env) # Update env to new window data
        else:
            console.print(f"[dim]Initializing new PPO agent for {asset}...[/dim]")
            params = {k: v for k, v in SPECIALIST_PPO_PARAMS.items() if k not in ('verbose',)}
            agent = PPOAgent(env=env, verbose=0, **params)

        # Curriculum Learning
        curriculum_cb = CurriculumCallback(total_timesteps=current_timesteps)
        
        # Loss Audit
        loss_audit_cb = TrainingLossAuditCallback(filename=f"loss_audit_specialist_{asset}_w{window_idx}.csv")

        # Train
        agent.train(total_timesteps=current_timesteps, custom_callbacks=[rich_callback, step_logger, curriculum_cb, loss_audit_cb])

        # Explicitly Save Model
        model_save_dir = os.path.join(RESULTS, "models")
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"marl_specialist_{asset}_{window_idx}.zip")
        agent.save(model_save_path)
        console.print(f"[dim]   [Model Saved] {os.path.basename(model_save_path)}[/dim]")
        
        specialists[asset] = agent

    mode_str = "Fine-Tuned" if window_idx > 0 else "Trained"
    console.rule(f"[bold green]All Specialists {mode_str} (Window {window_idx})[/bold green]")
    return specialists


# ──────────────────────────────────────────────
# Phase 2: MARL Coordinator Execution
# ──────────────────────────────────────────────
def run_coordinator(specialists, test_df, window_idx=0, initial_capital=INITIAL_CAPITAL):
    """
    CTDE Phase 2: Coordinator executes over test data.
    
    Collects Conviction Scores from specialists and routes capital
    using Softmax VDN. Logs a detailed audit trail per step.
    """
    console.rule(f"[bold blue]MARL Coordinator :: Execution Window {window_idx}[/bold blue]")

    coordinator = PortfolioCoordinator(
        assets=MARL_CONFIG["assets"],
        temperature=MARL_CONFIG["coordinator_temperature"],
        cash_buffer_penalty=MARL_CONFIG["cash_buffer_penalty"]
    )

    # Build test environments
    test_envs = {}
    for asset in MARL_CONFIG["assets"]:
        if asset not in specialists:
            continue
        asset_test_df = _filter_asset(test_df, asset)
        test_envs[asset] = SingleAssetTradingEnv(
            df=asset_test_df,
            asset_name=asset,
            initial_amount=INITIAL_CAPITAL,
            tech_indicator_list=MARL_CONFIG["tech_indicators"],
            include_pinn_features=MARL_CONFIG["include_pinn_features"]
        )

    active_assets = list(test_envs.keys())
    current_obs = {asset: test_envs[asset].reset()[0] for asset in active_assets}

    global_capital = initial_capital
    num_steps = min(len(test_df[test_df['asset'] == a]) for a in active_assets)

    # Tracking lists
    portfolio_history = []
    weights_history = []
    conviction_history = []
    audit_log = []

    selic = MARL_CONFIG["selic_daily"]

    for step in range(num_steps - 1):
        # A. Specialists emit Convictions (Decentralized)
        convictions = {}
        for asset in active_assets:
            action, _ = specialists[asset].predict(current_obs[asset], deterministic=True)
            convictions[asset] = float(action[0])

        # B. Coordinator routes capital (Centralized logic)
        target_weights = coordinator.route_capital(convictions)

        # C. Execute environment steps and collect returns
        step_returns = {}
        episode_done = {}
        env_infos = {}
        for asset in active_assets:
            obs, reward, terminal, truncated, info = test_envs[asset].step(np.array([convictions[asset]]))
            current_obs[asset] = obs
            step_returns[asset] = info.get('daily_return', 0.0)
            episode_done[asset] = bool(terminal or truncated)
            env_infos[asset] = info

        # D. Compute global portfolio return
        global_return = sum(target_weights.get(a, 0.0) * step_returns[a] for a in active_assets)
        global_return += target_weights.get('Cash', 0.0) * selic
        global_capital *= (1 + global_return)

        # E. Live log every 50 steps
        if step % 50 == 0:
            conviction_str = " | ".join(f"{a}={v:+.3f}" for a, v in convictions.items())
            weight_str = " | ".join(
                f"{a}={target_weights.get(a, 0.0)*100:.1f}%" for a in active_assets + ['Cash']
            )
            console.print(
                f"[dim]Step {step:>6d}[/dim]  "
                f"Capital: [bold green]${global_capital:,.2f}[/bold green]  "
                f"Return: [yellow]{global_return*100:+.4f}%[/yellow]  "
                f"Conv: {conviction_str}  Wts: {weight_str}"
            )

        # F. Append to lists
        portfolio_history.append(global_capital)
        weights_history.append(dict(target_weights))
        conviction_history.append(dict(convictions))

        # G. Audit row
        current_regime = "N/A"
        if active_assets:
            # Regime is global, so extracting from the first active asset's info is enough
            # We must pull it from the last executed step info
            # We already ran envs[asset].step and put info in an implicit variable... wait, info is overwritten in the loop. Let's get it.
            # actually we have step_returns but not the full info. Let's get regime from test_envs[active_assets[0]].current_regime
            current_regime = test_envs[active_assets[0]].current_regime.value if hasattr(test_envs[active_assets[0]], 'current_regime') else "N/A"

        audit_row = {
            "step": step,
            "window": window_idx,
            "global_capital": round(global_capital, 4),
            "global_return_pct": round(global_return * 100, 6),
            "cash_weight_pct": round(target_weights.get("Cash", 0.0) * 100, 4),
            "regime": current_regime
        }
        for asset in active_assets:
            audit_row[f"conviction_{asset}"]   = round(convictions[asset], 6)
            audit_row[f"weight_{asset}_pct"]   = round(target_weights.get(asset, 0.0) * 100, 4)
            audit_row[f"return_{asset}_pct"]   = round(step_returns[asset] * 100, 6)
            
            # --- Robust Feature Logging (QA Master request) ---
            # Prices
            info = env_infos.get(asset, {})
            raw_data = test_envs[asset].data.to_dict() if hasattr(test_envs[asset], 'data') else {}
            
            audit_row[f"price_{asset}"] = round(raw_data.get('close', raw_data.get('acao_close_ajustado', 0.0)), 4)
            
            # Heston Params
            heston = info.get('heston_params', {})
            for hp in ['nu', 'xi', 'rho', 'kappa', 'theta']:
                if hp in heston:
                    audit_row[f"pinn_{hp}_{asset}"] = round(heston[hp], 6)
            
            # Tech Indicators
            for col, val in raw_data.items():
                if any(tech in col for tech in ["SMA", "RSI", "MACD", "ATR"]):
                    audit_row[f"{col}_{asset}"] = round(val, 6) if isinstance(val, (int, float)) else val
        audit_log.append(audit_row)

        # H. Stop if all environments terminated
        if all(episode_done.values()):
            console.print("[bold yellow]All environments concluded. Stopping.[/bold yellow]")
            break

    console.rule(f"[bold green]Execution Complete -- Final Capital: ${global_capital:,.2f}[/bold green]")
    return portfolio_history, weights_history, conviction_history, audit_log, active_assets


# ──────────────────────────────────────────────
# Audit CSV Saver
# ──────────────────────────────────────────────
def save_audit_csv(audit_log, save_dir):
    """Save full audit trail to CSV for post-training analysis."""
    if not audit_log:
        return

    csv_path = os.path.join(save_dir, "marl_audit_trail.csv")
    fieldnames = list(audit_log[0].keys())

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_log)

    console.print(f"[bold cyan]Audit CSV saved:[/bold cyan] {csv_path}")
    console.print(f"   Rows: {len(audit_log):,} | Columns: {fieldnames}")


# ──────────────────────────────────────────────
# Main Orchestrator
# ──────────────────────────────────────────────
def run_marl_specialist_pipeline():
    _banner()

    # 1. Data loading and processing
    df_processed, date_col = _load_robust_data(MARL_CONFIG["assets"])
    
    # 2. Split Data into Train, Test and Validation based on config.py
    # Filter only relevant range [TRAIN_START, VAL_END]
    mask_all = (df_processed['date'] >= pd.to_datetime(TRAIN_START)) & (df_processed['date'] <= pd.to_datetime(VAL_END))
    df_active = df_processed[mask_all].copy()
    
    # 2.1 Initialize Rolling Window Strategy for the TRAINING + TEST segment
    # We use fixed ends if ROLLING_WINDOW_CONFIG['enabled'] is False, 
    # but here we follow the user's split:
    # Train: [TRAIN_START, TRAIN_END]
    # Test: [TEST_START, TEST_END]
    # Validation: [VAL_START, VAL_END]
    
    calendar_df = pd.DataFrame({'date': df_active['date'].unique()}).sort_values('date').reset_index(drop=True)
    calendar_df['data'] = calendar_df['date'] 
    
    # We use ROLLING strategy primarily for the Walk-Forward of Specialists until TEST_END
    # Filter calendar for Train + Test pool
    train_test_cal = calendar_df[calendar_df['date'] <= pd.to_datetime(TEST_END)].copy()
    
    rolling = RollingWindowStrategy(
        df=train_test_cal,
        train_weeks=MARL_CONFIG["rolling_train_weeks"],
        test_weeks=MARL_CONFIG["rolling_test_weeks"],
        overlap_weeks=MARL_CONFIG["rolling_overlap"],
        purge_days=MARL_CONFIG["rolling_purge"],
        purge_kfold_days=MARL_CONFIG["rolling_purge_kfold"]
    )
    
    # Extra check: If no windows generated (data too short), force a single window
    windows = list(rolling.generate_rolling_windows())
    if not windows:
        console.print("[yellow]Rolling window too short for parameters. Forcing single Train/Test split.[/yellow]")
        # Force a single split
        train_cal = train_test_cal[train_test_cal['date'] <= pd.to_datetime(TRAIN_END)]
        test_cal = train_test_cal[(train_test_cal['date'] >= pd.to_datetime(TEST_START)) & (train_test_cal['date'] <= pd.to_datetime(TEST_END))]
        windows = [(train_cal, test_cal, 0, {
            'train_start': train_cal['date'].min(), 'train_end': train_cal['date'].max(),
            'test_start': test_cal['date'].min(), 'test_end': test_cal['date'].max()
        })]

    # State persistence for Walk-Forward
    specialists = {} 
    
    # Aggregated results
    full_portfolio_history = []
    full_weights_history = []
    full_conviction_history = []
    full_audit_log = []
    active_assets = MARL_CONFIG["assets"]
    current_capital = INITIAL_CAPITAL

    # 3. Iterate Windows (Train -> Test)
    for train_cal, test_cal, window_idx, date_range in windows:
        console.print(f"\n[bold magenta]>>> Processing Window {window_idx}[/bold magenta]")
        console.print(f"    Train: {date_range['train_start'].date()} -> {date_range['train_end'].date()}")
        console.print(f"    Test:  {date_range['test_start'].date()} -> {date_range['test_end'].date()}")

        # Extract actual data using the calendar dates
        train_df = df_active[df_active['date'].isin(train_cal['date'])].copy()
        test_df = df_active[df_active['date'].isin(test_cal['date'])].copy()

        # Phase 1: Train/Fine-tune Specialists
        specialists = train_specialists(train_df, existing_agents=specialists, window_idx=window_idx)

        if not specialists:
            console.print("[bold red]No specialists trained. Aborting.[/bold red]")
            return

        # Phase 2: Coordinator Execution (Test Set)
        p_hist, w_hist, c_hist, audit, _ = run_coordinator(
            specialists, test_df, window_idx=window_idx, initial_capital=current_capital
        )

        # Update capital for next window
        if p_hist:
            current_capital = p_hist[-1]
            
        full_portfolio_history.extend(p_hist) 
        full_weights_history.extend(w_hist)
        full_conviction_history.extend(c_hist)
        full_audit_log.extend(audit)

    # 3.5 FINAL VALIDATION PHASE (H2 2025)
    console.rule("[bold red]FINAL VALIDATION PHASE :: Blind Hold-Out (H2 2025)[/bold red]")
    val_df = df_active[df_active['date'] >= pd.to_datetime(VAL_START)].copy()
    
    if not val_df.empty:
        p_hist_v, w_hist_v, c_hist_v, audit_v, _ = run_coordinator(
            specialists, val_df, window_idx=999, initial_capital=current_capital
        )
        
        full_portfolio_history.extend(p_hist_v)
        full_weights_history.extend(w_hist_v)
        full_conviction_history.extend(c_hist_v)
        full_audit_log.extend(audit_v)
        
        console.print(f"[bold red]Validation Complete. Final Capital: ${p_hist_v[-1]:,.2f}[/bold red]")
    else:
        console.print("[yellow]No validation data available for the specified range.[/yellow]")

    # 4. Save outputs (Global)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(RESULTS, "plots", f"marl_run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)


    # Plots
    plot_marl_routing(full_weights_history, active_assets, save_dir)
    plot_marl_convictions(full_conviction_history, active_assets, save_dir)
    plot_global_equity(full_portfolio_history, save_dir)
    
    # Robust Buy and Hold Baseline from Audit Log
    bnh_equity = [INITIAL_CAPITAL]
    for row in full_audit_log:
        avg_ret = sum(row.get(f"return_{a}_pct", 0.0) for a in active_assets) / max(len(active_assets), 1) / 100.0
        bnh_equity.append(bnh_equity[-1] * (1 + avg_ret))
    bnh_equity = bnh_equity[1:] # match length
    plot_bnh_comparison(full_portfolio_history, bnh_equity, save_dir)
    
    plot_regime_efficacy(full_audit_log, save_dir)
    
    pinn_check_cols = ['pinn_nu', 'pinn_xi', 'pinn_rho']
    if MARL_CONFIG["include_pinn_features"] and all(c in df_processed.columns for c in pinn_check_cols):
        try:
            console.print("[cyan]>> Generating PINN Regime Clusters (K-Means)...[/cyan]")
            from src.reward.regime_detector import RegimeDetector
            rd = RegimeDetector()
            # Rename columns to what RegimeDetector expects
            pinn_data = df_processed.rename(columns={'pinn_nu': 'nu', 'pinn_xi': 'xi', 'pinn_rho': 'rho'})
            rd.fit_kmeans(pinn_data)
            plot_kmeans_clusters(pinn_data, rd.kmeans, rd.scaler, save_dir)
        except Exception as e:
            console.print(f"[yellow]Could not generate KMeans regime plots: {e}[/yellow]")
    else:
        console.print("[dim]PINN columns missing or feature disabled - skipping KMeans plot.[/dim]")

    # Audit trail
    save_audit_csv(full_audit_log, save_dir)
    
    # Heston Param Audit (Merged Inputs + PINN Params)
    heston_audit_path = os.path.join(save_dir, "heston_audit.csv")
    df_processed.to_csv(heston_audit_path, index=False)
    console.print(f"[bold cyan]Heston Audit CSV saved:[/bold cyan] {heston_audit_path}")

    console.print(Panel(
        f"[bold green]MARL Pipeline Complete[/bold green]\n"
        f"Output directory: [cyan]{save_dir}[/cyan]\n"
        f"Plots: marl_capital_routing.html | marl_specialist_convictions.html | marl_global_equity.html | marl_bnh_comparison.html | kmeans_regime_clusters.html | marl_regime_efficacy.html\n"
        f"Audit: marl_audit_trail.csv ({len(full_audit_log):,} rows)",
        title="[bold white]Run Summary[/bold white]",
        border_style="green"
    ))


if __name__ == "__main__":
    run_marl_specialist_pipeline()

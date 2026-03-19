"""
Parallel Asset-Specific Ensemble Architecture
==============================================
Trains independent Ensemble (PPO + A2C + DDPG) agents for specific assets 
in parallel processes.

Usage:
    python main_parallel_ensemble.py --assets PETR4 VALE3 ABEV3
"""

import os
import sys
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
import logging

import multiprocessing as mp
# Deve ser chamado antes de qualquer outra operação com multiprocessing
#mp.set_start_method('spawn', force=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import load_raw_options_data, prepare_pinn_dataset, prepare_drl_dataset, rolling_window_ensemble
from src.data.data_processor import DataProcessor
from src.agents.unified_callbacks import ParallelMultiAssetDashboard
from config.config import RESULTS, DATA_RAW

def train_asset_ensemble(asset_name, pinn_features_enabled=True, monitor_queue=None):
    """Worker function to train an ensemble for a single asset."""
    log_file = f"parallel_{asset_name}_training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='w', encoding='utf-8')]
    )
    logger = logging.getLogger(f"Parallel_{asset_name}")
    logger.info(f"Starting parallel ensemble training for {asset_name}")

    try:
        # 1. Load and process data for THIS asset only
        options_data_map = load_raw_options_data(assets=[asset_name])
        if not options_data_map:
            logger.error(f"No data found for {asset_name}")
            return False
            
        processor = DataProcessor()
        df_drl, _ = prepare_drl_dataset(options_data_map, processor)
        
        # 2. Setup PINN engine if enabled
        pinn_engine = None
        if pinn_features_enabled:
            try:
                from src.pinn.inference_wrapper import PINNInferenceEngine
                from config.config import PROJECT_ROOT
                checkpoint_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "best_model_weights.pth"
                stats_path = PROJECT_ROOT / "src" / "pinn" / "weights" / "data_stats.json"
                
                # Use specialist weights if available (logic inside PINNInferenceEngine handles this via asset_name)
                pinn_engine = PINNInferenceEngine(
                    checkpoint_path=str(checkpoint_path),
                    data_stats_path=str(stats_path),
                    asset_name=asset_name,
                    verbose=False
                )
                
                # --- PRECOMPUTE PINN FEATURES (Batch Inference) ---
                # Notify dashboard of pre-computation phase
                if monitor_queue:
                    monitor_queue.put({"name": asset_name, "status": "Precomputing Physics..."})

                # Fixes K-Means 'Could not find PINN features' warning & live-inference overhead
                import pandas as pd
                import numpy as np
                from src.data.pinn_data_preprocessor import PINNDataPreprocessor
                
                logger.info(f"Precomputing PINN features for {asset_name} via Batch Inference...")
                preprocessor = PINNDataPreprocessor(data_stats_path=pinn_engine.data_stats_path if hasattr(pinn_engine, 'data_stats_path') else None, verbose=False)
                temp_df = df_drl.copy()
                
                # --- MAP OHLCV COLUMNS (Expected by preprocessor) ---
                close_col = next((c for c in ['stock_0_close', 'stock_0_acao_close_ajustado', 'close'] if c in temp_df.columns), None)
                high_col = next((c for c in ['stock_0_high', 'high'] if c in temp_df.columns), None)
                low_col = next((c for c in ['stock_0_low', 'low'] if c in temp_df.columns), None)
                vol_col = next((c for c in ['stock_0_volume', 'volume', 'acao_vol_fin'] if c in temp_df.columns), None)
                
                if close_col:
                    # FORCE NUMERIC: Critical to prevent '1704931200.0 (date)' as price
                    valid_close = pd.to_numeric(temp_df[close_col], errors='coerce').ffill().fillna(0.0)
                    temp_df['close'] = valid_close
                    temp_df['spot_price'] = valid_close
                    temp_df['strike'] = valid_close # ATM assumption
                if high_col: temp_df['high'] = pd.to_numeric(temp_df[high_col], errors='coerce').ffill().fillna(0.0)
                if low_col: temp_df['low'] = pd.to_numeric(temp_df[low_col], errors='coerce').ffill().fillna(0.0)
                if vol_col: temp_df['volume'] = pd.to_numeric(temp_df[vol_col], errors='coerce').ffill().fillna(0.0)
                
                temp_df['days_to_maturity'] = 30
                temp_df['r_rate'] = 0.105 # Default SELIC proxy
                temp_df['Dividend_Yield'] = 0.0
                temp_df['symbol'] = asset_name
                
                # Need datetimes for sequence assembly
                temp_df['date'] = pd.to_datetime(temp_df['date'])
                
                x_seq, x_phy, metadata = preprocessor.calculate_lstm_features(temp_df, window_size=30)
                
                if len(metadata) > 0:
                    # Notify progress of inference
                    if monitor_queue:
                         monitor_queue.put({"name": asset_name, "status": f"Inference ({len(metadata)} days)..."})
                         
                    results = pinn_engine.infer_heston_params(x_seq, x_phy, return_price=False)
                    
                    pinn_res = pd.DataFrame({
                        'date': [m[0] for m in metadata],
                        'stock_0_pinn_kappa': results['kappa'].flatten(),
                        'stock_0_pinn_theta': results['theta'].flatten(),
                        'stock_0_pinn_xi': results['xi'].flatten(),
                        'stock_0_pinn_rho': results['rho'].flatten(),
                        'stock_0_pinn_nu': results['nu'].flatten(),
                    })
                    pinn_res['date'] = pd.to_datetime(pinn_res['date'])
                    
                    df_drl['date'] = pd.to_datetime(df_drl['date'])
                    df_drl = pd.merge(df_drl, pinn_res, on='date', how='left')
                    
                    pinn_cols = ['stock_0_pinn_kappa', 'stock_0_pinn_theta', 'stock_0_pinn_xi', 'stock_0_pinn_rho', 'stock_0_pinn_nu']
                    # Use standard methods instead of deprecated 'method='
                    df_drl[pinn_cols] = df_drl[pinn_cols].bfill().fillna(0.0)
                    logger.info(f"Successfully appended {len(pinn_cols)} PINN features for {asset_name}.")
                
            except Exception as e:
                logger.warning(f"Could not initialize PINN for {asset_name}: {e}")

        # 3. Run Rolling Window Ensemble
        results_prefix = f"parallel_{asset_name}_"
        rolling_window_ensemble(
            df=df_drl,
            assets=[asset_name],
            pinn_engine=pinn_engine,
            pinn_features_enabled=pinn_features_enabled,
            results_prefix=results_prefix,
            monitor_queue=monitor_queue
        )
        
        logger.info(f"Successfully completed ensemble training for {asset_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error during parallel training for {asset_name}: {e}", exc_info=True)
        return False

def main():
    parser = argparse.ArgumentParser(description="Parallel Asset-Specific Ensemble Training")
    parser.add_argument("--assets", nargs="+", default=["PETR4", "VALE3", "ABEV3"], help="Assets to train")
    parser.add_argument("--no-pinn", action="store_true", help="Disable PINN features")
    args = parser.parse_args()

    assets = args.assets
    pinn_enabled = not args.no_pinn

    print(f"Launching parallel ensemble training for: {assets}")
    print(f"   PINN Features: {'Enabled' if pinn_enabled else 'Disabled'}")
    print(f"   Dashboard: [Layout A] Multi-Asset Command Center Enabled\n")
    
    # Setup Communications and Dashboard
    ctx = mp.get_context('spawn')
    monitor_queue = ctx.Queue()
    dashboard = ParallelMultiAssetDashboard(assets)
    
    processes = []
    for asset in assets:
        p = ctx.Process(target=train_asset_ensemble, args=(asset, pinn_enabled, monitor_queue))
        p.start()
        processes.append(p)
        print(f"   [Started] Process for {asset} (PID: {p.pid})")

    # Master Monitor Loop
    dashboard.start()
    try:
        active_processes = len(processes)
        while active_processes > 0:
            # Consume available data in the queue
            # Increased efficiency: process up to 50 messages per loop
            for _ in range(50):
                import queue
                try:
                    data = monitor_queue.get_nowait()
                    dashboard.update_state(data)
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"\n[Error Dashboard Update] {e}\n")
            
            # Refresh dashboard display
            try:
                dashboard.refresh()
            except Exception:
                pass
            
            # Real health check
            active_processes = sum(1 for p in processes if p.is_alive())
            import time
            time.sleep(0.5) # Increased frequency from 1.0s to 0.5s
            
    except KeyboardInterrupt:
        print("\n[Terminating] Parallel training interrupted by user...")
        for p in processes:
            p.terminate()
    finally:
        dashboard.stop()
        
    print("\nAll parallel training processes completed.")
    print(f"   Check results in: {RESULTS}")

if __name__ == "__main__":
    # Multiprocessing on Windows requires this
    multiprocessing.freeze_support()
    main()

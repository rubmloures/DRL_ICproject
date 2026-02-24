import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.env.stock_env_b3 import StockTradingEnv
from src.reward.composite_reward import CompositeRewardCalculator
from src.optimization.hyperparameter_optimizer import run_all_optimizers


def verify_selic():
    logger.info("--- Verifying SELIC Dynamic Rate ---")
    calc = CompositeRewardCalculator(taxa_selic_path="data/raw/taxa_selic.csv")
    
    # Test specific dates if the dataframe loaded
    if calc.selic_df is not None:
        test_date1 = pd.to_datetime('2025-08-11')
        rate1 = calc._get_daily_selic(test_date1)
        logger.info(f"Rate for {test_date1.date()}: {rate1:.6f}")
        
        test_date2 = pd.to_datetime('2023-01-01')
        rate2 = calc._get_daily_selic(test_date2)
        logger.info(f"Rate for {test_date2.date()}: {rate2:.6f}")
    else:
        logger.error("SELIC dataframe did not load properly.")


def verify_pinn_normalization():
    logger.info("\n--- Verifying PINN Normalization ---")
    # Create dummy data
    dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "0_close": np.random.normal(20, 1, 10),
        "pinn_nu": np.random.uniform(0.1, 0.5, 10),
        "pinn_theta": np.random.uniform(0.1, 0.5, 10),
        "pinn_kappa": np.random.uniform(1.0, 5.0, 10),
        "pinn_xi": np.random.uniform(0.1, 0.5, 10),
        "pinn_rho": np.random.uniform(-0.9, 0.9, 10),
    }).set_index("date")
    
    env = StockTradingEnv(
        df=df,
        stock_dim=1,
        include_pinn_features=True
    )
    
    # Force the env to process a few row to build history
    features_list = []
    for i in range(5):
        env.day = i
        env.data = env.df.iloc[i]
        feats = env._get_pinn_features()
        features_list.append(feats)
        logger.info(f"Day {i} PINN Features (Normalized/Clipped): {feats}")

def verify_optimizer():
    logger.info("\n--- Verifying Hyperparameter Optimizer Output ---")
    
    def make_dummy_env():
        dates = pd.date_range(start="2024-01-01", periods=10, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "0_close": np.random.normal(20, 1, 10),
        }).set_index("date")
        return StockTradingEnv(df=df, stock_dim=1, initial_amount=100000)
    
    # Only run 1 trial per agent to see if it saves the file
    run_all_optimizers(env_fn=make_dummy_env, n_trials=1, results_dir="results_test")
    
    result_file = Path("results_test/best_hyperparameter.txt")
    if result_file.exists():
        logger.info(f"File created successfully at {result_file}")
        with open(result_file, 'r') as f:
            logger.info("File contents preview:")
            logger.info(f.read(200) + "...")
    else:
        logger.error("Failed to create best_hyperparameter.txt")


if __name__ == "__main__":
    verify_selic()
    verify_pinn_normalization()
    verify_optimizer()

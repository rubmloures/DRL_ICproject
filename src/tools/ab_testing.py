"""
A/B Testing Framework for PINN Integration
===========================================
Compares DRL Ensemble performance with and without PINN features
using statistical rigor and proper cross-validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from dataclasses import dataclass
import logging
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Container for A/B test results."""
    window_idx: int
    fold_idx: Optional[int]
    model_a_sharpe: float
    model_b_sharpe: float
    model_a_returns: float
    model_b_returns: float
    model_a_max_dd: float
    model_b_max_dd: float
    improvement_pct: float
    statistically_significant: bool
    p_value: float
    n_episodes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'window_idx': self.window_idx,
            'fold_idx': self.fold_idx,
            'model_a_sharpe': self.model_a_sharpe,
            'model_b_sharpe': self.model_b_sharpe,
            'improvement_pct': self.improvement_pct,
            'statistically_significant': self.statistically_significant,
            'p_value': self.p_value,
            'model_a_returns': self.model_a_returns,
            'model_b_returns': self.model_b_returns,
            'model_a_max_dd': self.model_a_max_dd,
            'model_b_max_dd': self.model_b_max_dd,
        }


class ABTestingFramework:
    """
    A/B Testing framework for comparing DRL ensemble with/without PINN.
    
    Model A: Ensemble (PPO+DDPG+A2C) WITHOUT PINN features
    Model B: Ensemble (PPO+DDPG+A2C) WITH PINN features
    """
    
    def __init__(
        self,
        assets: List[str],
        statistical_test: str = 'ttest',
        significance_level: float = 0.05,
        use_same_seeds: bool = True,
        verbose: bool = True
    ):
        """
        Initialize A/B testing framework.
        
        Args:
            assets: List of assets to trade
            statistical_test: 'ttest' or 'mann_whitney'
            significance_level: Alpha for hypothesis test
            use_same_seeds: Use same random seeds for fair comparison
            verbose: Print logging info
        """
        self.assets = assets
        self.statistical_test = statistical_test
        self.significance_level = significance_level
        self.use_same_seeds = use_same_seeds
        self.verbose = verbose
        
        self.results: List[ABTestResult] = []
    
    def train_model_a_no_pinn(
        self,
        env,
        ensemble_config: Dict[str, Any],
        total_timesteps: int = 100_000,
        seed: Optional[int] = None
    ):
        """
        Train DRL ensemble WITHOUT PINN features (Model A).
        
        Args:
            env: StockTradingEnv with pinn_engine=None
            ensemble_config: Configuration for ensemble
            total_timesteps: Training steps
            seed: Random seed for reproducibility
        
        Returns:
            Trained ensemble agent
        """
        try:
            from src.agents import EnsembleAgent, PPOAgent, DDPGAgent, A2CAgent
            
            if self.verbose:
                logger.info("Training Model A (NO PINN)...")
            
            # Create individual agents
            agents = {
                'PPO': PPOAgent(env, policy='MlpPolicy', seed=seed),
                'DDPG': DDPGAgent(env, policy='MlpPolicy', seed=seed),
                'A2C': A2CAgent(env, policy='MlpPolicy', seed=seed)
            }
            
            # Create ensemble
            ensemble = EnsembleAgent(
                env,
                agents,
                voting_strategy=ensemble_config.get('voting_strategy', 'weighted')
            )
            
            # Train
            ensemble.train(total_timesteps=total_timesteps)
            
            if self.verbose:
                logger.info("Model A training complete")
            
            return ensemble
        
        except Exception as e:
            logger.error(f"Model A training failed: {e}")
            raise
    
    def train_model_b_with_pinn(
        self,
        env,
        ensemble_config: Dict[str, Any],
        total_timesteps: int = 100_000,
        seed: Optional[int] = None
    ):
        """
        Train DRL ensemble WITH PINN features (Model B).
        
        Args:
            env: StockTradingEnv with pinn_engine active
            ensemble_config: Configuration for ensemble
            total_timesteps: Training steps
            seed: Random seed for reproducibility
        
        Returns:
            Trained ensemble agent
        """
        try:
            from src.agents import EnsembleAgent, PPOAgent, DDPGAgent, A2CAgent
            
            if self.verbose:
                logger.info("Training Model B (WITH PINN)...")
            
            # Create individual agents
            agents = {
                'PPO': PPOAgent(env, policy='MlpPolicy', seed=seed),
                'DDPG': DDPGAgent(env, policy='MlpPolicy', seed=seed),
                'A2C': A2CAgent(env, policy='MlpPolicy', seed=seed)
            }
            
            # Create ensemble
            ensemble = EnsembleAgent(
                env,
                agents,
                voting_strategy=ensemble_config.get('voting_strategy', 'weighted')
            )
            
            # Train
            ensemble.train(total_timesteps=total_timesteps)
            
            if self.verbose:
                logger.info("Model B training complete")
            
            return ensemble
        
        except Exception as e:
            logger.error(f"Model B training failed: {e}")
            raise
    
    def evaluate_and_compare(
        self,
        ensemble_a,
        ensemble_b,
        test_env,
        window_idx: int,
        fold_idx: Optional[int] = None,
        n_episodes: int = 10
    ) -> ABTestResult:
        """
        Evaluate both ensembles and compare statistically.
        
        Args:
            ensemble_a: Model A (no PINN)
            ensemble_b: Model B (with PINN)
            test_env: Test environment
            window_idx: Rolling window index
            fold_idx: Fold index (if using K-fold)
            n_episodes: Number of evaluation episodes
        
        Returns:
            ABTestResult with comparison metrics
        """
        
        if self.verbose:
            logger.info(f"Evaluating Window {window_idx}, Fold {fold_idx}...")
        
        # Evaluate Model A
        try:
            rewards_a, _ = ensemble_a.evaluate(
                n_episodes=n_episodes,
                env=test_env,
                deterministic=False
            )
            sharpe_a = self._compute_sharpe_ratio(rewards_a)
            returns_a = np.mean(rewards_a)
            dd_a = self._compute_max_drawdown(rewards_a)
        except Exception as e:
            logger.warning(f"Model A evaluation failed: {e}, using zeros")
            rewards_a = np.zeros(n_episodes)
            sharpe_a = 0.0
            returns_a = 0.0
            dd_a = 0.0
        
        # Evaluate Model B
        try:
            rewards_b, _ = ensemble_b.evaluate(
                n_episodes=n_episodes,
                env=test_env,
                deterministic=False
            )
            sharpe_b = self._compute_sharpe_ratio(rewards_b)
            returns_b = np.mean(rewards_b)
            dd_b = self._compute_max_drawdown(rewards_b)
        except Exception as e:
            logger.warning(f"Model B evaluation failed: {e}, using zeros")
            rewards_b = np.zeros(n_episodes)
            sharpe_b = 0.0
            returns_b = 0.0
            dd_b = 0.0
        
        # Statistical test
        is_significant, p_value = self._statistical_test(rewards_a, rewards_b)
        
        # Compute improvement
        improvement_pct = (
            (sharpe_b - sharpe_a) / (abs(sharpe_a) + 1e-8) * 100
        )
        
        result = ABTestResult(
            window_idx=window_idx,
            fold_idx=fold_idx,
            model_a_sharpe=sharpe_a,
            model_b_sharpe=sharpe_b,
            model_a_returns=returns_a,
            model_b_returns=returns_b,
            model_a_max_dd=dd_a,
            model_b_max_dd=dd_b,
            improvement_pct=improvement_pct,
            statistically_significant=is_significant,
            p_value=p_value,
            n_episodes=n_episodes
        )
        
        self.results.append(result)
        
        if self.verbose:
            logger.info(
                f"  Model A Sharpe: {sharpe_a:.3f} | "
                f"Model B Sharpe: {sharpe_b:.3f} | "
                f"Improvement: {improvement_pct:+.1f}% "
                f"(p={p_value:.4f})"
            )
        
        return result
    
    def _compute_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """Compute Sharpe ratio from returns."""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        sharpe = sharpe * np.sqrt(periods_per_year)
        
        return sharpe
    
    def _compute_max_drawdown(self, returns: np.ndarray) -> float:
        """Compute maximum drawdown from returns."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        
        return max_dd
    
    def _statistical_test(
        self,
        rewards_a: np.ndarray,
        rewards_b: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Perform statistical test to check if Model B significantly outperforms Model A.
        
        Returns:
            (is_significant: bool, p_value: float)
        """
        if self.statistical_test == 'ttest':
            # Paired t-test (same episodes)
            t_stat, p_value = sp_stats.ttest_rel(rewards_b, rewards_a)
        elif self.statistical_test == 'mann_whitney':
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = sp_stats.mannwhitneyu(rewards_b, rewards_a, alternative='greater')
        else:
            raise ValueError(f"Unknown statistical test: {self.statistical_test}")
        
        is_significant = p_value < self.significance_level
        
        return is_significant, p_value
    
    def aggregate_results(self) -> pd.DataFrame:
        """
        Aggregate all A/B test results into a summary table.
        
        Returns:
            DataFrame with summary metrics
        """
        if not self.results:
            logger.warning("⚠️ No results to aggregate")
            return pd.DataFrame()
        
        # Convert to DataFrame
        results_list = [r.to_dict() for r in self.results]
        df = pd.DataFrame(results_list)
        
        return df
    
    def print_summary_report(self):
        """Print formatted summary report of A/B testing."""
        if not self.results:
            logger.warning("No results to report")
            return
        
        df = self.aggregate_results()
        
        # Compute overall statistics
        mean_improvement = df['improvement_pct'].mean()
        std_improvement = df['improvement_pct'].std()
        min_improvement = df['improvement_pct'].min()
        max_improvement = df['improvement_pct'].max()
        n_significant = df['statistically_significant'].sum()
        
        report = "\n" + "=" * 80 + "\n"
        report += "A/B TESTING SUMMARY: Model A (NO PINN) vs Model B (WITH PINN)\n"
        report += "=" * 80 + "\n\n"
        
        # Overall metrics
        report += "OVERALL RESULTS:\n"
        report += f"  Mean Improvement:           {mean_improvement:+.2f}%\n"
        report += f"  Std Dev Improvement:        {std_improvement:.2f}%\n"
        report += f"  Min Improvement:            {min_improvement:+.2f}%\n"
        report += f"  Max Improvement:            {max_improvement:+.2f}%\n"
        report += f"  Statistically Significant:  {n_significant}/{len(df)} "
        report += f"({n_significant/len(df)*100:.1f}%)\n\n"
        
        # Mean Sharpe comparison
        mean_sharpe_a = df['model_a_sharpe'].mean()
        mean_sharpe_b = df['model_b_sharpe'].mean()
        report += "SHARPE RATIO COMPARISON:\n"
        report += f"  Model A (NO PINN):  {mean_sharpe_a:.3f}\n"
        report += f"  Model B (WITH PINN): {mean_sharpe_b:.3f}\n"
        report += f"  Difference:         {mean_sharpe_b - mean_sharpe_a:+.3f}\n\n"
        
        # Mean Returns
        mean_returns_a = df['model_a_returns'].mean()
        mean_returns_b = df['model_b_returns'].mean()
        report += "AVERAGE RETURNS COMPARISON:\n"
        report += f"  Model A (NO PINN):  {mean_returns_a:.4f}\n"
        report += f"  Model B (WITH PINN): {mean_returns_b:.4f}\n"
        report += f"  Difference:         {mean_returns_b - mean_returns_a:+.4f}\n\n"
        
        # Detail table
        report += "PER WINDOW DETAILS:\n"
        report += df[[
            'window_idx', 'fold_idx', 'model_a_sharpe', 'model_b_sharpe',
            'improvement_pct', 'statistically_significant'
        ]].to_string(index=False) + "\n"
        
        report += "=" * 80 + "\n"
        
        print(report)
        logger.info(report)
    
    def save_results(self, filepath: str):
        """Save A/B test results to CSV."""
        df = self.aggregate_results()
        df.to_csv(filepath, index=False)
        if self.verbose:
            logger.info(f"A/B test results saved to {filepath}")

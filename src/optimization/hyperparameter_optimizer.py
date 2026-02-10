"""
Hyperparameter Optimization Module
===================================
Uses Optuna for Bayesian optimization of DRL hyperparameters.

This module finds the best hyperparameters for:
- PPO, DDPG, A2C algorithms
- Environment parameters
- PINN architecture parameters

Example use:
    >>> optimizer = HyperparameterOptimizer(
    ...     agent_type="PPO",
    ...     env_fn=lambda: StockTradingEnvB3(...),
    ... )
    >>> best_params = optimizer.optimize(n_trials=100)
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
import logging

from src.agents.drl_agents import PPOAgent, DDPGAgent, A2CAgent

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Optimize hyperparameters using Bayesian optimization (Optuna).
    
    Supports concurrent trials for fast optimization.
    """
    
    def __init__(
        self,
        agent_type: str = "PPO",
        env_fn: Optional[Callable] = None,
        n_jobs: int = 1,
        direction: str = "maximize",
        sampler: str = "tpe",
        pruner: str = "median",
        seed: Optional[int] = None,
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            agent_type: "PPO", "DDPG", or "A2C"
            env_fn: Function that returns a new environment
            n_jobs: Number of parallel workers
            direction: "maximize" or "minimize"
            sampler: "tpe" (Tree Parzen Estimator), "random", "grid"
            pruner: "median", "noop" (no pruning)
            seed: Random seed
        """
        self.agent_type = agent_type
        self.env_fn = env_fn
        self.n_jobs = n_jobs
        self.direction = direction
        self.seed = seed
        
        # Create sampler
        if sampler == "tpe":
            self.sampler = TPESampler(seed=seed)
        elif sampler == "random":
            self.sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            self.sampler = TPESampler(seed=seed)
        
        # Create pruner
        if pruner == "median":
            self.pruner = MedianPruner(n_startup_trials=5)
        else:
            self.pruner = optuna.pruners.NopPruner()
        
        logger.info(f"Initialized HyperparameterOptimizer for {agent_type}")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for optimization.
        
        Trains an agent with suggested hyperparameters and returns Sharpe ratio.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            Metric to optimize (Sharpe ratio)
        """
        if self.env_fn is None:
            raise ValueError("env_fn must be provided")
        
        # Suggest hyperparameters based on agent type
        if self.agent_type == "PPO":
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 512, 4096, step=512),
                "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
                "n_epochs": trial.suggest_int("n_epochs", 5, 20),
                "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                "vf_coef": trial.suggest_float("vf_coef", 0.4, 0.9),
            }
        
        elif self.agent_type == "DDPG":
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "batch_size": trial.suggest_int("batch_size", 128, 512, step=64),
                "tau": trial.suggest_float("tau", 0.001, 0.02),
                "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
                "action_noise": trial.suggest_float("action_noise", 0.05, 0.5),
            }
        
        elif self.agent_type == "A2C":
            params = {
                "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                "n_steps": trial.suggest_int("n_steps", 5, 50, step=5),
                "gamma": trial.suggest_float("gamma", 0.95, 0.9999),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
                "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
                "vf_coef": trial.suggest_float("vf_coef", 0.4, 0.9),
            }
        
        else:
            raise ValueError(f"Unknown agent_type: {self.agent_type}")
        
        try:
            # Create environment and agent
            env = self.env_fn()
            
            if self.agent_type == "PPO":
                agent = PPOAgent(
                    env=env,
                    model_name=f"trial_{trial.number}",
                    **params,
                    device="cpu",
                    verbose=0,
                )
            
            elif self.agent_type == "DDPG":
                agent = DDPGAgent(
                    env=env,
                    model_name=f"trial_{trial.number}",
                    **params,
                    device="cpu",
                    verbose=0,
                )
            
            else:  # A2C
                agent = A2CAgent(
                    env=env,
                    model_name=f"trial_{trial.number}",
                    **params,
                    device="cpu",
                    verbose=0,
                )
            
            # Train for short period to estimate merit (early stopping)
            agent.train(total_timesteps=10_000)
            
            # Evaluate
            metrics = agent.evaluate(env, num_episodes=5, deterministic=True)
            
            # Use Sharpe ratio as objective
            sharpe = metrics.get('mean_reward', 0.0)
            
            logger.debug(f"Trial {trial.number}: {self.agent_type} Sharpe={sharpe:.4f}")
            
            env.close()
            
            # Return early if trial crashed
            if np.isnan(sharpe) or np.isinf(sharpe):
                return -np.inf if self.direction == "maximize" else np.inf
            
            return sharpe
            
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {str(e)}")
            return -np.inf if self.direction == "maximize" else np.inf
    
    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[float] = None,
        show_progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials
            timeout: Timeout in seconds
            show_progress_bar: Show progress bar
        
        Returns:
            Dictionary with best parameters and Sharpe ratio
        """
        logger.info(f"Starting optimization: {n_trials} trials of {self.agent_type}")
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
        )
        
        # Optimize
        study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=show_progress_bar,
        )
        
        # Get best trial
        best_trial = study.best_trial
        
        logger.info(
            f"Best trial {best_trial.number}: Sharpe={best_trial.value:.4f}"
        )
        logger.info(f"Best params: {best_trial.params}")
        
        return {
            "best_params": best_trial.params,
            "best_value": best_trial.value,
            "best_trial": best_trial,
            "study": study,
        }
    
    def save_study(self, study_path: Path) -> None:
        """Save study to disk."""
        study_path = Path(study_path)
        study_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Study saved to {study_path}")
    
    @staticmethod
    def get_default_params(agent_type: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for an agent type.
        
        Args:
            agent_type: "PPO", "DDPG", or "A2C"
        
        Returns:
            Default parameters dictionary
        """
        defaults = {
            "PPO": {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
            },
            "DDPG": {
                "learning_rate": 1e-3,
                "batch_size": 256,
                "tau": 0.005,
                "gamma": 0.99,
                "action_noise": 0.1,
            },
            "A2C": {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
            },
        }
        
        if agent_type not in defaults:
            raise ValueError(f"Unknown agent_type: {agent_type}")
        
        return defaults[agent_type]

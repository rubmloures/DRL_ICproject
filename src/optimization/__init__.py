"""
Optimization Layer - Bayesian Hyperparameter Tuning
====================================================
Integrates Optuna for automated hyperparameter optimization of DRL agents.

Features:
- Parallel trial execution
- Early stopping (pruning)
- Tree Parzen Estimator sampling
- Support for PPO, DDPG, A2C

Example:
    >>> from src.optimization import HyperparameterOptimizer
    >>> optimizer = HyperparameterOptimizer(
    ...     agent_type="PPO",
    ...     env_fn=lambda: StockTradingEnvB3(...),
    ... )
    >>> best_params = optimizer.optimize(n_trials=100)
"""

from .hyperparameter_optimizer import HyperparameterOptimizer

__all__ = ["HyperparameterOptimizer"]

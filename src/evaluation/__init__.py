"""
Evaluation Framework for DRL Trading Agents

Three-phase evaluation framework:
1. TrainingMonitor: Track convergence during training (TensorBoard)
2. BacktestEvaluator: Comprehensive backtesting with Pyfolio
3. CustomVisualizer: DRL-specific visualizations
"""

from .backtest_evaluator import BacktestEvaluator, BenchmarkData, setup_backtesting
from .custom_visualizer import CustomVisualizer
from .training_monitor import (
    ConvergenceMetrics,
    TrainingMonitor,
    TrainingMonitorCallback,
    setup_training_monitor,
)

__all__ = [
    'TrainingMonitor',
    'ConvergenceMetrics',
    'TrainingMonitorCallback',
    'setup_training_monitor',
    'BacktestEvaluator',
    'BenchmarkData',
    'setup_backtesting',
    'CustomVisualizer',
]

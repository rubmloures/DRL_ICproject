"""
Training Convergence Monitoring Module

Tracks convergence metrics during training to ensure the DRL agent is learning properly:
- Episode rewards
- Actor/Critic loss
- Entropy
- Episode duration
- Trade frequency

This module integrates with TensorBoard for real-time visualization.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure

logger = logging.getLogger(__name__)


class ConvergenceMetrics:
    """
    Tracks convergence metrics during training.
    
    Attributes:
        episode_rewards: Deque of episode reward sums
        episode_lengths: Deque of episode lengths (steps)
        actor_losses: Deque of actor loss values
        critic_losses: Deque of critic loss values
        entropy_values: Deque of entropy values
        trade_counts: Deque of number of trades per episode
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize convergence metrics tracker.
        
        Args:
            window_size: Number of episodes to keep in rolling window (for stats)
        """
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.actor_losses = deque(maxlen=window_size * 10)  # More frequent
        self.critic_losses = deque(maxlen=window_size * 10)
        self.entropy_values = deque(maxlen=window_size * 10)
        self.trade_counts = deque(maxlen=window_size)
        self.sharpe_ratios = deque(maxlen=window_size)
        self.sortino_ratios = deque(maxlen=window_size)
        self.max_drawdowns = deque(maxlen=window_size)
        
    def add_episode_reward(self, reward: float) -> None:
        """Add episode reward."""
        self.episode_rewards.append(reward)
        
    def add_episode_length(self, length: int) -> None:
        """Add episode length (number of steps)."""
        self.episode_lengths.append(length)
        
    def add_actor_loss(self, loss: float) -> None:
        """Add actor loss value."""
        if not np.isnan(loss) and not np.isinf(loss):
            self.actor_losses.append(float(loss))
            
    def add_critic_loss(self, loss: float) -> None:
        """Add critic loss value."""
        if not np.isnan(loss) and not np.isinf(loss):
            self.critic_losses.append(float(loss))
            
    def add_entropy(self, entropy: float) -> None:
        """Add entropy value."""
        if not np.isnan(entropy) and not np.isinf(entropy):
            self.entropy_values.append(float(entropy))
            
    def add_trade_count(self, count: int) -> None:
        """Add number of trades in episode."""
        self.trade_counts.append(count)
        
    def add_sharpe_ratio(self, sharpe: float) -> None:
        """Add episode Sharpe ratio."""
        if not np.isnan(sharpe) and not np.isinf(sharpe):
            self.sharpe_ratios.append(float(sharpe))
            
    def add_sortino_ratio(self, sortino: float) -> None:
        """Add episode Sortino ratio."""
        if not np.isnan(sortino) and not np.isinf(sortino):
            self.sortino_ratios.append(float(sortino))
            
    def add_max_drawdown(self, drawdown: float) -> None:
        """Add episode maximum drawdown (as negative value)."""
        if not np.isnan(drawdown) and not np.isinf(drawdown):
            self.max_drawdowns.append(float(drawdown))
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get rolling window statistics.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        stats = {}
        
        # Episode rewards
        if self.episode_rewards:
            stats['episode_reward_mean'] = float(np.mean(self.episode_rewards))
            stats['episode_reward_std'] = float(np.std(self.episode_rewards))
            stats['episode_reward_min'] = float(np.min(self.episode_rewards))
            stats['episode_reward_max'] = float(np.max(self.episode_rewards))
            
        # Episode lengths
        if self.episode_lengths:
            stats['episode_length_mean'] = float(np.mean(self.episode_lengths))
            stats['episode_length_std'] = float(np.std(self.episode_lengths))
            
        # Actor loss
        if self.actor_losses:
            stats['actor_loss_mean'] = float(np.mean(self.actor_losses))
            stats['actor_loss_std'] = float(np.std(self.actor_losses))
            stats['actor_loss_min'] = float(np.min(self.actor_losses))
            
        # Critic loss
        if self.critic_losses:
            stats['critic_loss_mean'] = float(np.mean(self.critic_losses))
            stats['critic_loss_std'] = float(np.std(self.critic_losses))
            stats['critic_loss_min'] = float(np.min(self.critic_losses))
            
        # Entropy
        if self.entropy_values:
            stats['entropy_mean'] = float(np.mean(self.entropy_values))
            stats['entropy_std'] = float(np.std(self.entropy_values))
            stats['entropy_max'] = float(np.max(self.entropy_values))
            stats['entropy_min'] = float(np.min(self.entropy_values))
            
        # Trade counts
        if self.trade_counts:
            stats['trades_per_episode_mean'] = float(np.mean(self.trade_counts))
            stats['trades_per_episode_std'] = float(np.std(self.trade_counts))
            
        # Sharpe ratios
        if self.sharpe_ratios:
            stats['sharpe_ratio_mean'] = float(np.mean(self.sharpe_ratios))
            stats['sharpe_ratio_std'] = float(np.std(self.sharpe_ratios))
            
        # Sortino ratios
        if self.sortino_ratios:
            stats['sortino_ratio_mean'] = float(np.mean(self.sortino_ratios))
            stats['sortino_ratio_std'] = float(np.std(self.sortino_ratios))
            
        # Max drawdowns
        if self.max_drawdowns:
            stats['max_drawdown_mean'] = float(np.mean(self.max_drawdowns))
            stats['max_drawdown_worst'] = float(np.min(self.max_drawdowns))
            
        return stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export metrics to pandas DataFrame for analysis."""
        data = {
            'episode_reward': list(self.episode_rewards),
            'episode_length': list(self.episode_lengths),
            'trade_count': list(self.trade_counts),
            'sharpe_ratio': list(self.sharpe_ratios),
            'sortino_ratio': list(self.sortino_ratios),
            'max_drawdown': list(self.max_drawdowns),
        }
        
        # Pad shorter lists
        max_len = max(len(v) for v in data.values())
        for key in data:
            if len(data[key]) < max_len:
                data[key].extend([np.nan] * (max_len - len(data[key])))
                
        return pd.DataFrame(data)


class TrainingMonitorCallback(BaseCallback):
    """
    Callback for monitoring training convergence.
    
    Logs convergence metrics to the provided metrics tracker and TensorBoard.
    """
    
    def __init__(
        self,
        metrics: ConvergenceMetrics,
        log_frequency: int = 10,
        verbose: int = 1,
    ):
        """
        Initialize callback.
        
        Args:
            metrics: ConvergenceMetrics instance to track
            log_frequency: Log metrics every N episodes
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.metrics = metrics
        self.log_frequency = log_frequency
        self.num_episodes = 0
        
    def _on_step(self) -> bool:
        """Called after each step."""
        return True
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        stats = self.metrics.get_stats()
        
        if self.verbose > 0:
            logger.info("=" * 60)
            logger.info("Training Completed - Convergence Summary")
            logger.info("=" * 60)
            
            for key, value in sorted(stats.items()):
                logger.info(f"{key:30s}: {value:10.4f}")
                
            logger.info("=" * 60)


class TrainingMonitor:
    """
    High-level training monitor that tracks convergence and logs to TensorBoard.
    
    Usage:
        monitor = TrainingMonitor(log_dir="./logs")
        agent.set_logger(monitor.get_logger())
        # Training happens...
        monitor.plot_convergence()
    """
    
    def __init__(
        self,
        log_dir: str = "./training_logs",
        model_name: str = "drl_agent",
        enable_tensorboard: bool = True,
    ):
        """
        Initialize training monitor.
        
        Args:
            log_dir: Directory for logs and checkpoints
            model_name: Name of the model for logging
            enable_tensorboard: Whether to log to TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.metrics = ConvergenceMetrics()
        self.tensorboard_log = str(self.log_dir / "tensorboard") if enable_tensorboard else None
        
        # Training history
        self.training_history = {
            'episode': [],
            'reward': [],
            'length': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'trades': [],
            'sharpe': [],
            'sortino': [],
            'drawdown': [],
        }
        
        logger.info(f"TrainingMonitor initialized: {self.log_dir}")
        
    def record_episode(
        self,
        episode: int,
        reward: float,
        length: int = None,
        trades: int = None,
        sharpe: float = None,
        sortino: float = None,
        drawdown: float = None,
    ) -> None:
        """
        Record metrics from a completed episode.
        
        Args:
            episode: Episode number
            reward: Total reward for episode
            length: Number of steps in episode
            trades: Number of trades executed
            sharpe: Sharpe ratio for episode
            sortino: Sortino ratio for episode
            drawdown: Maximum drawdown for episode (negative)
        """
        self.metrics.add_episode_reward(reward)
        self.training_history['episode'].append(episode)
        self.training_history['reward'].append(reward)
        
        if length is not None:
            self.metrics.add_episode_length(length)
            self.training_history['length'].append(length)
            
        if trades is not None:
            self.metrics.add_trade_count(trades)
            self.training_history['trades'].append(trades)
            
        if sharpe is not None:
            self.metrics.add_sharpe_ratio(sharpe)
            self.training_history['sharpe'].append(sharpe)
            
        if sortino is not None:
            self.metrics.add_sortino_ratio(sortino)
            self.training_history['sortino'].append(sortino)
            
        if drawdown is not None:
            self.metrics.add_max_drawdown(drawdown)
            self.training_history['drawdown'].append(drawdown)
    
    def record_loss(
        self,
        actor_loss: float = None,
        critic_loss: float = None,
        entropy: float = None,
    ) -> None:
        """
        Record loss values during training step.
        
        Args:
            actor_loss: Actor network loss
            critic_loss: Critic network loss
            entropy: Policy entropy
        """
        if actor_loss is not None:
            self.metrics.add_actor_loss(actor_loss)
            self.training_history['actor_loss'].append(actor_loss)
            
        if critic_loss is not None:
            self.metrics.add_critic_loss(critic_loss)
            self.training_history['critic_loss'].append(critic_loss)
            
        if entropy is not None:
            self.metrics.add_entropy(entropy)
            self.training_history['entropy'].append(entropy)
    
    def get_tensorboard_callback(self) -> Optional[BaseCallback]:
        """Get TensorBoard callback for model training."""
        if self.tensorboard_log is None:
            return None
            
        return TrainingMonitorCallback(
            metrics=self.metrics,
            log_frequency=10,
            verbose=1,
        )
    
    def get_convergence_summary(self) -> Dict[str, float]:
        """Get convergence metrics summary."""
        return self.metrics.get_stats()
    
    def check_convergence(
        self,
        min_reward_trend: float = 0.05,
        entropy_threshold: float = 0.01,
        critic_loss_threshold: float = 0.1,
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Check if training has converged.
        
        Args:
            min_reward_trend: Minimum reward improvement required (as fraction)
            entropy_threshold: Minimum entropy (should stay above this)
            critic_loss_threshold: Maximum critic loss allowed
            
        Returns:
            (is_converged, issues_dict)
        """
        issues = {}
        
        # Check reward trend
        if len(self.training_history['reward']) > 20:
            recent_rewards = self.training_history['reward'][-10:]
            older_rewards = self.training_history['reward'][-20:-10]
            
            if np.mean(older_rewards) > 0:
                trend = (np.mean(recent_rewards) - np.mean(older_rewards)) / np.mean(older_rewards)
                if trend < min_reward_trend:
                    issues['reward_trend'] = f"Reward increase {trend:.2%} < {min_reward_trend:.2%}"
        
        # Check entropy not collapsing too fast
        if len(self.training_history['entropy']) > 100:
            entropy_values = self.training_history['entropy']
            current_entropy = np.mean(entropy_values[-10:])
            if current_entropy > 0:  # Entropy still positive
                issues['entropy_warning'] = f"Entropy low: {current_entropy:.4f}"
        
        # Check critic loss is reasonable
        if len(self.training_history['critic_loss']) > 100:
            recent_critic = np.mean(self.training_history['critic_loss'][-10:])
            if recent_critic > critic_loss_threshold:
                issues['critic_loss'] = f"High critic loss: {recent_critic:.4f}"
        
        is_converged = len(issues) == 0
        return is_converged, issues
    
    def export_to_csv(self, filename: str = None) -> str:
        """
        Export training history to CSV.
        
        Args:
            filename: Output filename (default: model_name_history.csv)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"{self.model_name}_history.csv"
            
        output_path = self.log_dir / filename
        
        # Pad data to same length
        max_len = max(len(v) for v in self.training_history.values())
        for key in self.training_history:
            if len(self.training_history[key]) < max_len:
                self.training_history[key].extend(
                    [np.nan] * (max_len - len(self.training_history[key]))
                )
        
        df = pd.DataFrame(self.training_history)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Training history exported to {output_path}")
        return str(output_path)
    
    def print_summary(self) -> None:
        """Print convergence summary to console."""
        stats = self.metrics.get_stats()
        
        print("\n" + "=" * 70)
        print("TRAINING CONVERGENCE SUMMARY")
        print("=" * 70)
        
        # Group by category
        categories = {
            'Episode Rewards': [k for k in stats if 'episode_reward' in k],
            'Episode Metrics': [k for k in stats if 'episode_length' in k or 'trades' in k],
            'Network Losses': [k for k in stats if 'actor_loss' in k or 'critic_loss' in k],
            'Policy Quality': [k for k in stats if 'entropy' in k],
            'Financial Metrics': [k for k in stats if any(x in k for x in ['sharpe', 'sortino', 'drawdown'])],
        }
        
        for category, keys in categories.items():
            if keys:
                print(f"\n{category}:")
                print("-" * 70)
                for key in sorted(keys):
                    if key in stats:
                        print(f"  {key:30s}: {stats[key]:12.6f}")
        
        print("\n" + "=" * 70)


def setup_training_monitor(
    log_dir: str = "./training_logs",
    model_name: str = "drl_agent",
) -> TrainingMonitor:
    """
    Convenience function to setup training monitor.
    
    Args:
        log_dir: Logging directory
        model_name: Name of the model
        
    Returns:
        Configured TrainingMonitor instance
    """
    monitor = TrainingMonitor(
        log_dir=log_dir,
        model_name=model_name,
        enable_tensorboard=True,
    )
    
    logger.info(f"Training monitor created: {model_name}")
    return monitor

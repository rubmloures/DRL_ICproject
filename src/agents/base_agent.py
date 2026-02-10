"""
Base DRL Agent
==============
Abstract base class for all DRL agents.
Provides common interface for training, evaluation, and prediction.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseDRLAgent(ABC):
    """
    Abstract base class for DRL agents.
    
    Defines the common interface for:
    - Training on environments
    - Making predictions
    - Saving/loading models
    - Evaluation
    
    All specific algorithms (PPO, DDPG, A2C) should inherit this class.
    """
    
    def __init__(
        self,
        env: Any,
        model_name: str,
        device: str = "cpu",
        verbose: int = 0,
    ):
        """
        Initialize base agent.
        
        Args:
            env: Gymnasium trading environment
            model_name: Name identifier for this model
            device: Device to train on ('cpu', 'cuda')
            verbose: Logging verbosity
        """
        self.env = env
        self.model_name = model_name
        self.device = device
        self.verbose = verbose
        self.model = None
        
        logger.debug(f"Initialized BaseDRLAgent: {model_name}")
    
    @abstractmethod
    def train(
        self,
        total_timesteps: int,
        callback_type: Optional[str] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        """
        Train the agent.
        
        Args:
            total_timesteps: Total training steps
            callback_type: Callback type ('callback' or None)
            save_dir: Directory to save checkpoints
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Make prediction (action) for an observation.
        
        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy
        
        Returns:
            (action, state)
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass
    
    def evaluate(
        self,
        env: Any,
        num_episodes: int = 5,
        deterministic: bool = True,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate agent on an environment.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
            render: Render environment during evaluation
        
        Returns:
            Metrics dictionary
        """
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    env.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
        }
        
        logger.info(
            f"Evaluation: mean_reward={metrics['mean_reward']:.4f} "
            f"(±{metrics['std_reward']:.4f})"
        )
        
        return metrics
    
    def get_model(self) -> Any:
        """Get underlying model (for advanced usage)."""
        return self.model

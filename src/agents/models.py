"""
PPO Agent with PINN Integration
================================
Deep Reinforcement Learning agent using PPO with PINN feature extraction.
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.hyperparameters import FEATURE_EXTRACTOR_PARAMS, PPO_PARAMS
from src.pinn.model import DeepHestonHybrid


class PINNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that incorporates PINN embeddings.
    
    Architecture:
    - Input layer processes raw observations
    - PINN embeddings are concatenated
    - Hidden layers with ReLU activation
    - Output: feature vector for policy/value heads
    """
    
    def __init__(
        self,
        observation_space,
        pinn_model: Optional[DeepHestonHybrid] = None,
        net_arch: List[int] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize feature extractor.
        
        Args:
            observation_space: Gymnasium observation space
            pinn_model: Pre-trained PINN model
            net_arch: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__(observation_space, features_dim=1)
        
        self.pinn_model = pinn_model
        net_arch = net_arch or FEATURE_EXTRACTOR_PARAMS.get("net_arch", [256, 256])
        
        # Get input dimension
        input_dim = observation_space.shape[0]
        
        # Build feature extraction network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in net_arch:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        output_dim = net_arch[-1] if net_arch else input_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.feature_network = nn.Sequential(*layers)
        self.features_dim = output_dim
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observations.
        
        Args:
            observations: Raw observations from environment
        
        Returns:
            Extracted features
        """
        features = self.feature_network(observations)
        return features


class PPOPINNPolicy(ActorCriticPolicy):
    """Custom policy class that uses PINNFeatureExtractor."""
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        pinn_model: Optional[DeepHestonHybrid] = None,
        net_arch: Optional[Dict[str, List[int]]] = None,
        **kwargs,
    ):
        """
        Initialize PPO policy with PINN integration.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            lr_schedule: Learning rate schedule
            pinn_model: Pre-trained PINN model
            net_arch: Network architecture specification
            **kwargs: Additional arguments for ActorCriticPolicy
        """
        # Use PINNFeatureExtractor
        if net_arch is None:
            net_arch = {
                "pi": FEATURE_EXTRACTOR_PARAMS.get("net_arch", [256, 256]),
                "vf": FEATURE_EXTRACTOR_PARAMS.get("net_arch", [256, 256]),
            }
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            **kwargs,
        )
        
        self.pinn_model = pinn_model
    
    def _build_mlp_extractor(self) -> None:
        """Override to use PINNFeatureExtractor."""
        self.mlp_extractor = PINNFeatureExtractor(
            self.observation_space,
            pinn_model=self.pinn_model,
        )


class PPOPINNAgent:
    """
    PPO agent with PINN feature extraction for stock trading.
    
    Combines:
    - PPO algorithm from stable-baselines3
    - PINN-extracted features for physics-informed decision making
    - Custom reward shaping for trading
    """
    
    def __init__(
        self,
        env,
        pinn_model: Optional[DeepHestonHybrid] = None,
        ppo_params: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        verbose: int = 1,
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: Trading environment
            pinn_model: Pre-trained PINN model
            ppo_params: PPO hyperparameters
            device: Device to train on ('cpu' or 'cuda')
            verbose: Verbosity level
        """
        self.env = env
        self.pinn_model = pinn_model
        self.ppo_params = ppo_params or PPO_PARAMS.copy()
        self.device = device
        self.verbose = verbose
        
        # Initialize PPO agent
        self.model = PPO(
            "MlpPolicy",
            env,
            **self.ppo_params,
            device=device,
            tensorboard_log="./tensorbard_logs/",
            verbose=verbose,
        )
        
        # Integrate PINN if provided
        if pinn_model is not None:
            self._integrate_pinn()
    
    def _integrate_pinn(self) -> None:
        """Integrate PINN features into the policy."""
        # For now, PINN features are processed in the environment observation
        # In advanced implementation, could modify the feature extractor directly
        if self.pinn_model is not None:
            self.pinn_model = self.pinn_model.to(self.device)
            self.pinn_model.eval()
    
    def train(
        self,
        total_timesteps: int = 100_000,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Train the PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_dir: Directory to save checkpoints
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            callback=None,  # Can add custom callbacks here
        )
        
        if save_dir:
            save_path = Path(save_dir) / "ppo_agent"
            self.model.save(str(save_path))
            print(f"Model saved to {save_path}")
    
    def predict(
        self,
        obs: Any,
        deterministic: bool = True,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Predict action for observation.
        
        Args:
            obs: Observation from environment
            deterministic: Whether to use deterministic policy
        
        Returns:
            action, state
        """
        action, state = self.model.predict(obs, deterministic=deterministic)
        return action, state
    
    def evaluate(
        self,
        env,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> Tuple[float, float]:
        """
        Evaluate agent on environment.
        
        Args:
            env: Evaluation environment
            n_episodes: Number of episodes
            deterministic: Use deterministic policy
        
        Returns:
            mean_reward, std_reward
        """
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
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
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward, std_reward
    
    def get_model(self) -> PPO:
        """Get underlying PPO model."""
        return self.model
    
    def save(self, path: str) -> None:
        """Save agent to disk."""
        self.model.save(path)
        if self.pinn_model is not None:
            pinn_path = path.replace(".zip", "_pinn.pt")
            torch.save(self.pinn_model.state_dict(), pinn_path)
    
    def load(self, path: str) -> None:
        """Load agent from disk."""
        self.model = PPO.load(path, env=self.env, device=self.device)
        if Path(path.replace(".zip", "_pinn.pt")).exists():
            pinn_path = path.replace(".zip", "_pinn.pt")
            if self.pinn_model is not None:
                self.pinn_model.load_state_dict(torch.load(pinn_path))


import numpy as np

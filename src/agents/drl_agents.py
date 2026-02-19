"""
DRL Agents Implementation
=========================
Concrete implementations of PPO, DDPG, and A2C agents using Stable-Baselines3.
"""

import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.noise import NormalActionNoise

from .base_agent import BaseDRLAgent

logger = logging.getLogger(__name__)


class PPOAgent(BaseDRLAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    
    Best for:
    - Continuous action spaces
    - Sample efficiency
    - Stable training
    
    Paper: https://arxiv.org/abs/1707.06347
    """
    
    def __init__(
        self,
        env: Any,
        model_name: str = "ppo",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PPO agent.
        
        Args:
            env: Gymnasium environment
            model_name: Model identifier
            learning_rate: Learning rate
            n_steps: Rollout buffer size
            batch_size: Mini-batch size
            n_epochs: Update epochs per rollout
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clip parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Gradient clipping norm
            device: 'cpu' or 'cuda'
            verbose: Verbosity level
            tensorboard_log: TensorBoard log directory
            policy_kwargs: Custom policy keywords
        """
        super().__init__(env, model_name, device, verbose)
        
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
        )
        
        logger.info(f"Created PPO agent with learning_rate={learning_rate}")
    
    def train(
        self,
        total_timesteps: int,
        callback_type: Optional[str] = None,
        save_dir: Optional[Path] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
    ) -> None:
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total training timesteps
            callback_type: Type of callback ('callback' or None)
            save_dir: Directory for checkpoints
            eval_freq: Evaluation frequency (steps)
            n_eval_episodes: Episodes per evaluation
        """
        callback = None
        
        if callback_type == "callback" and save_dir:
            callback = EvalCallback(
                eval_env=self.env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                log_path=str(save_dir),
                best_model_save_path=str(save_dir),
                deterministic=True,
                render=False,
            )
        
        logger.info(f"Training PPO for {total_timesteps} steps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        
        if save_dir:
            self.save(Path(save_dir) / f"{self.model_name}_final")
    
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict action for observation."""
        action, _state = self.model.predict(obs, deterministic=deterministic)
        return action, _state
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Saved PPO model to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model = PPO.load(str(path), env=self.env, device=self.device)
        logger.info(f"Loaded PPO model from {path}")


class DDPGAgent(BaseDRLAgent):
    """
    Deep Deterministic Policy Gradient (DDPG) agent.
    
    Best for:
    - Continuous action spaces
    - Off-policy efficiency
    - Continuous control
    
    Paper: https://arxiv.org/abs/1509.02971
    """
    
    def __init__(
        self,
        env: Any,
        model_name: str = "ddpg",
        learning_rate: float = 1e-3,
        buffer_size: int = 100_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        action_noise: Optional[float] = 0.1,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize DDPG agent.
        
        Args:
            env: Gymnasium environment
            model_name: Model identifier
            learning_rate: Learning rate
            buffer_size: Experience replay buffer size
            learning_starts: Steps before learning
            batch_size: Mini-batch size
            tau: Target network update rate
            gamma: Discount factor
            action_noise: Exploration noise std
            device: 'cpu' or 'cuda'
            verbose: Verbosity level
            tensorboard_log: TensorBoard log directory
            policy_kwargs: Custom policy keywords
        """
        super().__init__(env, model_name, device, verbose)
        
        # Action noise for exploration
        n_actions = env.action_space.shape[0]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise * np.ones(n_actions)
        ) if action_noise else None
        
        self.model = DDPG(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            action_noise=noise,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
        )
        
        logger.info(f"Created DDPG agent with learning_rate={learning_rate}")
    
    def train(
        self,
        total_timesteps: int,
        callback_type: Optional[str] = None,
        save_dir: Optional[Path] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
    ) -> None:
        """
        Train DDPG agent.
        
        Args:
            total_timesteps: Total training timesteps
            callback_type: Type of callback
            save_dir: Directory for checkpoints
            eval_freq: Evaluation frequency (steps)
            n_eval_episodes: Episodes per evaluation
        """
        callback = None
        
        if callback_type == "callback" and save_dir:
            callback = EvalCallback(
                eval_env=self.env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                log_path=str(save_dir),
                best_model_save_path=str(save_dir),
                deterministic=True,
                render=False,
            )
        
        logger.info(f"Training DDPG for {total_timesteps} steps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        
        if save_dir:
            self.save(Path(save_dir) / f"{self.model_name}_final")
    
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict action for observation."""
        action, _state = self.model.predict(obs, deterministic=deterministic)
        return action, _state
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Saved DDPG model to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model = DDPG.load(str(path), env=self.env, device=self.device)
        logger.info(f"Loaded DDPG model from {path}")


class A2CAgent(BaseDRLAgent):
    """
    Advantage Actor-Critic (A2C) agent.
    
    Best for:
    - Fast training
    - Parallel environments
    - On-policy learning
    
    Paper: https://arxiv.org/abs/1602.01783
    """
    
    def __init__(
        self,
        env: Any,
        model_name: str = "a2c",
        learning_rate: float = 7e-4,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_rms_prop: bool = True,
        device: str = "auto",
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize A2C agent.
        
        Args:
            env: Gymnasium environment
            model_name: Model identifier
            learning_rate: Learning rate
            n_steps: Steps per update
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Gradient clipping norm
            use_rms_prop: Use RMSProp optimizer
            device: 'cpu' or 'cuda'
            verbose: Verbosity level
            tensorboard_log: TensorBoard log directory
            policy_kwargs: Custom policy keywords
        """
        super().__init__(env, model_name, device, verbose)
        
        self.model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_rms_prop=use_rms_prop,
            device=device,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
        )
        
        logger.info(f"Created A2C agent with learning_rate={learning_rate}")
    
    def train(
        self,
        total_timesteps: int,
        callback_type: Optional[str] = None,
        save_dir: Optional[Path] = None,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
    ) -> None:
        """
        Train A2C agent.
        
        Args:
            total_timesteps: Total training timesteps
            callback_type: Type of callback
            save_dir: Directory for checkpoints
            eval_freq: Evaluation frequency (steps)
            n_eval_episodes: Episodes per evaluation
        """
        callback = None
        
        if callback_type == "callback" and save_dir:
            callback = EvalCallback(
                eval_env=self.env,
                n_eval_episodes=n_eval_episodes,
                eval_freq=eval_freq,
                log_path=str(save_dir),
                best_model_save_path=str(save_dir),
                deterministic=True,
                render=False,
            )
        
        logger.info(f"Training A2C for {total_timesteps} steps")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False,
        )
        
        if save_dir:
            self.save(Path(save_dir) / f"{self.model_name}_final")
    
    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """Predict action for observation."""
        action, _state = self.model.predict(obs, deterministic=deterministic)
        return action, _state
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Saved A2C model to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        self.model = A2C.load(str(path), env=self.env, device=self.device)
        logger.info(f"Loaded A2C model from {path}")

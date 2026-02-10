"""
Ensemble Agent Strategy
=======================

Combines decisions from multiple DRL agents (PPO, DDPG, A2C) using different voting strategies:
- mean: Simple average of actions
- weighted: Actions weighted by agent performance
- majority: Discrete voting (buy/hold/sell)
- best: Select best performing agent

Example:
    >>> ppo = PPOAgent(env=env)
    >>> ddpg = DDPGAgent(env=env)
    >>> a2c = A2CAgent(env=env)
    >>> ensemble = EnsembleAgent(
    ...     env=env,
    ...     agents={'PPO': ppo, 'DDPG': ddpg, 'A2C': a2c},
    ...     voting_strategy='weighted'
    ... )
    >>> ensemble.train(total_timesteps=50_000)
    >>> action, info = ensemble.predict(obs)
"""

import logging
import json
import os
from typing import Dict, Any
import numpy as np
import pandas as pd

from src.agents.base_agent import BaseDRLAgent

logger = logging.getLogger(__name__)


class EnsembleAgent(BaseDRLAgent):
    """
    Combines predictions from multiple DRL agents.
    
    Parameters
    ----------
    env : gym.Env
        Trading environment
    agents : dict
        Dictionary mapping agent names to BaseDRLAgent instances
        Example: {'PPO': ppo_agent, 'DDPG': ddpg_agent, 'A2C': a2c_agent}
    voting_strategy : str, default='weighted'
        Strategy for combining predictions:
        - 'mean': Simple arithmetic mean
        - 'weighted': Weighted by performance history
        - 'majority': Discrete voting (sign-based)
        - 'best': Use only best performing agent
    
    Attributes
    ----------
    agent_weights : dict
        Weights for each agent in weighted voting
    voting_history : list
        History of voting decisions for analysis
    
    Example
    -------
    >>> ensemble = EnsembleAgent(env, agents, voting_strategy='weighted')
    >>> ensemble.train(total_timesteps=100_000)
    >>> ensemble.set_agent_weights({'PPO': 0.5, 'DDPG': 0.3, 'A2C': 0.2})
    >>> metrics = ensemble.evaluate(n_episodes=10)
    """
    
    def __init__(self, env, agents: Dict[str, BaseDRLAgent], 
                 voting_strategy: str = 'weighted', model_name: str = "ensemble",
                 device: str = "cpu", verbose: int = 0):
        """Initialize ensemble agent."""
        super().__init__(env, model_name=model_name, device=device, verbose=verbose)
        
        if not agents:
            raise ValueError("At least one agent must be provided")
        
        self.agents = agents
        self.voting_strategy = voting_strategy
        self.n_agents = len(agents)
        
        # Initialize equal weights
        self.agent_weights = {name: 1.0 / self.n_agents for name in agents.keys()}
        
        # Track voting decisions
        self.voting_history = []
        
        logger.info(f"EnsembleAgent initialized with {self.n_agents} agents:")
        for name in agents.keys():
            logger.info(f"  - {name}")
        logger.info(f"Voting strategy: {voting_strategy}")
    
    def train(self, total_timesteps: int, save_dir: str = None, **kwargs) -> None:
        """
        Train all agents on the same data.
        
        Each agent is trained independently with its own learning dynamics,
        then combined during prediction.
        
        Parameters
        ----------
        total_timesteps : int
            Number of timesteps for training
        save_dir : str, optional
            Directory to save trained models
        **kwargs
            Additional arguments passed to agent.train()
        
        Example
        -------
        >>> ensemble.train(total_timesteps=100_000, save_dir='trained_models/ensemble/')
        """
        logger.info(f"Training {self.n_agents} agents for {total_timesteps} timesteps...")
        
        for agent_name, agent in self.agents.items():
            logger.info(f"Training {agent_name}...")
            agent.train(total_timesteps=total_timesteps, **kwargs)
            
            # Save individual agent
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                agent_path = os.path.join(save_dir, f"{agent_name}_model")
                agent.save(agent_path)
                logger.info(f"  Saved {agent_name} to {agent_path}")
        
        logger.info("Training complete for all agents")
    
    def predict(self, observation, deterministic: bool = False):
        """
        Make ensemble prediction by combining all agent predictions.
        
        Parameters
        ----------
        observation : np.ndarray
            Current observation from environment
        deterministic : bool, default=False
            Use deterministic policy (for testing)
        
        Returns
        -------
        action : np.ndarray
            Combined action
        metadata : dict
            Information about predictions:
            - agent_predictions: dict with each agent's action
            - voting_strategy: strategy used
            - agent_weights: weights applied
        
        Example
        -------
        >>> action, info = ensemble.predict(obs, deterministic=True)
        >>> print(f"Action: {action}")
        >>> print(f"Agent predictions: {info['agent_predictions']}")
        """
        # Get predictions from all agents
        predictions = {}
        
        for agent_name, agent in self.agents.items():
            action, _ = agent.predict(observation, deterministic=deterministic)
            predictions[agent_name] = action
        
        # Apply voting strategy
        if self.voting_strategy == 'mean':
            combined_action = self._mean_voting(predictions)
        
        elif self.voting_strategy == 'weighted':
            combined_action = self._weighted_voting(predictions)
        
        elif self.voting_strategy == 'majority':
            combined_action = self._majority_voting(predictions)
        
        elif self.voting_strategy == 'best':
            combined_action = self._best_voting(predictions)
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        # Store voting record
        self.voting_history.append({
            'predictions': predictions,
            'combined': combined_action,
            'strategy': self.voting_strategy,
            'weights': self.agent_weights.copy(),
        })
        
        # Return action and metadata
        metadata = {
            'agent_predictions': predictions,
            'voting_strategy': self.voting_strategy,
            'agent_weights': self.agent_weights,
        }
        
        return combined_action, metadata
    
    def _mean_voting(self, predictions: Dict) -> np.ndarray:
        """
        Simple average of all agent predictions.
        
        action_ensemble = (action_1 + action_2 + ... + action_n) / n
        """
        stacked = np.stack(list(predictions.values()))
        return np.mean(stacked, axis=0)
    
    def _weighted_voting(self, predictions: Dict) -> np.ndarray:
        """
        Weighted average by agent performance.
        
        action_ensemble = sum(weight_i * action_i) / sum(weight_i)
        """
        total_weight = sum(self.agent_weights.values())
        weighted_action = np.zeros_like(list(predictions.values())[0], dtype=np.float32)
        
        for agent_name, action in predictions.items():
            weight = self.agent_weights[agent_name] / total_weight
            weighted_action += weight * action
        
        return weighted_action
    
    def _majority_voting(self, predictions: Dict) -> np.ndarray:
        """
        Discrete voting: each agent votes buy (+1), hold (0), or sell (-1).
        
        For continuous action space, discretize first:
        - Buy: action > 0.5
        - Sell: action < -0.5
        - Hold: -0.5 <= action <= 0.5
        """
        # Discretize actions
        actions_discrete = {}
        for agent_name, action in predictions.items():
            discrete = np.where(np.abs(action) < 0.5, 0, np.sign(action))
            actions_discrete[agent_name] = discrete
        
        # Majority vote per action dimension
        combined = np.zeros_like(list(predictions.values())[0], dtype=np.float32)
        
        for action_dim in range(combined.shape[0]):
            votes = [actions_discrete[agent][action_dim] for agent in actions_discrete.keys()]
            vote_result = np.sign(np.sum(votes))
            combined[action_dim] = vote_result
        
        return combined
    
    def _best_voting(self, predictions: Dict) -> np.ndarray:
        """
        Use only the best performing agent.
        
        Best agent determined by highest weight.
        """
        best_agent = max(self.agent_weights.items(), key=lambda x: x[1])[0]
        return predictions[best_agent]
    
    def set_agent_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for individual agents (used in weighted voting).
        
        Robustly handles edge cases like all-zero or negative weights.
        
        Parameters
        ----------
        weights : dict
            Dictionary mapping agent name to weight (will be normalized to sum to 1.0)
        
        Raises:
            ValueError: If weights dict doesn't contain all agents
        
        Example
        -------
        >>> # Set weights based on Sharpe ratio (automatically normalized)
        >>> ppo_sharpe = 0.50
        >>> ddpg_sharpe = 0.30
        >>> a2c_sharpe = 0.20
        >>> ensemble.set_agent_weights({
        ...     'PPO': ppo_sharpe,
        ...     'DDPG': ddpg_sharpe,
        ...     'A2C': a2c_sharpe
        ... })  # Auto-normalizes to sum=1.0
        
        >>> # Edge case: All negative Sharpe ratios (all agents underperformed)
        >>> ensemble.set_agent_weights({
        ...     'PPO': -0.10,
        ...     'DDPG': -0.05,
        ...     'A2C': -0.15
        ... })  # Falls back to uniform equal weights
        """
        # Validate that all agents are represented
        missing_agents = set(self.agents.keys()) - set(weights.keys())
        if missing_agents:
            raise ValueError(
                f"Weights missing for agents: {missing_agents}. "
                f"All agents must be in weights dict: {list(self.agents.keys())}"
            )
        
        # Robust normalization: handle all-zero and negative weights
        total_weight = sum(weights.values())
        
        if total_weight <= 0:
            # All agents have non-positive weights (e.g., all negative Sharpe)
            logger.warning(
                f"⚠️ All agent weights are non-positive (sum={total_weight:.4f}). "
                f"Falling back to uniform equal weights (1/{self.n_agents} each)."
            )
            self.agent_weights = {name: 1.0 / self.n_agents for name in self.agents}
        else:
            # Normalize to sum to 1.0
            self.agent_weights = {name: w / total_weight for name, w in weights.items()}
            
            # Warn if weights were significantly off
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(
                    f"Weights sum to {total_weight:.4f}. "
                    f"Normalized to sum=1.0."
                )
        
        # Log final weights
        logger.info(f"✓ Agent weights updated:")
        for agent_name in sorted(self.agent_weights.keys()):
            logger.info(f"  {agent_name}: {self.agent_weights[agent_name]:.4f}")
    
    
    def evaluate(self, n_episodes: int, env=None):
        """
        Evaluate ensemble on multiple episodes.
        
        Parameters
        ----------
        n_episodes : int
            Number of episodes to run
        env : gym.Env, optional
            Environment to evaluate on (default: self.env)
        
        Returns
        -------
        dict
            Metrics: mean_reward, std_reward, min_reward, max_reward
        
        Example
        -------
        >>> metrics = ensemble.evaluate(n_episodes=10, env=test_env)
        >>> print(f"Sharpe: {metrics['mean_reward']:.4f}")
        """
        if env is None:
            env = self.env
        
        episode_rewards = []
        
        logger.info(f"Evaluating ensemble on {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            step = 0
            
            while not done:
                # Use deterministic actions during evaluation
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step += 1
            
            episode_rewards.append(episode_reward)
            logger.debug(f"Episode {episode}: reward={episode_reward:.4f}, steps={step}")
        
        # Compute metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'n_episodes': n_episodes,
        }
        
        logger.info(f"Evaluation complete:")
        logger.info(f"  Mean reward: {metrics['mean_reward']:.4f}")
        logger.info(f"  Std reward: {metrics['std_reward']:.4f}")
        logger.info(f"  Min/Max: {metrics['min_reward']:.4f} / {metrics['max_reward']:.4f}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save all agent models and ensemble configuration.
        
        Parameters
        ----------
        path : str
            Directory to save to
        
        Example
        -------
        >>> ensemble.save('trained_models/ensemble/')
        """
        os.makedirs(path, exist_ok=True)
        
        # Save individual agents
        for agent_name, agent in self.agents.items():
            agent_path = os.path.join(path, agent_name)
            agent.save(agent_path)
            logger.info(f"Saved {agent_name} to {agent_path}")
        
        # Save ensemble configuration
        config = {
            'voting_strategy': self.voting_strategy,
            'agent_weights': self.agent_weights,
            'n_agents': self.n_agents,
            'agent_names': list(self.agents.keys()),
        }
        
        config_path = os.path.join(path, 'ensemble_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved ensemble configuration to {config_path}")
    
    def load(self, path: str) -> None:
        """
        Load all agent models and ensemble configuration.
        
        Parameters
        ----------
        path : str
            Directory to load from
        
        Example
        -------
        >>> ensemble.load('trained_models/ensemble/')
        """
        # Load ensemble configuration
        config_path = os.path.join(path, 'ensemble_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.voting_strategy = config.get('voting_strategy', 'weighted')
            self.agent_weights = config.get('agent_weights', self.agent_weights)
            logger.info(f"Loaded ensemble configuration from {config_path}")
        
        # Load individual agents
        for agent_name in self.agents.keys():
            agent_path = os.path.join(path, agent_name)
            self.agents[agent_name].load(agent_path)
            logger.info(f"Loaded {agent_name} from {agent_path}")
    
    def get_voting_summary(self) -> pd.DataFrame:
        """
        Analyze voting patterns from history.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics of voting decisions
        
        Example
        -------
        >>> summary = ensemble.get_voting_summary()
        >>> print(summary)
        """
        if not self.voting_history:
            logger.warning("No voting history available")
            return None
        
        # Analyze agreement between agents
        agreement_count = 0
        total_votes = len(self.voting_history)
        
        for vote_record in self.voting_history:
            predictions = vote_record['predictions']
            actions = list(predictions.values())
            
            # Check if all agents agree (within tolerance)
            if np.allclose(actions[0], actions[1:], atol=0.1):
                agreement_count += 1
        
        agreement_rate = agreement_count / total_votes if total_votes > 0 else 0
        
        summary = pd.DataFrame({
            'metric': ['Total Votes', 'Agreement Count', 'Agreement Rate', 'Voting Strategy'],
            'value': [total_votes, agreement_count, f'{agreement_rate:.2%}', self.voting_strategy]
        })
        
        return summary

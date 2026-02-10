"""
Agent Layer - Deep Reinforcement Learning Algorithms
=====================================================
Implements state-of-the-art DRL algorithms (PPO, DDPG, A2C) using Stable-Baselines3.

Compatible with any Gymnasium environment, designed for stock trading.

Key Algorithms:
- PPO (Proximal Policy Optimization): Best for continuous control, stable training
- DDPG (Deep Deterministic Policy Gradient): Off-policy, sample efficient
- A2C (Advantage Actor-Critic): Fast training, supports parallelization
- EnsembleAgent: Combines multiple agents with voting strategies

Ensemble Voting Strategies:
- mean: Simple average of actions
- weighted: Weighted by agent performance
- majority: Discrete voting (buy/hold/sell)
- best: Select best performing agent
"""

from .base_agent import BaseDRLAgent
from .drl_agents import PPOAgent, DDPGAgent, A2CAgent
from .ensemble_agent import EnsembleAgent
from .models import PPOPINNAgent

__all__ = [
    "BaseDRLAgent",
    "PPOAgent",
    "DDPGAgent", 
    "A2CAgent",
    "EnsembleAgent",
    "PPOPINNAgent",
]
